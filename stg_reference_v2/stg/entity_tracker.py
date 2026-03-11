from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover - optional dependency
    linear_sum_assignment = None  # type: ignore

from .config import EntityMatchingConfig
from .utils import (
    EmbeddingManager,
    box_center,
    compute_displacement,
    compute_iou_matrix,
    normalize_attributes,
)


ACTIVE = "active"
INACTIVE = "inactive"
DISAPPEARED = "disappeared"


@dataclass
class EntityRecord:
    entity_id: str
    label: str
    tag: str
    first_frame: int
    last_frame: int
    first_bbox: List[float]
    last_bbox: List[float]
    first_object: Dict[str, Any]
    last_object: Dict[str, Any]
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    attributes_history: List[Dict[str, Any]] = field(default_factory=list)
    relations_history: List[Dict[str, Any]] = field(default_factory=list)
    status_history: List[Dict[str, Any]] = field(default_factory=list)
    active: bool = True
    state: str = ACTIVE
    missed_frames: int = 0
    disappeared_frame: int | None = None

    def mark_state(self, state: str, frame_index: int) -> None:
        if self.state == state:
            return
        self.state = state
        self.active = state == ACTIVE
        self.status_history.append({"frame_index": frame_index, "state": state})

    def update(self, obj: Dict[str, Any], frame_index: int) -> None:
        was_inactive = self.state == INACTIVE
        self.last_frame = frame_index
        self.last_bbox = list(map(float, obj["bbox"]))
        self.last_object = copy.deepcopy(obj)
        self.missed_frames = 0
        self.active = True
        self.disappeared_frame = None
        if was_inactive:
            self.mark_state(ACTIVE, frame_index)
        elif not self.status_history:
            self.status_history.append({"frame_index": frame_index, "state": ACTIVE})
        self.trajectory.append(
            {
                "frame_index": frame_index,
                "center": box_center(obj["bbox"]),
                "bbox": list(map(float, obj["bbox"])),
            }
        )
        self.attributes_history.append(
            {"frame_index": frame_index, "attributes": normalize_attributes(obj.get("attributes", []))}
        )
        self.relations_history.append(
            {"frame_index": frame_index, "relations": copy.deepcopy(obj.get("relations", []))}
        )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "tag": self.tag,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_bbox": list(self.first_bbox),
            "last_bbox": list(self.last_bbox),
            "first_object": copy.deepcopy(self.first_object),
            "last_object": copy.deepcopy(self.last_object),
            "trajectory": copy.deepcopy(self.trajectory),
            "attributes_history": copy.deepcopy(self.attributes_history),
            "relations_history": copy.deepcopy(self.relations_history),
            "status_history": copy.deepcopy(self.status_history),
            "active": self.active,
            "state": self.state,
            "missed_frames": self.missed_frames,
            "disappeared_frame": self.disappeared_frame,
        }


@dataclass
class MatchResult:
    entity_id: str
    prev_snapshot: Dict[str, Any]
    curr_object: Dict[str, Any]
    score: float
    iou: float
    label_similarity: float
    was_reactivated: bool = False


@dataclass
class FrameAssociationResult:
    frame_index: int
    matched: List[MatchResult] = field(default_factory=list)
    new_entities: List[EntityRecord] = field(default_factory=list)
    disappeared_entities: List[Dict[str, Any]] = field(default_factory=list)
    current_active_records: List[EntityRecord] = field(default_factory=list)
    is_first_frame: bool = False


class EntityTracker:
    def __init__(self, config: EntityMatchingConfig, embedder: EmbeddingManager):
        self.config = config
        self.embedder = embedder
        self.registry: Dict[str, EntityRecord] = {}
        self._next_entity_idx = 1

    def reset(self) -> None:
        self.registry.clear()
        self._next_entity_idx = 1

    def _allocate_entity_id(self) -> str:
        entity_id = f"entity_{self._next_entity_idx:04d}"
        self._next_entity_idx += 1
        return entity_id

    def _register_new_entity(self, obj: Dict[str, Any], frame_index: int) -> EntityRecord:
        entity_id = self._allocate_entity_id()
        label = str(obj.get("label", obj.get("category", "object")))
        tag = str(obj.get("tag", obj.get("name", entity_id)))
        bbox = list(map(float, obj["bbox"]))
        record = EntityRecord(
            entity_id=entity_id,
            label=label,
            tag=tag,
            first_frame=frame_index,
            last_frame=frame_index,
            first_bbox=bbox,
            last_bbox=bbox,
            first_object=copy.deepcopy(obj),
            last_object=copy.deepcopy(obj),
            status_history=[{"frame_index": frame_index, "state": ACTIVE}],
        )
        record.update(obj, frame_index)
        self.registry[entity_id] = record
        return record

    def _candidate_records(self) -> List[EntityRecord]:
        return [record for record in self.registry.values() if record.state != DISAPPEARED]

    def _compute_label_similarity_matrix(
        self,
        prev_records: Sequence[EntityRecord],
        curr_objects: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        if not prev_records or not curr_objects:
            return np.zeros((len(prev_records), len(curr_objects)), dtype=np.float32)
        prev_labels = [self.embedder.label_embedding(record.label) for record in prev_records]
        curr_labels = [
            self.embedder.label_embedding(str(obj.get("label", obj.get("category", "object"))))
            for obj in curr_objects
        ]
        matrix = np.zeros((len(prev_records), len(curr_objects)), dtype=np.float32)
        for i, prev_vec in enumerate(prev_labels):
            for j, curr_vec in enumerate(curr_labels):
                matrix[i, j] = float(np.dot(prev_vec, curr_vec))
        return matrix

    def _compute_tag_match_matrix(
        self,
        prev_records: Sequence[EntityRecord],
        curr_objects: Sequence[Dict[str, Any]],
    ) -> np.ndarray:
        if not prev_records or not curr_objects:
            return np.zeros((len(prev_records), len(curr_objects)), dtype=np.float32)
        matrix = np.zeros((len(prev_records), len(curr_objects)), dtype=np.float32)
        for i, record in enumerate(prev_records):
            for j, obj in enumerate(curr_objects):
                matrix[i, j] = 1.0 if str(record.tag) == str(obj.get("tag", "")) else 0.0
        return matrix

    def _hungarian(self, score_matrix: np.ndarray) -> List[Tuple[int, int]]:
        if score_matrix.size == 0:
            return []
        if linear_sum_assignment is None:
            matches: List[Tuple[int, int]] = []
            used_rows = set()
            used_cols = set()
            flat_indices = np.dstack(np.unravel_index(np.argsort(score_matrix.ravel())[::-1], score_matrix.shape))[0]
            for row_idx, col_idx in flat_indices:
                if row_idx in used_rows or col_idx in used_cols:
                    continue
                matches.append((int(row_idx), int(col_idx)))
                used_rows.add(int(row_idx))
                used_cols.add(int(col_idx))
            return matches
        row_ind, col_ind = linear_sum_assignment(-score_matrix)
        return [(int(r), int(c)) for r, c in zip(row_ind.tolist(), col_ind.tolist())]

    def _mark_missed(self, record: EntityRecord, frame_index: int) -> Dict[str, Any] | None:
        record.missed_frames += 1
        if record.missed_frames > self.config.miss_tolerance:
            record.active = False
            record.disappeared_frame = frame_index
            record.mark_state(DISAPPEARED, frame_index)
            return record.snapshot()
        record.active = False
        record.mark_state(INACTIVE, frame_index)
        return None

    def process_frame(self, objects: Sequence[Dict[str, Any]], frame_index: int) -> FrameAssociationResult:
        result = FrameAssociationResult(frame_index=frame_index)

        if not self.registry:
            result.is_first_frame = True
            for obj in objects:
                result.new_entities.append(self._register_new_entity(obj, frame_index))
            result.current_active_records = [record for record in self.registry.values() if record.active]
            return result

        prev_records = self._candidate_records()
        if not prev_records:
            for obj in objects:
                result.new_entities.append(self._register_new_entity(obj, frame_index))
            result.current_active_records = [record for record in self.registry.values() if record.active]
            return result

        prev_boxes = [record.last_bbox for record in prev_records]
        curr_boxes = [obj["bbox"] for obj in objects]
        iou_matrix = compute_iou_matrix(prev_boxes, curr_boxes)
        label_sim_matrix = self._compute_label_similarity_matrix(prev_records, objects)
        tag_match_matrix = self._compute_tag_match_matrix(prev_records, objects)
        score_matrix = self.config.iou_weight * iou_matrix + (1.0 - self.config.iou_weight) * label_sim_matrix

        proposed_matches = self._hungarian(score_matrix)
        matched_prev = set()
        matched_curr = set()

        for prev_idx, curr_idx in proposed_matches:
            score = float(score_matrix[prev_idx, curr_idx])
            label_sim = float(label_sim_matrix[prev_idx, curr_idx])
            iou = float(iou_matrix[prev_idx, curr_idx])
            tag_match = bool(tag_match_matrix[prev_idx, curr_idx] >= 1.0)
            if iou < self.config.min_iou_threshold and not tag_match:
                continue
            if score < self.config.combined_threshold or label_sim < self.config.label_threshold:
                continue
            record = prev_records[prev_idx]
            curr_obj = copy.deepcopy(objects[curr_idx])
            prev_snapshot = record.snapshot()
            was_reactivated = record.state == INACTIVE
            record.update(curr_obj, frame_index)
            result.matched.append(
                MatchResult(
                    entity_id=record.entity_id,
                    prev_snapshot=prev_snapshot,
                    curr_object=curr_obj,
                    score=score,
                    iou=iou,
                    label_similarity=label_sim,
                    was_reactivated=was_reactivated,
                )
            )
            matched_prev.add(prev_idx)
            matched_curr.add(curr_idx)

        for idx, obj in enumerate(objects):
            if idx not in matched_curr:
                result.new_entities.append(self._register_new_entity(copy.deepcopy(obj), frame_index))

        for idx, record in enumerate(prev_records):
            if idx in matched_prev:
                continue
            disappeared_snapshot = self._mark_missed(record, frame_index)
            if disappeared_snapshot is not None:
                result.disappeared_entities.append(disappeared_snapshot)

        result.current_active_records = [record for record in self.registry.values() if record.active]
        return result

    def current_frame_observations(self, frame_index: int) -> List[Dict[str, Any]]:
        observations: List[Dict[str, Any]] = []
        for record in self.registry.values():
            if record.last_frame != frame_index:
                continue
            if record.state != ACTIVE:
                continue
            observations.append(
                {
                    "entity_id": record.entity_id,
                    "label": record.label,
                    "tag": record.tag,
                    "frame_index": frame_index,
                    "center": box_center(record.last_bbox),
                    "bbox": list(record.last_bbox),
                }
            )
        return observations

    def export_registry(self) -> List[Dict[str, Any]]:
        return [record.snapshot() for record in sorted(self.registry.values(), key=lambda r: r.entity_id)]

    def total_tracked(self) -> int:
        return len(self.registry)

    def active_count(self) -> int:
        return sum(1 for record in self.registry.values() if record.state == ACTIVE)

    def total_displacement(self, record: EntityRecord) -> float:
        return compute_displacement(record.first_bbox, record.last_bbox)
