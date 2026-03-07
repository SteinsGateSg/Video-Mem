"""
实体跟踪模块
负责跨帧实体身份关联：IoU + label embedding 相似度 + 匈牙利算法
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from scipy.optimize import linear_sum_assignment
from copy import deepcopy
import logging

from .utils import (
    compute_iou_matrix, box_center, euclidean_distance,
    object_to_description, EmbeddingManager
)
from .config import STGConfig

logger = logging.getLogger(__name__)


class EntityRecord:
    """单个实体的完整跟踪记录"""

    def __init__(self, entity_id: str, label: str, tag: str,
                 first_frame_idx: int, obj_data: Dict):
        self.entity_id = entity_id
        self.label = label
        self.tag = tag
        self.first_seen_frame = first_frame_idx
        self.last_seen_frame = first_frame_idx
        self.is_static = False
        self.disappeared_count = 0  # 连续消失帧数

        # 属性历史
        self.attributes_history = [{
            "frame_idx": first_frame_idx,
            "attributes": obj_data.get("attributes", "")
        }]

        # 轨迹
        box = obj_data.get("box", [0, 0, 0, 0])
        center = box_center(box)
        self.trajectory = [{
            "frame_idx": first_frame_idx,
            "center": list(center),
            "box": box
        }]

        # 关系历史
        self.relations_history = [{
            "frame_idx": first_frame_idx,
            "subject_relations": deepcopy(obj_data.get("subject_relations", [])),
            "object_relations": deepcopy(obj_data.get("object_relations", []))
        }]

        # 层级信息
        self.layer_id = obj_data.get("layer_id", 1)
        self.layer_mapping = deepcopy(obj_data.get("layer_mapping", []))

        # 当前帧的原始对象数据（用于匹配时比较）
        self.current_obj_data = deepcopy(obj_data)

        # embedding（延迟计算）
        self.current_embedding: Optional[np.ndarray] = None
        self.description_embedding: Optional[np.ndarray] = None

    def get_latest_box(self) -> List[float]:
        """获取最新的 box"""
        if self.trajectory:
            return self.trajectory[-1]["box"]
        return [0, 0, 0, 0]

    def get_latest_center(self) -> Tuple[float, float]:
        """获取最新的中心点"""
        if self.trajectory:
            c = self.trajectory[-1]["center"]
            return (c[0], c[1])
        return (0.0, 0.0)

    def get_total_displacement(self) -> float:
        """获取总位移"""
        if len(self.trajectory) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(self.trajectory)):
            p1 = tuple(self.trajectory[i - 1]["center"])
            p2 = tuple(self.trajectory[i]["center"])
            total += euclidean_distance(p1, p2)
        return total

    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "entity_id": self.entity_id,
            "label": self.label,
            "tag": self.tag,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "is_static": self.is_static,
            "attributes_history": self.attributes_history,
            "trajectory": self.trajectory,
            "relations_history": self.relations_history,
            "layer_id": self.layer_id,
            "layer_mapping": self.layer_mapping,
        }


class EntityTracker:
    """
    实体跟踪器
    维护全局实体注册表，执行跨帧匹配
    """

    def __init__(self, config: STGConfig, emb_manager: EmbeddingManager):
        self.config = config
        self.emb_manager = emb_manager
        self.match_config = config.entity_matching
        self.traj_config = config.trajectory

        # 全局实体注册表: entity_id -> EntityRecord
        self.entity_registry: Dict[str, EntityRecord] = {}

        # 帧内 idx -> entity_id 的映射（当前帧）
        self.current_frame_mapping: Dict[int, str] = {}

        # 实体计数器（用于分配唯一 ID）
        self._entity_counter = 0

        # label embedding 缓存: label -> embedding
        self._label_emb_cache: Dict[str, np.ndarray] = {}

    def _next_entity_id(self) -> str:
        self._entity_counter += 1
        return f"entity_{self._entity_counter:04d}"

    def _get_label_embedding(self, label: str) -> np.ndarray:
        """获取 label 的 embedding，带缓存"""
        if label not in self._label_emb_cache:
            self._label_emb_cache[label] = self.emb_manager.embed(label)
        return self._label_emb_cache[label]

    def _compute_label_similarity_matrix(
        self, labels_a: List[str], labels_b: List[str]
    ) -> np.ndarray:
        """计算两组 label 的语义相似度矩阵"""
        # 收集所有唯一label并批量embed
        all_labels = list(set(labels_a + labels_b))
        for lbl in all_labels:
            if lbl not in self._label_emb_cache:
                self._label_emb_cache[lbl] = self.emb_manager.embed(lbl)

        m, n = len(labels_a), len(labels_b)
        sim_matrix = np.zeros((m, n), dtype=np.float32)
        for i, la in enumerate(labels_a):
            for j, lb in enumerate(labels_b):
                sim_matrix[i, j] = self.emb_manager.cosine_similarity(
                    self._label_emb_cache[la], self._label_emb_cache[lb]
                )
        return sim_matrix

    def match_entities(
        self, prev_objects: List[Dict], curr_objects: List[Dict]
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        匹配两帧中的对象
        
        Args:
            prev_objects: 上一帧的对象列表
            curr_objects: 当前帧的对象列表
        
        Returns:
            matched_pairs: [(prev_idx, curr_idx, score), ...]
            unmatched_prev: [prev_idx, ...]  (消失的对象)
            unmatched_curr: [curr_idx, ...]  (新出现的对象)
        """
        if not prev_objects or not curr_objects:
            return [], list(range(len(prev_objects))), list(range(len(curr_objects)))

        m, n = len(prev_objects), len(curr_objects)

        # 1. 计算 IoU 矩阵
        prev_boxes = [obj["box"] for obj in prev_objects]
        curr_boxes = [obj["box"] for obj in curr_objects]
        iou_matrix = compute_iou_matrix(prev_boxes, curr_boxes)

        # 2. 计算 label 相似度矩阵
        prev_labels = [obj.get("label", "") for obj in prev_objects]
        curr_labels = [obj.get("label", "") for obj in curr_objects]
        sim_matrix = self._compute_label_similarity_matrix(prev_labels, curr_labels)

        # 3. 综合得分
        alpha = self.match_config.alpha
        cost_matrix = alpha * iou_matrix + (1 - alpha) * sim_matrix

        # 4. 匈牙利算法（scipy 求最小化，所以用负值）
        row_indices, col_indices = linear_sum_assignment(-cost_matrix)

        # 5. 根据阈值过滤匹配
        threshold = self.match_config.combined_threshold
        matched_pairs = []
        matched_prev = set()
        matched_curr = set()

        for r, c in zip(row_indices, col_indices):
            score = cost_matrix[r, c]
            if score >= threshold:
                # 额外检查: label 相似度不能太低
                if sim_matrix[r, c] >= self.match_config.label_sim_threshold * 0.5:
                    matched_pairs.append((r, c, float(score)))
                    matched_prev.add(r)
                    matched_curr.add(c)

        unmatched_prev = [i for i in range(m) if i not in matched_prev]
        unmatched_curr = [j for j in range(n) if j not in matched_curr]

        return matched_pairs, unmatched_prev, unmatched_curr

    def process_frame(
        self, frame_objects: List[Dict], frame_idx: int, 
        prev_objects: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        处理一帧，执行实体匹配与注册

        Args:
            frame_objects: 当前帧的对象列表
            frame_idx: 当前帧序号
            prev_objects: 上一帧的对象列表（首帧为 None）

        Returns:
            {
                "matched": [(entity_id, prev_obj, curr_obj), ...],
                "new_entities": [(entity_id, curr_obj), ...],
                "disappeared": [(entity_id, prev_obj), ...],
                "frame_idx": int
            }
        """
        result = {
            "matched": [],
            "new_entities": [],
            "disappeared": [],
            "frame_idx": frame_idx
        }

        if prev_objects is None:
            # 首帧：所有对象都是新实体
            self.current_frame_mapping = {}
            for obj in frame_objects:
                entity_id = self._next_entity_id()
                record = EntityRecord(entity_id, obj["label"], obj["tag"], frame_idx, obj)
                self.entity_registry[entity_id] = record
                self.current_frame_mapping[obj["idx"]] = entity_id
                result["new_entities"].append((entity_id, obj))
            return result

        # 非首帧：执行匹配
        prev_idx_to_entity = dict(self.current_frame_mapping)  # 保存上一帧映射
        
        matched_pairs, unmatched_prev, unmatched_curr = self.match_entities(
            prev_objects, frame_objects
        )

        # 更新当前帧映射
        self.current_frame_mapping = {}

        # 1. 处理匹配成功的实体
        for prev_local_idx, curr_local_idx, score in matched_pairs:
            prev_obj = prev_objects[prev_local_idx]
            curr_obj = frame_objects[curr_local_idx]
            prev_frame_idx = prev_obj["idx"]

            # 找到对应的 entity_id
            entity_id = prev_idx_to_entity.get(prev_frame_idx)
            if entity_id is None:
                # 防御：如果上一帧映射丢失，创建新实体
                entity_id = self._next_entity_id()
                record = EntityRecord(entity_id, curr_obj["label"], curr_obj["tag"], frame_idx, curr_obj)
                self.entity_registry[entity_id] = record
                self.current_frame_mapping[curr_obj["idx"]] = entity_id
                result["new_entities"].append((entity_id, curr_obj))
                continue

            record = self.entity_registry[entity_id]
            record.last_seen_frame = frame_idx
            record.disappeared_count = 0

            # 更新轨迹
            prev_center = record.get_latest_center()
            curr_center = box_center(curr_obj["box"])
            displacement = euclidean_distance(prev_center, curr_center)

            if displacement > self.traj_config.movement_threshold:
                record.trajectory.append({
                    "frame_idx": frame_idx,
                    "center": list(curr_center),
                    "box": curr_obj["box"]
                })

            # 更新属性（如果有变化）
            curr_attrs = curr_obj.get("attributes", "")
            if record.attributes_history:
                last_attrs = record.attributes_history[-1].get("attributes", "")
                if curr_attrs != last_attrs and curr_attrs:
                    record.attributes_history.append({
                        "frame_idx": frame_idx,
                        "attributes": curr_attrs
                    })

            # 更新关系（记录变化）
            record.relations_history.append({
                "frame_idx": frame_idx,
                "subject_relations": deepcopy(curr_obj.get("subject_relations", [])),
                "object_relations": deepcopy(curr_obj.get("object_relations", []))
            })

            # 更新当前对象数据
            record.current_obj_data = deepcopy(curr_obj)

            self.current_frame_mapping[curr_obj["idx"]] = entity_id
            result["matched"].append((entity_id, prev_obj, curr_obj))

        # 2. 处理新出现的对象
        for curr_local_idx in unmatched_curr:
            curr_obj = frame_objects[curr_local_idx]
            entity_id = self._next_entity_id()
            record = EntityRecord(entity_id, curr_obj["label"], curr_obj["tag"], frame_idx, curr_obj)
            self.entity_registry[entity_id] = record
            self.current_frame_mapping[curr_obj["idx"]] = entity_id
            result["new_entities"].append((entity_id, curr_obj))

        # 3. 处理消失的对象
        for prev_local_idx in unmatched_prev:
            prev_obj = prev_objects[prev_local_idx]
            prev_frame_idx = prev_obj["idx"]
            entity_id = prev_idx_to_entity.get(prev_frame_idx)
            if entity_id and entity_id in self.entity_registry:
                self.entity_registry[entity_id].disappeared_count += 1
                result["disappeared"].append((entity_id, prev_obj))

        return result

    def detect_static_entities(self):
        """
        检测静态实体
        遍历所有实体，根据轨迹判断是否为静态（背景物体）
        """
        for entity_id, record in self.entity_registry.items():
            if len(record.trajectory) < self.traj_config.static_min_frames:
                # 轨迹点太少，如果出现帧数足够多但位移极小也算静态
                frames_span = record.last_seen_frame - record.first_seen_frame
                if frames_span >= self.traj_config.static_min_frames * 10:
                    # 出现了很多帧，但轨迹记录点很少 -> 说明大部分帧移动都小于阈值
                    record.is_static = True
                continue

            total_displacement = record.get_total_displacement()
            if total_displacement < self.traj_config.static_threshold:
                record.is_static = True
            else:
                record.is_static = False

        # 额外：某些标签天然是静态的
        static_labels = {
            "wall", "floor", "ceiling", "door", "window", "sign",
            "banner", "backboard", "basketball hoop", "court lines",
            "advertisement", "staircase", "steps", "logo", "slogan"
        }
        for entity_id, record in self.entity_registry.items():
            if record.label.lower() in static_labels:
                record.is_static = True

    def get_active_entities(self, max_disappeared: int = 3) -> List[EntityRecord]:
        """获取仍在活跃的实体（未消失太久的）"""
        return [
            record for record in self.entity_registry.values()
            if record.disappeared_count <= max_disappeared
        ]

    def get_dynamic_entities(self) -> List[EntityRecord]:
        """获取所有动态实体"""
        return [
            record for record in self.entity_registry.values()
            if not record.is_static
        ]

    def get_entity_by_tag(self, tag: str) -> Optional[EntityRecord]:
        """根据 tag 查找实体"""
        for record in self.entity_registry.values():
            if record.tag == tag:
                return record
        return None

    def get_all_entities(self) -> Dict[str, EntityRecord]:
        """获取所有实体"""
        return self.entity_registry

    def get_entity_summary(self) -> Dict[str, Any]:
        """获取实体跟踪摘要"""
        total = len(self.entity_registry)
        static = sum(1 for r in self.entity_registry.values() if r.is_static)
        dynamic = total - static
        return {
            "total_entities": total,
            "static_entities": static,
            "dynamic_entities": dynamic,
            "entity_ids": list(self.entity_registry.keys())
        }
