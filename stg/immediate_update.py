"""
即时更新模块
每处理一个关键帧就立即执行的更新逻辑
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import logging

from .entity_tracker import EntityTracker, EntityRecord
from .event_generator import EventGenerator
from .faiss_store import FAISSStore
from .utils import (
    EmbeddingManager, box_center, euclidean_distance,
    diff_relations, has_relation_change, entity_state_description,
    filter_objects_by_score
)
from .config import STGConfig

logger = logging.getLogger(__name__)


class ImmediateUpdater:
    """
    即时更新器
    处理每一帧时立即执行：实体匹配、轨迹记录、关系差异检测、事件生成、写入向量库
    """

    def __init__(self, config: STGConfig, tracker: EntityTracker,
                 event_gen: EventGenerator, emb_manager: EmbeddingManager,
                 faiss_store: FAISSStore):
        self.config = config
        self.tracker = tracker
        self.event_gen = event_gen
        self.emb_manager = emb_manager
        self.faiss_store = faiss_store
        
        self.prev_frame_objects: Optional[List[Dict]] = None
        self.prev_frame_idx: int = -1
        self.frame_count = 0

    def process_frame(self, frame_data: Dict, sample_id: str) -> Dict[str, Any]:
        """
        处理一帧场景图数据

        Args:
            frame_data: {"image_path": str, "objects": [...]}
            sample_id: 视频/样本 ID
        
        Returns:
            {
                "frame_idx": int,
                "events": List[Dict],         # 本帧产生的事件
                "matched_count": int,
                "new_count": int,
                "disappeared_count": int,
                "buffered_observations": List  # 传给 buffer 的观测数据
            }
        """
        from .utils import frame_index_from_path

        image_path = frame_data.get("image_path", "")
        frame_idx = frame_index_from_path(image_path)
        objects = frame_data.get("objects", [])

        # 过滤低置信度对象
        objects = filter_objects_by_score(objects, self.config.entity_matching.score_filter)

        events = []
        self.frame_count += 1

        if self.prev_frame_objects is None:
            # ========== 首帧处理 ==========
            tracking_result = self.tracker.process_frame(objects, frame_idx, prev_objects=None)

            # 生成首帧场景描述事件
            scene_event = self.event_gen.gen_initial_scene_description(objects, frame_idx)
            events.append(scene_event)
            self._write_event_to_faiss(scene_event, sample_id)

            # 为每个新实体生成出现事件
            for entity_id, obj in tracking_result["new_entities"]:
                event = self.event_gen.gen_entity_appeared(
                    entity_id, obj.get("tag", ""), obj.get("label", ""),
                    obj.get("attributes", ""), frame_idx
                )
                events.append(event)
                self._write_event_to_faiss(event, sample_id)

                # 写入实体向量库
                self._write_entity_state(entity_id, sample_id)

        else:
            # ========== 非首帧处理 ==========
            tracking_result = self.tracker.process_frame(
                objects, frame_idx, prev_objects=self.prev_frame_objects
            )

            # 处理匹配成功的实体
            for entity_id, prev_obj, curr_obj in tracking_result["matched"]:
                entity = self.tracker.entity_registry.get(entity_id)
                if entity is None:
                    continue

                # 检测显著移动
                prev_center = box_center(prev_obj["box"])
                curr_center = box_center(curr_obj["box"])
                displacement = euclidean_distance(prev_center, curr_center)

                if displacement > self.config.trajectory.movement_threshold:
                    move_event = self.event_gen.gen_entity_moved(
                        entity_id, entity.tag,
                        prev_center, curr_center,
                        displacement, frame_idx
                    )
                    events.append(move_event)
                    self._write_event_to_faiss(move_event, sample_id)

                # 检测关系变化
                rel_event = self.event_gen.gen_relation_changed(
                    entity_id, entity.tag,
                    prev_obj, curr_obj, frame_idx
                )
                if rel_event:
                    events.append(rel_event)
                    self._write_event_to_faiss(rel_event, sample_id)

                # 检测属性变化
                prev_attrs = prev_obj.get("attributes", "")
                curr_attrs = curr_obj.get("attributes", "")
                if prev_attrs and curr_attrs and prev_attrs != curr_attrs:
                    # 用 embedding 相似度判断是否为显著变化
                    if self._attributes_significantly_changed(prev_attrs, curr_attrs):
                        attr_event = self.event_gen.gen_attribute_changed(
                            entity_id, entity.tag,
                            prev_attrs, curr_attrs, frame_idx
                        )
                        events.append(attr_event)
                        self._write_event_to_faiss(attr_event, sample_id)

                # 更新实体向量库
                self._write_entity_state(entity_id, sample_id)

            # 处理新出现的实体
            for entity_id, obj in tracking_result["new_entities"]:
                event = self.event_gen.gen_entity_appeared(
                    entity_id, obj.get("tag", ""), obj.get("label", ""),
                    obj.get("attributes", ""), frame_idx
                )
                events.append(event)
                self._write_event_to_faiss(event, sample_id)
                self._write_entity_state(entity_id, sample_id)

            # 处理消失的实体
            for entity_id, obj in tracking_result["disappeared"]:
                entity = self.tracker.entity_registry.get(entity_id)
                if entity and entity.disappeared_count == 1:
                    # 只在首次消失时记录
                    event = self.event_gen.gen_entity_disappeared(
                        entity_id, entity.tag, entity.label, frame_idx
                    )
                    events.append(event)
                    self._write_event_to_faiss(event, sample_id)

        # 保存当前帧数据用于下一帧比较
        self.prev_frame_objects = objects
        self.prev_frame_idx = frame_idx

        # 构造给 buffer 的观测数据
        buffered_observations = {
            "frame_idx": frame_idx,
            "objects": objects,
            "tracking_result": {
                "matched": [
                    (eid, self.tracker.entity_registry[eid].to_dict())
                    for eid, _, _ in tracking_result.get("matched", [])
                    if eid in self.tracker.entity_registry
                ],
                "new_entities": [
                    (eid, self.tracker.entity_registry[eid].to_dict())
                    for eid, _ in tracking_result.get("new_entities", [])
                    if eid in self.tracker.entity_registry
                ]
            }
        }

        result = {
            "frame_idx": frame_idx,
            "events": events,
            "matched_count": len(tracking_result.get("matched", [])),
            "new_count": len(tracking_result.get("new_entities", [])),
            "disappeared_count": len(tracking_result.get("disappeared", [])),
            "buffered_observations": buffered_observations
        }

        if self.config.verbose:
            print(
                f"  Frame {frame_idx}: "
                f"matched={result['matched_count']}, "
                f"new={result['new_count']}, "
                f"disappeared={result['disappeared_count']}, "
                f"events={len(events)}"
            )

        return result

    def _write_event_to_faiss(self, event: Dict, sample_id: str):
        """将事件写入 events 向量库"""
        summary = event.get("summary", "")
        if not summary:
            return

        embedding = self.emb_manager.embed(summary)
        metadata = {
            "event_id": event.get("event_id"),
            "event_type": event.get("event_type"),
            "summary": summary,
            "involved_entities": event.get("involved_entities", []),
            "frame_range": event.get("frame_range", []),
            "timestamp": event.get("timestamp", ""),
        }

        self.faiss_store.add_memory(sample_id, "events", embedding, metadata)

    def _write_entity_state(self, entity_id: str, sample_id: str):
        """将实体当前状态写入实体向量库"""
        entity = self.tracker.entity_registry.get(entity_id)
        if entity is None:
            return

        # 生成实体状态描述
        description = entity_state_description(entity.to_dict())
        embedding = self.emb_manager.embed(description)

        metadata = {
            "entity_id": entity_id,
            "tag": entity.tag,
            "label": entity.label,
            "description": description,
            "frame_idx": entity.last_seen_frame,
            "is_static": entity.is_static,
            "timestamp": "",
        }

        entity_key = f"entity_{entity_id}"
        self.faiss_store.add_memory(sample_id, entity_key, embedding, metadata)

    def _attributes_significantly_changed(self, prev_attrs: str, curr_attrs: str,
                                           threshold: float = 0.85) -> bool:
        """判断属性是否发生了显著变化（低于阈值说明变化大）"""
        if prev_attrs == curr_attrs:
            return False
        if not prev_attrs or not curr_attrs:
            return True

        emb_prev = self.emb_manager.embed(prev_attrs)
        emb_curr = self.emb_manager.embed(curr_attrs)
        sim = self.emb_manager.cosine_similarity(emb_prev, emb_curr)
        return sim < threshold
