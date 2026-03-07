"""
事件生成模块
基于实体跟踪和运动分析的结果，生成自然语言事件描述
不依赖 LLM，采用模板/规则生成
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import pytz

from .utils import diff_relations, euclidean_distance


class EventType:
    """事件类型枚举"""
    ENTITY_APPEARED = "entity_appeared"
    ENTITY_DISAPPEARED = "entity_disappeared"
    ENTITY_MOVED = "entity_moved"
    RELATION_CHANGED = "relation_changed"
    ATTRIBUTE_CHANGED = "attribute_changed"
    INTERACTION = "interaction"
    TRAJECTORY_SUMMARY = "trajectory_summary"
    COMPOUND = "compound"


class EventGenerator:
    """
    事件生成器
    接收跟踪与分析结果，产出结构化事件记录
    """

    def __init__(self):
        self._event_counter = 0

    def _next_event_id(self) -> str:
        self._event_counter += 1
        return f"evt_{self._event_counter:05d}"

    def _make_event(self, event_type: str, summary: str,
                    involved_entities: List[str],
                    frame_range: List[int],
                    extra: Dict = None) -> Dict:
        """构造标准事件记录"""
        event = {
            "event_id": self._next_event_id(),
            "event_type": event_type,
            "summary": summary,
            "involved_entities": involved_entities,
            "frame_range": frame_range,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
        }
        if extra:
            event.update(extra)
        return event

    # ============================================================
    # 即时事件生成
    # ============================================================

    def gen_entity_appeared(self, entity_id: str, tag: str, label: str,
                            attributes: str, frame_idx: int) -> Dict:
        """生成实体出现事件"""
        summary = f"{tag} (a {label}) appeared at frame {frame_idx}"
        if attributes:
            summary += f", described as: {attributes}"
        return self._make_event(
            EventType.ENTITY_APPEARED, summary,
            [entity_id], [frame_idx, frame_idx]
        )

    def gen_entity_disappeared(self, entity_id: str, tag: str, label: str,
                                frame_idx: int) -> Dict:
        """生成实体消失事件"""
        summary = f"{tag} (a {label}) disappeared from the scene at frame {frame_idx}"
        return self._make_event(
            EventType.ENTITY_DISAPPEARED, summary,
            [entity_id], [frame_idx, frame_idx]
        )

    def gen_entity_moved(self, entity_id: str, tag: str,
                          prev_center: tuple, curr_center: tuple,
                          displacement: float, frame_idx: int) -> Dict:
        """生成实体移动事件"""
        summary = (
            f"{tag} moved from ({prev_center[0]:.0f},{prev_center[1]:.0f}) "
            f"to ({curr_center[0]:.0f},{curr_center[1]:.0f}), "
            f"displacement {displacement:.0f}px at frame {frame_idx}"
        )
        return self._make_event(
            EventType.ENTITY_MOVED, summary,
            [entity_id], [frame_idx, frame_idx],
            extra={"displacement": displacement}
        )

    def gen_relation_changed(self, entity_id: str, tag: str,
                              prev_obj: Dict, curr_obj: Dict,
                              frame_idx: int) -> Optional[Dict]:
        """生成关系变化事件"""
        diff = diff_relations(prev_obj, curr_obj)
        if not diff["added"] and not diff["removed"]:
            return None

        parts = []
        if diff["added"]:
            for rel in diff["added"][:3]:  # 限制数量避免过长
                parts.append(f"new relation: {rel['predicate']}")
        if diff["removed"]:
            for rel in diff["removed"][:3]:
                parts.append(f"lost relation: {rel['predicate']}")

        summary = f"{tag} relation changes at frame {frame_idx}: " + "; ".join(parts)

        involved = [entity_id]
        return self._make_event(
            EventType.RELATION_CHANGED, summary,
            involved, [frame_idx, frame_idx],
            extra={"added_relations": diff["added"], "removed_relations": diff["removed"]}
        )

    def gen_attribute_changed(self, entity_id: str, tag: str,
                               prev_attrs: str, curr_attrs: str,
                               frame_idx: int) -> Dict:
        """生成属性变化事件"""
        summary = (
            f"{tag} appearance changed at frame {frame_idx}: "
            f"from '{prev_attrs}' to '{curr_attrs}'"
        )
        return self._make_event(
            EventType.ATTRIBUTE_CHANGED, summary,
            [entity_id], [frame_idx, frame_idx]
        )

    # ============================================================
    # 缓存期事件生成（基于 MotionAnalyzer 的结果）
    # ============================================================

    def gen_trajectory_summary(self, entity_id: str, tag: str,
                                motion_analysis: Dict,
                                frame_range: List[int]) -> Dict:
        """生成轨迹摘要事件"""
        description = motion_analysis.get("description", "unknown motion")
        summary = f"{tag}: {description}"
        return self._make_event(
            EventType.TRAJECTORY_SUMMARY, summary,
            [entity_id], frame_range,
            extra={
                "pattern": motion_analysis.get("pattern"),
                "total_displacement": motion_analysis.get("total_displacement"),
                "avg_velocity": motion_analysis.get("avg_velocity")
            }
        )

    def gen_interaction_event(self, interaction: Dict) -> Dict:
        """生成多实体交互事件"""
        summary = interaction.get("description", "entity interaction detected")
        entities = [
            interaction.get("entity_a_id", ""),
            interaction.get("entity_b_id", "")
        ]
        frame_range = interaction.get("frame_range", [0, 0])
        return self._make_event(
            EventType.INTERACTION, summary,
            entities, frame_range,
            extra={"interaction_type": interaction.get("type")}
        )

    def gen_compound_event(self, entity_id: str, tag: str,
                            sub_events: List[Dict],
                            frame_range: List[int]) -> Dict:
        """
        生成复合事件（多个子事件合并）
        用于缓存 flush 时聚合
        """
        sub_summaries = [e["summary"] for e in sub_events]
        summary = f"[Compound] {tag} during frames {frame_range[0]}-{frame_range[1]}: " + \
                  " | ".join(sub_summaries)
        return self._make_event(
            EventType.COMPOUND, summary,
            [entity_id], frame_range,
            extra={"sub_events": [e["event_id"] for e in sub_events]}
        )

    # ============================================================
    # 首帧场景初始化事件
    # ============================================================

    def gen_initial_scene_description(self, objects: List[Dict], frame_idx: int) -> Dict:
        """生成首帧场景初始描述"""
        # 统计主要实体
        labels = [obj.get("label", "unknown") for obj in objects]
        label_counts = {}
        for lbl in labels:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # 按数量排序取前10
        top_labels = sorted(label_counts.items(), key=lambda x: -x[1])[:10]
        scene_desc = ", ".join([f"{count}x {label}" for label, count in top_labels])

        summary = (
            f"Initial scene at frame {frame_idx} contains {len(objects)} objects: {scene_desc}"
        )

        return self._make_event(
            EventType.ENTITY_APPEARED, summary,
            [], [frame_idx, frame_idx],
            extra={"is_initial_scene": True}
        )
