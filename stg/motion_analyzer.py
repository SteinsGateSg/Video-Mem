"""
运动分析模块
基于轨迹的几何特征分析运动模式、多实体交互
不依赖 LLM，纯规则/几何推断
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

from .utils import euclidean_distance, compute_direction, compute_velocity, box_center
from .config import STGConfig


class MotionPattern:
    """运动模式定义"""
    STATIONARY = "stationary"
    MOVING_LEFT = "moving_left"
    MOVING_RIGHT = "moving_right"
    MOVING_UP = "moving_up"
    MOVING_DOWN = "moving_down"
    APPROACHING = "approaching"        # 接近某实体
    DEPARTING = "departing"            # 远离某实体
    JUMPING = "jumping"                # 先上移后下移
    DIRECTION_CHANGE = "direction_change"  # 大幅转向


class InteractionType:
    """多实体交互类型"""
    APPROACHING_EACH_OTHER = "approaching_each_other"
    DEPARTING_FROM_EACH_OTHER = "departing_from_each_other"
    MOVING_TOGETHER = "moving_together"
    PASSING = "passing"  # 交错/传递


class MotionAnalyzer:
    """
    运动分析器
    输入实体轨迹，输出运动模式描述
    """

    def __init__(self, config: STGConfig):
        self.config = config
        self.motion_config = config.motion

    def analyze_single_entity(self, trajectory: List[Dict]) -> Dict[str, Any]:
        """
        分析单个实体的运动模式
        
        Args:
            trajectory: [{"frame_idx": int, "center": [x, y], "box": [x1,y1,x2,y2]}, ...]
        
        Returns:
            {
                "pattern": str,
                "total_displacement": float,
                "avg_velocity": float,
                "dominant_direction": str,
                "direction_changes": int,
                "description": str
            }
        """
        if len(trajectory) < 2:
            return {
                "pattern": MotionPattern.STATIONARY,
                "total_displacement": 0.0,
                "avg_velocity": 0.0,
                "dominant_direction": "none",
                "direction_changes": 0,
                "description": "stationary (insufficient trajectory data)"
            }

        # 计算逐段位移和方向
        displacements = []
        directions = []
        velocities = []

        for i in range(1, len(trajectory)):
            p1 = tuple(trajectory[i - 1]["center"])
            p2 = tuple(trajectory[i]["center"])
            f1 = trajectory[i - 1]["frame_idx"]
            f2 = trajectory[i]["frame_idx"]

            disp = euclidean_distance(p1, p2)
            displacements.append(disp)

            direction = compute_direction(p1, p2)
            directions.append(direction)

            frame_gap = max(1, f2 - f1)
            vel = compute_velocity(disp, frame_gap)
            velocities.append(vel)

        total_displacement = sum(displacements)
        avg_velocity = np.mean(velocities) if velocities else 0.0

        # 判断主方向
        dominant_direction = self._get_dominant_direction(trajectory)

        # 计算方向变化次数
        direction_changes = self._count_direction_changes(directions)

        # 判断模式
        pattern = self._classify_pattern(
            trajectory, total_displacement, avg_velocity, 
            dominant_direction, direction_changes
        )

        # 生成描述
        description = self._generate_motion_description(
            pattern, total_displacement, avg_velocity,
            dominant_direction, trajectory
        )

        return {
            "pattern": pattern,
            "total_displacement": total_displacement,
            "avg_velocity": avg_velocity,
            "dominant_direction": dominant_direction,
            "direction_changes": direction_changes,
            "description": description
        }

    def _get_dominant_direction(self, trajectory: List[Dict]) -> str:
        """判断主要运动方向"""
        if len(trajectory) < 2:
            return "none"

        start = trajectory[0]["center"]
        end = trajectory[-1]["center"]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        if abs(dx) < 5 and abs(dy) < 5:
            return "none"

        if abs(dx) > abs(dy):
            return "right" if dx > 0 else "left"
        else:
            return "down" if dy > 0 else "up"

    def _count_direction_changes(self, directions: List[float]) -> int:
        """计算方向显著变化次数"""
        angle_threshold = self.motion_config.direction_change_angle
        changes = 0
        for i in range(1, len(directions)):
            diff = abs(directions[i] - directions[i - 1])
            if diff > 180:
                diff = 360 - diff
            if diff > angle_threshold:
                changes += 1
        return changes

    def _classify_pattern(
        self, trajectory, total_displacement, avg_velocity,
        dominant_direction, direction_changes
    ) -> str:
        """分类运动模式"""
        if total_displacement < self.config.trajectory.static_threshold:
            return MotionPattern.STATIONARY

        # 检测跳跃：中间点y值明显低于起止点（画面坐标系y向下）
        if len(trajectory) >= 3:
            ys = [t["center"][1] for t in trajectory]
            min_y_idx = np.argmin(ys)
            if 0 < min_y_idx < len(ys) - 1:
                # 中间有y值最小（最高点），且明显低于起止
                y_diff = min(ys[0], ys[-1]) - ys[min_y_idx]
                if y_diff > 20:  # 至少 20px 的高度变化
                    return MotionPattern.JUMPING

        # 方向变化频繁
        if direction_changes >= 3:
            return MotionPattern.DIRECTION_CHANGE

        # 按主方向分类
        direction_map = {
            "left": MotionPattern.MOVING_LEFT,
            "right": MotionPattern.MOVING_RIGHT,
            "up": MotionPattern.MOVING_UP,
            "down": MotionPattern.MOVING_DOWN,
        }
        return direction_map.get(dominant_direction, MotionPattern.STATIONARY)

    def _generate_motion_description(
        self, pattern, total_displacement, avg_velocity,
        dominant_direction, trajectory
    ) -> str:
        """生成运动描述的自然语言"""
        start = trajectory[0]
        end = trajectory[-1]
        frame_range = f"frame {start['frame_idx']} to {end['frame_idx']}"

        if pattern == MotionPattern.STATIONARY:
            return f"remained mostly stationary during {frame_range}"

        if pattern == MotionPattern.JUMPING:
            return (
                f"performed a jumping motion during {frame_range}, "
                f"displacement {total_displacement:.0f}px"
            )

        if pattern == MotionPattern.DIRECTION_CHANGE:
            return (
                f"moved with frequent direction changes during {frame_range}, "
                f"total displacement {total_displacement:.0f}px"
            )

        direction_desc = {
            MotionPattern.MOVING_LEFT: "moved to the left",
            MotionPattern.MOVING_RIGHT: "moved to the right",
            MotionPattern.MOVING_UP: "moved upward",
            MotionPattern.MOVING_DOWN: "moved downward",
        }
        desc = direction_desc.get(pattern, "moved")

        return (
            f"{desc} during {frame_range}, "
            f"from ({start['center'][0]:.0f},{start['center'][1]:.0f}) "
            f"to ({end['center'][0]:.0f},{end['center'][1]:.0f}), "
            f"displacement {total_displacement:.0f}px, "
            f"avg velocity {avg_velocity:.1f}px/frame"
        )

    def analyze_pairwise_interaction(
        self, 
        entity_a_trajectory: List[Dict],
        entity_b_trajectory: List[Dict],
        entity_a_tag: str = "entity_a",
        entity_b_tag: str = "entity_b"
    ) -> Optional[Dict[str, Any]]:
        """
        分析两个实体之间的交互
        
        Returns:
            None 或 {"type": str, "description": str, "frame_range": [start, end]}
        """
        if len(entity_a_trajectory) < 2 or len(entity_b_trajectory) < 2:
            return None

        # 找到重叠帧区间
        a_frames = {t["frame_idx"]: t for t in entity_a_trajectory}
        b_frames = {t["frame_idx"]: t for t in entity_b_trajectory}
        common_frames = sorted(set(a_frames.keys()) & set(b_frames.keys()))

        if len(common_frames) < 2:
            # 没有足够的共同帧，尝试用最近的帧做比较
            a_all_frames = sorted(a_frames.keys())
            b_all_frames = sorted(b_frames.keys())
            
            # 找时间上最接近的帧对
            overlap_start = max(a_all_frames[0], b_all_frames[0])
            overlap_end = min(a_all_frames[-1], b_all_frames[-1])
            
            if overlap_start >= overlap_end:
                return None
            
            # 在重叠区间内各取首尾
            a_start = min(a_all_frames, key=lambda f: abs(f - overlap_start))
            a_end = min(a_all_frames, key=lambda f: abs(f - overlap_end))
            b_start = min(b_all_frames, key=lambda f: abs(f - overlap_start))
            b_end = min(b_all_frames, key=lambda f: abs(f - overlap_end))
            
            if a_start == a_end or b_start == b_end:
                return None

            dist_start = euclidean_distance(
                tuple(a_frames[a_start]["center"]),
                tuple(b_frames[b_start]["center"])
            )
            dist_end = euclidean_distance(
                tuple(a_frames[a_end]["center"]),
                tuple(b_frames[b_end]["center"])
            )
            frame_range = [overlap_start, overlap_end]
        else:
            # 有足够的共同帧
            dist_start = euclidean_distance(
                tuple(a_frames[common_frames[0]]["center"]),
                tuple(b_frames[common_frames[0]]["center"])
            )
            dist_end = euclidean_distance(
                tuple(a_frames[common_frames[-1]]["center"]),
                tuple(b_frames[common_frames[-1]]["center"])
            )
            frame_range = [common_frames[0], common_frames[-1]]

        if dist_start == 0:
            dist_start = 0.001

        ratio = dist_end / dist_start

        # 判断交互类型
        if ratio < self.motion_config.approach_distance_ratio:
            return {
                "type": InteractionType.APPROACHING_EACH_OTHER,
                "description": (
                    f"{entity_a_tag} and {entity_b_tag} moved closer to each other "
                    f"(distance: {dist_start:.0f}px -> {dist_end:.0f}px) "
                    f"during frames {frame_range[0]}-{frame_range[1]}"
                ),
                "frame_range": frame_range,
                "distance_change": (dist_start, dist_end)
            }
        elif ratio > 1.0 / self.motion_config.approach_distance_ratio:
            return {
                "type": InteractionType.DEPARTING_FROM_EACH_OTHER,
                "description": (
                    f"{entity_a_tag} and {entity_b_tag} moved away from each other "
                    f"(distance: {dist_start:.0f}px -> {dist_end:.0f}px) "
                    f"during frames {frame_range[0]}-{frame_range[1]}"
                ),
                "frame_range": frame_range,
                "distance_change": (dist_start, dist_end)
            }

        # 检测是否同向移动（moving together）
        if len(common_frames) >= 2:
            a_dir = compute_direction(
                tuple(a_frames[common_frames[0]]["center"]),
                tuple(a_frames[common_frames[-1]]["center"])
            )
            b_dir = compute_direction(
                tuple(b_frames[common_frames[0]]["center"]),
                tuple(b_frames[common_frames[-1]]["center"])
            )
            dir_diff = abs(a_dir - b_dir)
            if dir_diff > 180:
                dir_diff = 360 - dir_diff

            if dir_diff < 30 and abs(ratio - 1.0) < 0.3:
                return {
                    "type": InteractionType.MOVING_TOGETHER,
                    "description": (
                        f"{entity_a_tag} and {entity_b_tag} moved together "
                        f"in roughly the same direction "
                        f"during frames {frame_range[0]}-{frame_range[1]}"
                    ),
                    "frame_range": frame_range,
                    "distance_change": (dist_start, dist_end)
                }

        return None

    def analyze_all_interactions(
        self, entities: Dict[str, Any], min_trajectory_len: int = 2
    ) -> List[Dict[str, Any]]:
        """
        分析所有动态实体间的两两交互
        
        Args:
            entities: {entity_id: EntityRecord} (EntityRecord 应有 trajectory 和 tag)
        
        Returns:
            List 交互事件
        """
        # 筛选有足够轨迹的动态实体
        dynamic_entities = [
            (eid, e) for eid, e in entities.items()
            if not e.is_static and len(e.trajectory) >= min_trajectory_len
        ]

        interactions = []
        for i in range(len(dynamic_entities)):
            for j in range(i + 1, len(dynamic_entities)):
                eid_a, entity_a = dynamic_entities[i]
                eid_b, entity_b = dynamic_entities[j]

                interaction = self.analyze_pairwise_interaction(
                    entity_a.trajectory,
                    entity_b.trajectory,
                    entity_a.tag,
                    entity_b.tag
                )

                if interaction is not None:
                    interaction["entity_a_id"] = eid_a
                    interaction["entity_b_id"] = eid_b
                    interactions.append(interaction)

        return interactions

    def summarize_trajectory(
        self, entity_tag: str, trajectory: List[Dict]
    ) -> str:
        """
        为一个实体的轨迹生成自然语言摘要
        """
        analysis = self.analyze_single_entity(trajectory)
        return f"{entity_tag}: {analysis['description']}"
