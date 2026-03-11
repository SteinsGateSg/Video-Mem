from __future__ import annotations

import itertools
from typing import Any, Dict, List, Sequence

from .config import MotionConfig, TrajectoryConfig
from .utils import angle_difference_deg, compute_direction, euclidean_distance


class MotionAnalyzer:
    def __init__(self, traj_cfg: TrajectoryConfig, motion_cfg: MotionConfig):
        self.traj_cfg = traj_cfg
        self.motion_cfg = motion_cfg

    def _dominant_mode(self, dx: float, dy: float, direction_changes: int) -> str:
        if abs(dx) < self.traj_cfg.static_threshold and abs(dy) < self.traj_cfg.static_threshold:
            return "stationary"
        if direction_changes >= 1:
            return "direction_change"
        if abs(dy) > abs(dx) and dy < -self.traj_cfg.jump_vertical_threshold:
            return "jumping_up"
        if abs(dx) >= abs(dy):
            return "moving_right" if dx >= 0 else "moving_left"
        return "moving_down" if dy >= 0 else "moving_up"

    def analyze_single_entity(
        self,
        entity_info: Dict[str, Any],
        trajectory: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if len(trajectory) < self.traj_cfg.min_frames_for_summary:
            return None
        start = trajectory[0]
        end = trajectory[-1]
        start_center = tuple(start["center"])
        end_center = tuple(end["center"])
        total_displacement = euclidean_distance(start_center, end_center)
        if total_displacement < self.traj_cfg.movement_threshold:
            return None

        frame_span = max(1, int(end["frame_index"]) - int(start["frame_index"]))
        avg_speed = total_displacement / frame_span
        dx = end_center[0] - start_center[0]
        dy = end_center[1] - start_center[1]

        step_angles: List[float] = []
        for prev_obs, curr_obs in zip(trajectory[:-1], trajectory[1:]):
            if tuple(prev_obs["center"]) == tuple(curr_obs["center"]):
                continue
            step_angles.append(compute_direction(tuple(prev_obs["center"]), tuple(curr_obs["center"])))
        direction_changes = 0
        for a, b in zip(step_angles[:-1], step_angles[1:]):
            if angle_difference_deg(a, b) >= self.motion_cfg.direction_change_angle_deg:
                direction_changes += 1

        mode = self._dominant_mode(dx, dy, direction_changes)
        return {
            "entity_id": entity_info["entity_id"],
            "tag": entity_info["tag"],
            "label": entity_info["label"],
            "frame_start": int(start["frame_index"]),
            "frame_end": int(end["frame_index"]),
            "total_displacement": float(total_displacement),
            "avg_speed": float(avg_speed),
            "mode": mode,
            "direction_changes": direction_changes,
            "num_observations": len(trajectory),
        }

    def analyze_all_interactions(
        self,
        trajectories: Dict[str, List[Dict[str, Any]]],
        entity_info: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        entity_ids = sorted(trajectories.keys())
        for entity_a, entity_b in itertools.combinations(entity_ids, 2):
            traj_a = {int(obs["frame_index"]): obs for obs in trajectories[entity_a]}
            traj_b = {int(obs["frame_index"]): obs for obs in trajectories[entity_b]}
            common_frames = sorted(set(traj_a) & set(traj_b))
            if len(common_frames) < 2:
                continue
            start_frame, end_frame = common_frames[0], common_frames[-1]
            start_dist = euclidean_distance(tuple(traj_a[start_frame]["center"]), tuple(traj_b[start_frame]["center"]))
            end_dist = euclidean_distance(tuple(traj_a[end_frame]["center"]), tuple(traj_b[end_frame]["center"]))
            if start_dist < self.motion_cfg.min_interaction_distance:
                continue
            ratio = end_dist / max(start_dist, 1e-6)

            interaction_type = None
            if ratio < self.motion_cfg.approach_distance_ratio:
                interaction_type = "approaching_each_other"
            elif ratio > self.motion_cfg.depart_distance_ratio:
                interaction_type = "departing_from_each_other"
            else:
                dir_a = compute_direction(tuple(traj_a[start_frame]["center"]), tuple(traj_a[end_frame]["center"]))
                dir_b = compute_direction(tuple(traj_b[start_frame]["center"]), tuple(traj_b[end_frame]["center"]))
                if angle_difference_deg(dir_a, dir_b) <= self.motion_cfg.moving_together_angle_deg:
                    interaction_type = "moving_together"

            if interaction_type is None:
                continue
            results.append(
                {
                    "entity_a": entity_a,
                    "entity_b": entity_b,
                    "tag_a": entity_info[entity_a]["tag"],
                    "tag_b": entity_info[entity_b]["tag"],
                    "label_a": entity_info[entity_a]["label"],
                    "label_b": entity_info[entity_b]["label"],
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "distance_start": float(start_dist),
                    "distance_end": float(end_dist),
                    "interaction_type": interaction_type,
                    "num_common_frames": len(common_frames),
                }
            )
        return results
