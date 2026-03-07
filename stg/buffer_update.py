"""
缓存更新模块
参考 main.py 的 buffer + flush 机制
累积多帧观测后，进行实体聚类、轨迹分析、交互检测
"""

from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np
import logging

from .entity_tracker import EntityTracker, EntityRecord
from .motion_analyzer import MotionAnalyzer
from .event_generator import EventGenerator
from .faiss_store import FAISSStore
from .utils import EmbeddingManager, compute_iou, box_center, euclidean_distance
from .config import STGConfig

logger = logging.getLogger(__name__)


class BufferUpdater:
    """
    缓存更新器
    累积多帧观测数据，在缓冲区满后执行：
    1. 跨帧实体聚类
    2. 轨迹摘要与运动模式检测
    3. 多实体交互分析
    4. 写入向量库
    """

    def __init__(self, config: STGConfig, tracker: EntityTracker,
                 motion_analyzer: MotionAnalyzer, event_gen: EventGenerator,
                 emb_manager: EmbeddingManager, faiss_store: FAISSStore):
        self.config = config
        self.tracker = tracker
        self.motion_analyzer = motion_analyzer
        self.event_gen = event_gen
        self.emb_manager = emb_manager
        self.faiss_store = faiss_store

        # 缓冲区: sample_id -> List[observation]
        self.buffer: Dict[str, List[Dict]] = {}
        self.buffer_size = config.buffer.buffer_size

    def add_observation(self, observation: Dict, sample_id: str):
        """
        将一帧的观测数据添加到缓冲区
        
        Args:
            observation: ImmediateUpdater.process_frame 返回的 buffered_observations
            sample_id: 视频/样本 ID
        """
        if sample_id not in self.buffer:
            self.buffer[sample_id] = []

        self.buffer[sample_id].append(observation)

        # 检查是否需要 flush
        if len(self.buffer[sample_id]) >= self.buffer_size:
            self.flush(sample_id)

    def flush(self, sample_id: str) -> List[Dict]:
        """
        清空缓冲区并处理累积数据
        
        Returns:
            生成的事件列表
        """
        if sample_id not in self.buffer or not self.buffer[sample_id]:
            return []

        observations = self.buffer[sample_id]
        events = []

        if self.config.verbose:
            frame_indices = [obs["frame_idx"] for obs in observations]
            print(f"\n  [Buffer Flush] Processing {len(observations)} frames: {frame_indices}")

        # ======================================================
        # Step 1: 收集缓冲期内每个实体的轨迹片段
        # ======================================================
        entity_trajectories = self._collect_entity_trajectories(observations)

        # ======================================================
        # Step 2: 对每个动态实体做轨迹分析
        # ======================================================
        trajectory_events = self._analyze_trajectories(entity_trajectories, observations)
        events.extend(trajectory_events)

        # ======================================================
        # Step 3: 多实体交互分析
        # ======================================================
        interaction_events = self._analyze_interactions(sample_id)
        events.extend(interaction_events)

        # ======================================================
        # Step 4: 写入向量库
        # ======================================================
        for event in events:
            self._write_event_to_faiss(event, sample_id)

        if self.config.verbose:
            print(f"  [Buffer Flush] Generated {len(events)} events")

        # 清空缓冲区
        self.buffer[sample_id] = []

        # 保存FAISS
        self.faiss_store.save_index(sample_id, "events")

        return events

    def flush_all(self) -> Dict[str, List[Dict]]:
        """清空所有样本的缓冲区"""
        results = {}
        for sample_id in list(self.buffer.keys()):
            results[sample_id] = self.flush(sample_id)
        return results

    def _collect_entity_trajectories(
        self, observations: List[Dict]
    ) -> Dict[str, List[Dict]]:
        """
        从缓冲区的观测数据中收集每个实体在这段时间内的轨迹

        Returns:
            {entity_id: [{"frame_idx": int, "center": [x,y], "box": [...]}, ...]}
        """
        entity_trajectories: Dict[str, List[Dict]] = defaultdict(list)

        for obs in observations:
            frame_idx = obs["frame_idx"]
            tracking = obs.get("tracking_result", {})

            # 处理匹配的实体
            for entity_id, entity_dict in tracking.get("matched", []):
                trajectory = entity_dict.get("trajectory", [])
                # 取最后一个轨迹点（当前帧的位置）
                if trajectory:
                    latest = trajectory[-1]
                    entity_trajectories[entity_id].append({
                        "frame_idx": latest.get("frame_idx", frame_idx),
                        "center": latest["center"],
                        "box": latest["box"]
                    })

            # 处理新出现的实体
            for entity_id, entity_dict in tracking.get("new_entities", []):
                trajectory = entity_dict.get("trajectory", [])
                if trajectory:
                    latest = trajectory[-1]
                    entity_trajectories[entity_id].append({
                        "frame_idx": latest.get("frame_idx", frame_idx),
                        "center": latest["center"],
                        "box": latest["box"]
                    })

        # 按帧序号排序并去重
        for entity_id in entity_trajectories:
            seen_frames = set()
            unique_traj = []
            for point in sorted(entity_trajectories[entity_id], key=lambda x: x["frame_idx"]):
                if point["frame_idx"] not in seen_frames:
                    seen_frames.add(point["frame_idx"])
                    unique_traj.append(point)
            entity_trajectories[entity_id] = unique_traj

        return dict(entity_trajectories)

    def _analyze_trajectories(
        self, entity_trajectories: Dict[str, List[Dict]],
        observations: List[Dict]
    ) -> List[Dict]:
        """对缓冲期内有运动的实体生成轨迹摘要事件"""
        events = []

        frame_indices = [obs["frame_idx"] for obs in observations]
        if not frame_indices:
            return events

        frame_range = [min(frame_indices), max(frame_indices)]

        for entity_id, trajectory in entity_trajectories.items():
            if len(trajectory) < 2:
                continue

            entity = self.tracker.entity_registry.get(entity_id)
            if entity is None or entity.is_static:
                continue

            # 运动分析
            motion_analysis = self.motion_analyzer.analyze_single_entity(trajectory)

            # 只有有明显运动的才生成事件
            if motion_analysis["total_displacement"] > self.config.trajectory.movement_threshold:
                event = self.event_gen.gen_trajectory_summary(
                    entity_id, entity.tag,
                    motion_analysis, frame_range
                )
                events.append(event)

        return events

    def _analyze_interactions(self, sample_id: str) -> List[Dict]:
        """分析当前所有动态实体间的交互"""
        events = []

        interactions = self.motion_analyzer.analyze_all_interactions(
            self.tracker.entity_registry
        )

        for interaction in interactions:
            event = self.event_gen.gen_interaction_event(interaction)
            events.append(event)

        return events

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
