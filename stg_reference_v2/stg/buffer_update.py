"""
缓冲区更新模块（Buffer Update）

本模块实现了帧级观测数据的缓冲与定期刷新机制。
与 ImmediateUpdater 逐帧生成事件不同，BufferUpdater 积累若干帧后一次性分析：

工作流程：
    1. observe(frame_observations): 每帧调用，将当前帧的所有活跃实体观测加入缓冲区
    2. 当缓冲区满（达到 buffer_size 帧）后返回 True，由外层调用 flush()
    3. flush(sample_id):
       a. 将缓冲区内所有观测按 entity_id 分组组织轨迹
       b. 对每个实体调用 MotionAnalyzer.analyze_single_entity() 生成轨迹摘要
       c. 调用 MotionAnalyzer.analyze_all_interactions() 分析实体间交互
       d. 将生成的 trajectory_summary 和 interaction 事件向量化并写入 VectorStore
       e. 清空缓冲区

这一机制确保了跨帧运动模式和实体间交互关系能够被一次性捕获。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from .config import STGConfig
from .event_generator import EventGenerator
from .motion_analyzer import MotionAnalyzer
from .utils import EmbeddingManager
from .vector_store import VectorStore


class BufferUpdater:
    def __init__(
        self,
        config: STGConfig,
        motion_analyzer: MotionAnalyzer,
        event_generator: EventGenerator,
        embedder: EmbeddingManager,
        store: VectorStore,
    ):
        self.config = config
        self.motion_analyzer = motion_analyzer
        self.event_generator = event_generator
        self.embedder = embedder
        self.store = store
        self.buffer: List[List[Dict[str, Any]]] = []

    def reset(self) -> None:
        self.buffer.clear()

    def observe(self, frame_observations: Sequence[Dict[str, Any]]) -> bool:
        self.buffer.append(list(frame_observations))
        return len(self.buffer) >= self.config.buffer.buffer_size

    def _write_event(self, sample_id: str, event: Dict[str, Any]) -> None:
        vector = self.embedder.embed(event["summary"])
        self.store.add(sample_id, "events", vector, event)

    def flush(self, sample_id: str) -> None:
        if not self.buffer:
            return

        trajectories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        entity_info: Dict[str, Dict[str, Any]] = {}
        for frame_observations in self.buffer:
            for obs in frame_observations:
                entity_id = obs["entity_id"]
                trajectories[entity_id].append(obs)
                entity_info[entity_id] = {
                    "entity_id": entity_id,
                    "tag": obs["tag"],
                    "label": obs["label"],
                }

        for entity_id, trajectory in trajectories.items():
            analysis = self.motion_analyzer.analyze_single_entity(entity_info[entity_id], trajectory)
            if analysis is None:
                continue
            event = self.event_generator.gen_trajectory_summary(analysis)
            self._write_event(sample_id, event)

        interaction_events = self.motion_analyzer.analyze_all_interactions(trajectories, entity_info)
        for interaction in interaction_events:
            event = self.event_generator.gen_interaction(interaction)
            self._write_event(sample_id, event)

        self.buffer.clear()
