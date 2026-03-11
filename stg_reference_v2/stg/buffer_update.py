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
