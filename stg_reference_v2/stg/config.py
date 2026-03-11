from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Embedding backend configuration.

    backend:
        - "auto": prefer sentence-transformers, fall back to hashing
        - "sentence_transformers": force SentenceTransformer
        - "hashing": lightweight deterministic local fallback
    """

    backend: str = "auto"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dim: int = 384
    normalize: bool = True
    batch_size: int = 32
    device: str = "cpu"
    random_seed: int = 42


@dataclass
class EntityMatchingConfig:
    """Entity association hyper-parameters.

    Retrieval thresholds must remain separate from these matching thresholds.
    """

    detection_score_threshold: float = 0.35
    iou_weight: float = 0.50
    combined_threshold: float = 0.40
    label_threshold: float = 0.35
    min_iou_threshold: float = 0.01
    movement_event_threshold: float = 10.0
    miss_tolerance: int = 0


@dataclass
class TrajectoryConfig:
    movement_threshold: float = 10.0
    static_threshold: float = 15.0
    jump_vertical_threshold: float = 25.0
    min_frames_for_summary: int = 2


@dataclass
class BufferConfig:
    buffer_size: int = 5


@dataclass
class MotionConfig:
    approach_distance_ratio: float = 0.70
    depart_distance_ratio: float = 1.43
    moving_together_angle_deg: float = 30.0
    direction_change_angle_deg: float = 45.0
    min_interaction_distance: float = 5.0


@dataclass
class SearchConfig:
    top_k: int = 8
    similarity_threshold: float = 0.15
    entity_top_k: int = 4
    dense_candidate_multiplier: int = 3
    enable_subquery_decomposition: bool = True
    rerank_entity_bonus: float = 0.12
    rerank_relation_bonus: float = 0.10
    rerank_temporal_bonus: float = 0.08
    rerank_intent_bonus: float = 0.12


@dataclass
class STGConfig:
    output_dir: str = "./outputs"
    store_dir: Optional[str] = None
    clear_existing_sample: bool = True
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    matching: EntityMatchingConfig = field(default_factory=EntityMatchingConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.store_dir is None:
            self.store_dir = str(Path(self.output_dir) / "store")

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def store_path(self) -> Path:
        return Path(self.store_dir)
