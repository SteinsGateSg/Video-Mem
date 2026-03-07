"""
STG 配置模块
定义所有阈值、路径、模型参数等配置
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EmbeddingConfig:
    """Embedding 模型配置"""
    # 使用 sentence-transformers 本地模型（默认）
    use_local: bool = True
    local_model_name: str = "all-MiniLM-L6-v2"  # 轻量级，384维
    # 或使用 OpenAI 兼容 API
    openai_base_url: str = ""
    api_key: str = ""
    model: str = ""
    embedding_dim: int = 384  # all-MiniLM-L6-v2 的维度


@dataclass
class EntityMatchingConfig:
    """实体匹配参数"""
    iou_threshold: float = 0.3          # IoU 匹配最低阈值
    label_sim_threshold: float = 0.7     # label 语义相似度阈值
    combined_threshold: float = 0.4      # 综合得分阈值（用于匈牙利算法过滤）
    alpha: float = 0.5                   # IoU 权重 (1-alpha 为 label sim 权重)
    score_filter: float = 0.35           # 过滤置信度低于此值的检测对象


@dataclass 
class TrajectoryConfig:
    """轨迹记录参数"""
    movement_threshold: float = 10.0     # 像素位移超过此值才记录轨迹点
    static_threshold: float = 15.0       # 总位移小于此值的实体标记为静态
    static_min_frames: int = 5           # 判定静态实体需要的最小连续帧数


@dataclass
class BufferConfig:
    """缓冲区参数"""
    buffer_size: int = 5                 # 缓冲区满后触发 flush（帧数）
    cluster_sim_threshold: float = 0.85  # 聚类时的相似度阈值


@dataclass
class MotionConfig:
    """运动分析参数"""
    approach_distance_ratio: float = 0.7  # 距离缩小到原来的此比例视为接近
    velocity_threshold: float = 5.0       # 速度阈值（像素/帧）
    direction_change_angle: float = 45.0  # 方向变化角度阈值（度）


@dataclass
class STGConfig:
    """时空图谱总配置"""
    # 数据路径
    scene_graph_path: str = ""
    output_dir: str = "./stg_output"
    faiss_dir: str = "./stg_output/faiss"

    # 子配置
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    entity_matching: EntityMatchingConfig = field(default_factory=EntityMatchingConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)

    # 通用
    similarity_threshold: float = 0.3    # FAISS 检索相似度阈值（余弦相似度，越低越宽松）
    top_k: int = 10                      # FAISS 检索返回数量
    verbose: bool = True                 # 是否打印详细日志

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.faiss_dir).mkdir(parents=True, exist_ok=True)
