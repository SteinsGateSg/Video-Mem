"""
STG 工具函数模块
包含 IoU 计算、box 操作、embedding 工具、关系差异比较等
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math


# ============================================================
# Box 操作
# ============================================================

def box_center(box: List[float]) -> Tuple[float, float]:
    """计算 box 的几何中心 (cx, cy)"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_area(box: List[float]) -> float:
    """计算 box 面积"""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """计算两个 box 的 IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_iou_matrix(boxes_a: List[List[float]], boxes_b: List[List[float]]) -> np.ndarray:
    """
    计算两组 boxes 之间的 IoU 矩阵
    返回 shape: [len(boxes_a), len(boxes_b)]
    """
    m, n = len(boxes_a), len(boxes_b)
    iou_matrix = np.zeros((m, n), dtype=np.float32)
    for i in range(m):
        for j in range(n):
            iou_matrix[i, j] = compute_iou(boxes_a[i], boxes_b[j])
    return iou_matrix


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点的欧氏距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_displacement(box_a: List[float], box_b: List[float]) -> float:
    """计算两个 box 中心点的位移"""
    c1 = box_center(box_a)
    c2 = box_center(box_b)
    return euclidean_distance(c1, c2)


def compute_direction(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算从 p1 到 p2 的方向角（度数，0-360）"""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle = math.degrees(math.atan2(dy, dx))
    return angle % 360


def compute_velocity(displacement: float, frame_gap: int = 1) -> float:
    """计算速度（像素/帧间隔）"""
    if frame_gap == 0:
        return 0.0
    return displacement / frame_gap


# ============================================================
# 关系操作
# ============================================================

def extract_relations_set(obj: Dict) -> set:
    """
    提取一个对象的所有关系，返回可比较的 frozenset
    关系表示为 (predicate, target_tag) 的集合
    """
    relations = set()
    
    for rel in obj.get("subject_relations", []):
        relations.add(("subject", rel.get("predicate", ""), rel.get("object_tag", "")))
    
    for rel in obj.get("object_relations", []):
        relations.add(("object", rel.get("predicate", ""), rel.get("subject_tag", "")))
    
    return relations


def diff_relations(prev_obj: Dict, curr_obj: Dict) -> Dict[str, list]:
    """
    比较两帧中同一实体的关系变化
    返回：{"added": [...], "removed": [...]}
    """
    prev_rels = extract_relations_set(prev_obj)
    curr_rels = extract_relations_set(curr_obj)
    
    added = curr_rels - prev_rels
    removed = prev_rels - curr_rels
    
    return {
        "added": [{"role": r[0], "predicate": r[1], "tag": r[2]} for r in added],
        "removed": [{"role": r[0], "predicate": r[1], "tag": r[2]} for r in removed]
    }


def has_relation_change(prev_obj: Dict, curr_obj: Dict) -> bool:
    """判断两帧中同一实体的关系是否发生了变化"""
    diff = diff_relations(prev_obj, curr_obj)
    return len(diff["added"]) > 0 or len(diff["removed"]) > 0


# ============================================================
# Embedding 工具
# ============================================================

class EmbeddingManager:
    """统一管理 embedding 生成，支持本地模型和 API"""
    
    def __init__(self, config):
        self.config = config
        self._model = None
    
    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            if self.config.use_local:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.config.local_model_name)
                # 更新实际维度
                self.config.embedding_dim = self._model.get_sentence_embedding_dimension()
            else:
                from openai import OpenAI
                self._model = OpenAI(
                    base_url=self.config.openai_base_url,
                    api_key=self.config.api_key
                )
    
    def embed(self, text: str) -> np.ndarray:
        """生成单条文本的 embedding"""
        self._load_model()
        
        if self.config.use_local:
            emb = self._model.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)
        else:
            response = self._model.embeddings.create(
                model=self.config.model,
                input=text
            )
            emb = np.array(response.data[0].embedding, dtype=np.float32)
            emb = emb / np.linalg.norm(emb)
            return emb
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """批量生成 embeddings"""
        self._load_model()
        
        if self.config.use_local:
            embs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.array(embs, dtype=np.float32)
        else:
            results = []
            # API 可能有批量限制，分批处理
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = self._model.embeddings.create(
                    model=self.config.model,
                    input=batch
                )
                for item in response.data:
                    emb = np.array(item.embedding, dtype=np.float32)
                    emb = emb / np.linalg.norm(emb)
                    results.append(emb)
            return np.array(results, dtype=np.float32)
    
    def cosine_similarity(self, emb_a: np.ndarray, emb_b: np.ndarray) -> float:
        """计算两个 embedding 的余弦相似度（假设已归一化）"""
        return float(np.dot(emb_a.flatten(), emb_b.flatten()))

    def get_dim(self) -> int:
        """获取 embedding 维度"""
        self._load_model()
        return self.config.embedding_dim


# ============================================================
# 场景图对象描述生成
# ============================================================

def object_to_description(obj: Dict) -> str:
    """
    将场景图对象转换为自然语言描述，用于 embedding
    """
    parts = []
    label = obj.get("label", "unknown")
    tag = obj.get("tag", label)
    attributes = obj.get("attributes", "")
    
    parts.append(f"{tag}")
    if attributes:
        parts.append(f"({attributes})")
    
    # 添加关系描述
    for rel in obj.get("subject_relations", []):
        parts.append(f"; {rel.get('predicate', '')}")
    
    for rel in obj.get("object_relations", []):
        parts.append(f"; {rel.get('predicate', '')}")
    
    return " ".join(parts)


def entity_state_description(entity_record: Dict, frame_idx: int = None) -> str:
    """
    生成实体状态的自然语言描述，用于写入向量库
    """
    parts = []
    tag = entity_record.get("tag", "unknown")
    label = entity_record.get("label", "unknown")
    
    parts.append(f"{tag} (a {label})")
    
    # 最新属性
    attr_history = entity_record.get("attributes_history", [])
    if attr_history:
        latest_attr = attr_history[-1]
        parts.append(f"appears as: {latest_attr.get('attributes', '')}")
    
    # 静态标记
    if entity_record.get("is_static", False):
        parts.append("(static/background object)")
    
    # 轨迹摘要
    trajectory = entity_record.get("trajectory", [])
    if len(trajectory) >= 2 and not entity_record.get("is_static", False):
        first = trajectory[0]
        last = trajectory[-1]
        displacement = euclidean_distance(
            tuple(first["center"]), tuple(last["center"])
        )
        parts.append(
            f"moved from ({first['center'][0]:.0f},{first['center'][1]:.0f}) "
            f"to ({last['center'][0]:.0f},{last['center'][1]:.0f}), "
            f"total displacement: {displacement:.0f}px "
            f"over frames {first['frame_idx']}-{last['frame_idx']}"
        )
    
    return "; ".join(parts)


def frame_index_from_path(image_path: str) -> int:
    """
    从 image_path 中提取帧序号
    例如: '/home/.../frame_000000.png' -> 0
    """
    import re
    match = re.search(r'frame_(\d+)', image_path)
    if match:
        return int(match.group(1))
    return -1


def filter_objects_by_score(objects: List[Dict], min_score: float = 0.35) -> List[Dict]:
    """过滤低置信度对象"""
    return [obj for obj in objects if obj.get("score", 0) >= min_score]
