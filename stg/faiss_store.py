"""
FAISS 向量库管理模块
参考 main.py 的存取逻辑，管理实体向量库和事件向量库
"""

import json
import threading
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime
import pytz
import logging

logger = logging.getLogger(__name__)


class FAISSStore:
    """
    FAISS 向量库管理器
    支持按 (sample_id, key) 分区存取
    key 可以是 "events", "spatial", 或 "entity_{entity_id}" 等
    """

    def __init__(self, faiss_dir: str, default_dim: int = 384):
        self.faiss_dir = Path(faiss_dir)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        self.default_dim = default_dim

        # 内存中的索引和元数据缓存
        self.faiss_store: Dict[str, Dict[str, faiss.Index]] = {}
        self.metadata_store: Dict[str, Dict[str, List[Dict]]] = {}

        # 锁管理
        self._file_locks: Dict[str, threading.Lock] = {}
        self._store_lock = threading.Lock()  # 保护 faiss_store/metadata_store 的结构修改

    def _get_index_path(self, sample_id: str, key: str) -> Path:
        return self.faiss_dir / f"{sample_id}_{key}.index"

    def _get_metadata_path(self, sample_id: str, key: str) -> Path:
        return self.faiss_dir / f"{sample_id}_{key}_meta.json"

    def _get_file_lock(self, sample_id: str, key: str) -> threading.Lock:
        lock_key = f"{sample_id}_{key}"
        if lock_key not in self._file_locks:
            self._file_locks[lock_key] = threading.Lock()
        return self._file_locks[lock_key]

    def load_or_create_index(self, sample_id: str, key: str, dim: int = None):
        """加载或创建 FAISS 索引"""
        dim = dim or self.default_dim
        index_path = self._get_index_path(sample_id, key)
        meta_path = self._get_metadata_path(sample_id, key)

        with self._store_lock:
            if sample_id not in self.faiss_store:
                self.faiss_store[sample_id] = {}
                self.metadata_store[sample_id] = {}

            # 已加载则跳过
            if key in self.faiss_store[sample_id]:
                return

            if index_path.exists() and meta_path.exists():
                index = faiss.read_index(str(index_path))
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            else:
                index = faiss.IndexFlatIP(dim)  # 内积搜索（需归一化向量）
                metadata = []

            self.faiss_store[sample_id][key] = index
            self.metadata_store[sample_id][key] = metadata

    def save_index(self, sample_id: str, key: str):
        """将索引和元数据持久化到磁盘"""
        if sample_id not in self.faiss_store or key not in self.faiss_store[sample_id]:
            return

        file_lock = self._get_file_lock(sample_id, key)
        with file_lock:
            index = self.faiss_store[sample_id][key]
            metadata = self.metadata_store[sample_id][key]

            index_path = self._get_index_path(sample_id, key)
            meta_path = self._get_metadata_path(sample_id, key)

            faiss.write_index(index, str(index_path))
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

    def add_memory(self, sample_id: str, key: str, embedding: np.ndarray, 
                   metadata_entry: Dict, dim: int = None):
        """
        添加一条记忆到向量库
        embedding: 已归一化的 embedding 向量
        metadata_entry: 关联的元数据
        """
        self.load_or_create_index(sample_id, key, dim=dim or len(embedding))

        emb = np.array(embedding, dtype=np.float32).reshape(1, -1)
        # 确保归一化
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm

        file_lock = self._get_file_lock(sample_id, key)
        with file_lock:
            self.faiss_store[sample_id][key].add(emb)
            self.metadata_store[sample_id][key].append(metadata_entry)

    def add_memories_batch(self, sample_id: str, key: str, 
                           embeddings: np.ndarray, metadata_entries: List[Dict],
                           dim: int = None):
        """批量添加记忆"""
        if len(embeddings) == 0:
            return

        self.load_or_create_index(sample_id, key, dim=dim or embeddings.shape[1])

        embs = np.array(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        # 归一化
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs = embs / norms

        file_lock = self._get_file_lock(sample_id, key)
        with file_lock:
            self.faiss_store[sample_id][key].add(embs)
            self.metadata_store[sample_id][key].extend(metadata_entries)

    def search(self, sample_id: str, key: str, query_embedding: np.ndarray,
               top_k: int = 10, threshold: float = 0.5) -> List[Dict]:
        """
        搜索最相似的记忆
        返回: List[{"score": float, "metadata": Dict}]
        """
        self.load_or_create_index(sample_id, key, dim=len(query_embedding))

        index = self.faiss_store[sample_id][key]
        metadata = self.metadata_store[sample_id][key]

        if index.ntotal == 0:
            return []

        query_emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(query_emb)
        if norm > 0:
            query_emb = query_emb / norm

        k = min(top_k, index.ntotal)
        D, I = index.search(query_emb, k)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            results.append({
                "score": float(score),
                "metadata": metadata[idx],
                "faiss_idx": int(idx)
            })

        return results

    def search_all_entities(self, sample_id: str, query_embedding: np.ndarray,
                            top_k: int = 5, threshold: float = 0.5) -> List[Dict]:
        """
        搜索所有实体向量库，返回合并结果
        """
        if sample_id not in self.faiss_store:
            return []

        all_results = []
        for key in list(self.faiss_store[sample_id].keys()):
            if key.startswith("entity_"):
                results = self.search(sample_id, key, query_embedding, top_k, threshold)
                for r in results:
                    r["entity_key"] = key
                all_results.extend(results)

        # 按 score 排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    def get_index_size(self, sample_id: str, key: str) -> int:
        """获取索引中向量数量"""
        if sample_id in self.faiss_store and key in self.faiss_store[sample_id]:
            return self.faiss_store[sample_id][key].ntotal
        return 0

    def save_all(self, sample_id: str):
        """保存指定 sample_id 下的所有索引"""
        if sample_id not in self.faiss_store:
            return
        for key in self.faiss_store[sample_id]:
            self.save_index(sample_id, key)

    def clear_memory(self, sample_id: str):
        """清除指定 sample_id 的所有内存缓存"""
        with self._store_lock:
            if sample_id in self.faiss_store:
                del self.faiss_store[sample_id]
            if sample_id in self.metadata_store:
                del self.metadata_store[sample_id]

    def rebuild_index(self, sample_id: str, key: str, 
                      embeddings: np.ndarray, metadata_entries: List[Dict]):
        """
        完全重建索引（用于 flush 后覆盖写入）
        """
        dim = embeddings.shape[1] if embeddings.ndim == 2 else len(embeddings)
        
        with self._store_lock:
            if sample_id not in self.faiss_store:
                self.faiss_store[sample_id] = {}
                self.metadata_store[sample_id] = {}

        new_index = faiss.IndexFlatIP(dim)
        
        embs = np.array(embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        embs = embs / norms
        new_index.add(embs)

        file_lock = self._get_file_lock(sample_id, key)
        with file_lock:
            self.faiss_store[sample_id][key] = new_index
            self.metadata_store[sample_id][key] = list(metadata_entries)
