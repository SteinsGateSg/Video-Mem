from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None  # type: ignore

from .utils import dump_json, ensure_dir, load_json


class VectorPartition:
    def __init__(self, dim: int):
        self.dim = dim
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict[str, Any]] = []
        self._index = None
        self._dirty = True
        self._dedupe_keys: Dict[str, str] = {}

    def add(self, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        dedupe_key = str(metadata.get("dedupe_key", "")).strip()
        if dedupe_key and dedupe_key in self._dedupe_keys:
            return None
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self.dim:
            raise ValueError(f"Expected vector dim {self.dim}, got {vec.shape[0]}")
        self.vectors.append(vec)
        self.metadata.append(metadata)
        if dedupe_key:
            self._dedupe_keys[dedupe_key] = str(metadata.get("memory_id", ""))
        self._dirty = True
        return metadata

    def _build_index(self) -> None:
        if faiss is None:
            self._index = None
            self._dirty = False
            return
        index = faiss.IndexFlatIP(self.dim)
        if self.vectors:
            matrix = np.stack(self.vectors, axis=0).astype(np.float32)
            index.add(matrix)
        self._index = index
        self._dirty = False

    def search(self, query: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        if not self.vectors:
            return []
        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dim:
            raise ValueError(f"Expected query dim {self.dim}, got {q.shape[1]}")
        if faiss is not None:
            if self._index is None or self._dirty:
                self._build_index()
            scores, indices = self._index.search(q, min(top_k, len(self.vectors)))
            scores_arr = scores[0].tolist()
            idx_arr = indices[0].tolist()
        else:
            matrix = np.stack(self.vectors, axis=0).astype(np.float32)
            all_scores = (matrix @ q[0]).astype(np.float32)
            idx_arr = np.argsort(all_scores)[::-1][:top_k].tolist()
            scores_arr = [float(all_scores[idx]) for idx in idx_arr]
        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores_arr, idx_arr):
            if idx < 0:
                continue
            results.append(
                {
                    "score": float(score),
                    "metadata": self.metadata[int(idx)],
                    "rank": len(results) + 1,
                }
            )
        return results

    def vectors_matrix(self) -> np.ndarray:
        if not self.vectors:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(self.vectors, axis=0).astype(np.float32)


class VectorStore:
    """A lightweight persistent vector store with optional FAISS acceleration."""

    def __init__(self, root_dir: str | Path, dim: int):
        self.root_dir = ensure_dir(root_dir)
        self.dim = dim
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, VectorPartition]] = {}

    def _sample_dir(self, sample_id: str) -> Path:
        return ensure_dir(self.root_dir / sample_id)

    def _paths(self, sample_id: str, key: str) -> Dict[str, Path]:
        base = self._sample_dir(sample_id) / key
        return {
            "vectors": base.with_suffix(".npy"),
            "metadata": base.with_name(f"{base.name}_meta.json"),
            "faiss": base.with_suffix(".index"),
        }

    def _ensure_partition(self, sample_id: str, key: str) -> VectorPartition:
        if sample_id not in self._data:
            self._data[sample_id] = {}
        if key not in self._data[sample_id]:
            self._data[sample_id][key] = VectorPartition(dim=self.dim)
        return self._data[sample_id][key]

    def clear_sample(self, sample_id: str) -> None:
        with self._lock:
            self._data.pop(sample_id, None)
            sample_dir = self._sample_dir(sample_id)
            if sample_dir.exists():
                for path in sample_dir.iterdir():
                    if path.is_file():
                        try:
                            path.unlink()
                        except FileNotFoundError:
                            continue

    def add(self, sample_id: str, key: str, vector: np.ndarray, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        with self._lock:
            partition = self._ensure_partition(sample_id, key)
            metadata = dict(metadata)
            if "memory_id" not in metadata:
                metadata["memory_id"] = f"{key}_{len(partition.metadata):06d}"
            return partition.add(vector, metadata)

    def add_batch(self, sample_id: str, key: str, vectors: np.ndarray, metadatas: List[Dict[str, Any]]) -> None:
        for vector, metadata in zip(vectors, metadatas):
            self.add(sample_id, key, vector, metadata)

    def save_sample(self, sample_id: str) -> None:
        if sample_id not in self._data:
            return
        for key, partition in self._data[sample_id].items():
            paths = self._paths(sample_id, key)
            np.save(paths["vectors"], partition.vectors_matrix())
            dump_json(paths["metadata"], partition.metadata)
            if faiss is not None:
                partition._build_index()
                faiss.write_index(partition._index, str(paths["faiss"]))

    def _load_partition(self, sample_id: str, key: str) -> VectorPartition:
        partition = self._ensure_partition(sample_id, key)
        if partition.metadata or partition.vectors:
            return partition
        paths = self._paths(sample_id, key)
        if not paths["vectors"].exists() or not paths["metadata"].exists():
            return partition
        matrix = np.load(paths["vectors"])
        metadata = load_json(paths["metadata"])
        for vector, meta in zip(matrix, metadata):
            partition.add(np.asarray(vector, dtype=np.float32), meta)
        partition._dirty = True
        return partition

    def search(self, sample_id: str, key: str, query: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        partition = self._load_partition(sample_id, key)
        return partition.search(query, top_k=top_k)

    def all_metadata(self, sample_id: str, key: str) -> List[Dict[str, Any]]:
        partition = self._load_partition(sample_id, key)
        return list(partition.metadata)

    def available_keys(self, sample_id: str) -> List[str]:
        keys = set(self._data.get(sample_id, {}).keys())
        sample_dir = self._sample_dir(sample_id)
        if sample_dir.exists():
            for path in sample_dir.iterdir():
                if path.name.endswith("_meta.json"):
                    keys.add(path.name[:-10])
                elif path.suffix in {".npy", ".index"}:
                    keys.add(path.stem)
        return sorted(keys)
