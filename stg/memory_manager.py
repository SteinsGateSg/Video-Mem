"""
STG 主控类
整合所有模块，提供构建和查询的统一接口
参考 main.py 的 TeleMemory 类设计
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
from tqdm import tqdm
from datetime import datetime
import pytz

from .config import STGConfig
from .utils import EmbeddingManager, frame_index_from_path, filter_objects_by_score
from .faiss_store import FAISSStore
from .entity_tracker import EntityTracker
from .motion_analyzer import MotionAnalyzer
from .event_generator import EventGenerator
from .immediate_update import ImmediateUpdater
from .buffer_update import BufferUpdater

logger = logging.getLogger(__name__)


class STGraphMemory:
    """
    时空图谱记忆系统主控类

    提供:
    1. build(): 从 scene_graphs_target.json 逐帧构建时空图谱
    2. search(): 根据查询检索相关记忆
    3. export_graph(): 导出时空图谱为 JSON
    """

    def __init__(self, config: STGConfig = None):
        if config is None:
            config = STGConfig()
        self.config = config

        # 初始化 embedding 管理器
        self.emb_manager = EmbeddingManager(config.embedding)

        # 初始化 FAISS 存储
        self.faiss_store = FAISSStore(
            config.faiss_dir,
            default_dim=config.embedding.embedding_dim
        )

        # 初始化各模块
        self.tracker = EntityTracker(config, self.emb_manager)
        self.motion_analyzer = MotionAnalyzer(config)
        self.event_gen = EventGenerator()

        self.immediate_updater = ImmediateUpdater(
            config, self.tracker, self.event_gen,
            self.emb_manager, self.faiss_store
        )
        self.buffer_updater = BufferUpdater(
            config, self.tracker, self.motion_analyzer,
            self.event_gen, self.emb_manager, self.faiss_store
        )

    # ============================================================
    # 构建时空图谱
    # ============================================================

    def build(self, scene_graph_path: str = None, sample_id: str = "video_001"):
        """
        从场景图 JSON 文件逐帧构建时空图谱

        Args:
            scene_graph_path: scene_graphs_target.json 的路径
            sample_id: 视频/样本的唯一标识
        """
        if scene_graph_path is None:
            scene_graph_path = self.config.scene_graph_path
        
        if not scene_graph_path:
            raise ValueError("scene_graph_path not specified")

        # 加载场景图数据
        print(f"Loading scene graphs from: {scene_graph_path}")
        with open(scene_graph_path, 'r', encoding='utf-8') as f:
            frames_data = json.load(f)
        
        total_frames = len(frames_data)
        print(f"Total frames: {total_frames}")

        all_events = []

        # 逐帧处理（类似于 "一轮一轮的对话"）
        for i, frame_data in enumerate(tqdm(frames_data, desc="Building STG")):
            # 即时更新
            result = self.immediate_updater.process_frame(frame_data, sample_id)
            all_events.extend(result["events"])

            # 将观测数据放入缓冲区
            self.buffer_updater.add_observation(
                result["buffered_observations"], sample_id
            )

        # 处理完所有帧后，flush 剩余缓冲区
        remaining_events = self.buffer_updater.flush(sample_id)
        all_events.extend(remaining_events)

        # 检测静态实体
        self.tracker.detect_static_entities()

        # 保存所有 FAISS 索引
        self.faiss_store.save_all(sample_id)

        # 打印统计信息
        summary = self.tracker.get_entity_summary()
        print(f"\n{'='*60}")
        print(f"STG Build Complete for sample: {sample_id}")
        print(f"{'='*60}")
        print(f"Total entities tracked: {summary['total_entities']}")
        print(f"  Static entities: {summary['static_entities']}")
        print(f"  Dynamic entities: {summary['dynamic_entities']}")
        print(f"Total events generated: {len(all_events)}")
        print(f"Events FAISS index size: {self.faiss_store.get_index_size(sample_id, 'events')}")
        print(f"{'='*60}")

        return {
            "entity_summary": summary,
            "total_events": len(all_events),
            "events": all_events
        }

    # ============================================================
    # 查询 / 检索
    # ============================================================

    def search(self, query: str, sample_id: str = "video_001",
               top_k: int = 10, threshold: float = None) -> Dict[str, Any]:
        """
        根据自然语言查询检索相关记忆

        Args:
            query: 查询文本（如 "谁投篮了"、"球在哪里"）
            sample_id: 视频/样本 ID
            top_k: 返回结果数量
            threshold: 相似度阈值

        Returns:
            {
                "events": [...],      # 相关事件
                "entities": [...],    # 相关实体
                "combined_text": str  # 合并后的上下文文本
            }
        """
        if threshold is None:
            threshold = self.config.similarity_threshold

        query_embedding = self.emb_manager.embed(query)

        # 搜索事件向量库
        event_results = self.faiss_store.search(
            sample_id, "events", query_embedding,
            top_k=top_k, threshold=threshold
        )

        # 搜索所有实体向量库
        entity_results = self.faiss_store.search_all_entities(
            sample_id, query_embedding,
            top_k=top_k // 2, threshold=threshold
        )

        # 合并文本
        combined_parts = []

        if event_results:
            combined_parts.append("=== Related Events ===")
            for r in event_results:
                combined_parts.append(
                    f"[{r['metadata'].get('event_type', 'event')}] "
                    f"{r['metadata'].get('summary', '')} "
                    f"(score: {r['score']:.3f})"
                )

        if entity_results:
            combined_parts.append("\n=== Related Entities ===")
            for r in entity_results:
                combined_parts.append(
                    f"[{r['metadata'].get('tag', 'entity')}] "
                    f"{r['metadata'].get('description', '')} "
                    f"(score: {r['score']:.3f})"
                )

        combined_text = "\n".join(combined_parts)

        return {
            "events": event_results,
            "entities": entity_results,
            "combined_text": combined_text
        }

    def search_events_only(self, query: str, sample_id: str = "video_001",
                            top_k: int = 5, threshold: float = None) -> List[Dict]:
        """只搜索事件向量库"""
        if threshold is None:
            threshold = self.config.similarity_threshold
        query_embedding = self.emb_manager.embed(query)
        return self.faiss_store.search(
            sample_id, "events", query_embedding, top_k, threshold
        )

    def search_entity(self, query: str, entity_id: str,
                       sample_id: str = "video_001",
                       top_k: int = 5, threshold: float = None) -> List[Dict]:
        """搜索指定实体的向量库"""
        if threshold is None:
            threshold = self.config.similarity_threshold
        query_embedding = self.emb_manager.embed(query)
        entity_key = f"entity_{entity_id}"
        return self.faiss_store.search(
            sample_id, entity_key, query_embedding, top_k, threshold
        )

    # ============================================================
    # 导出
    # ============================================================

    def export_entity_registry(self, sample_id: str = "video_001",
                                output_path: str = None) -> Dict:
        """
        导出实体注册表为 JSON
        """
        registry = {}
        for entity_id, record in self.tracker.entity_registry.items():
            registry[entity_id] = record.to_dict()

        if output_path is None:
            output_path = str(Path(self.config.output_dir) / f"{sample_id}_entity_registry.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

        print(f"Entity registry exported to: {output_path}")
        return registry

    def export_stg_graph(self, sample_id: str = "video_001",
                          output_path: str = None) -> Dict:
        """
        导出时空图谱为节点-边 JSON 格式
        参考 main.py 的 offline_build_graph_json 方法
        
        节点：每个事件
        边：基于时间顺序和实体共现的关联
        """
        self.faiss_store.load_or_create_index(sample_id, "events")

        if sample_id not in self.faiss_store.metadata_store:
            return {"nodes": [], "edges": [], "metadata": {}}

        metadata = self.faiss_store.metadata_store.get(sample_id, {}).get("events", [])
        if not metadata:
            return {"nodes": [], "edges": [], "metadata": {}}

        # 构建节点
        nodes = []
        for i, meta in enumerate(metadata):
            nodes.append({
                "id": i,
                "event_id": meta.get("event_id", ""),
                "event_type": meta.get("event_type", ""),
                "summary": meta.get("summary", ""),
                "involved_entities": meta.get("involved_entities", []),
                "frame_range": meta.get("frame_range", []),
                "timestamp": meta.get("timestamp", "")
            })

        # 构建边（基于实体共现和时间顺序）
        edges = []
        n = len(nodes)

        for i in range(n):
            entities_i = set(nodes[i].get("involved_entities", []))
            frame_i = nodes[i].get("frame_range", [0, 0])

            for j in range(i + 1, min(i + 20, n)):  # 限制搜索范围
                entities_j = set(nodes[j].get("involved_entities", []))
                frame_j = nodes[j].get("frame_range", [0, 0])

                # 共享实体的事件之间建边
                shared = entities_i & entities_j
                if shared and len(shared) > 0:
                    edges.append({
                        "source": i,
                        "target": j,
                        "relation": "entity_co_occurrence",
                        "shared_entities": list(shared),
                        "weight": len(shared)
                    })

        result = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "sample_id": sample_id,
                "num_nodes": len(nodes),
                "num_edges": len(edges),
                "generated_at": datetime.now(pytz.UTC).isoformat()
            }
        }

        if output_path is None:
            output_path = str(Path(self.config.output_dir) / f"{sample_id}_stg_graph.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"STG graph exported to: {output_path}")
        return result

    # ============================================================
    # 供 LLM 使用的上下文生成
    # ============================================================

    def _split_sub_queries(self, question: str) -> List[str]:
        """
        将复合问题拆分为子查询，提升检索覆盖率。
        简单规则拆分：按 ? 和常见连接词分句。
        """
        import re
        # 按问号分句
        parts = re.split(r'[?？]', question)
        sub_queries = []
        for p in parts:
            p = p.strip()
            if len(p) > 5:  # 过滤太短的
                sub_queries.append(p)
        
        # 如果只有一句或拆分失败，返回原始问题
        if len(sub_queries) <= 1:
            return [question]
        return sub_queries

    def get_context_for_qa(self, question: str, sample_id: str = "video_001",
                            top_k: int = 10) -> str:
        """
        为视频问答生成上下文文本
        直接返回可以塞进 LLM prompt 的结构化文本

        支持复合问题：自动拆分子查询，分别检索后合并去重。

        Args:
            question: 用户问题
            sample_id: 视频 ID
            top_k: 检索结果数量

        Returns:
            格式化的上下文文本
        """
        # 拆分子查询
        sub_queries = self._split_sub_queries(question)
        
        # 对每个子查询分别检索，合并去重
        all_events = []
        all_entities = []
        seen_event_ids = set()
        seen_entity_descs = set()
        
        per_query_k = max(top_k, top_k // len(sub_queries) + 5)
        
        for sq in sub_queries:
            result = self.search(sq, sample_id, top_k=per_query_k)
            for evt in result.get("events", []):
                eid = evt["metadata"].get("event_id", "")
                if eid not in seen_event_ids:
                    seen_event_ids.add(eid)
                    all_events.append(evt)
            for ent in result.get("entities", []):
                desc = ent["metadata"].get("description", "")
                if desc not in seen_entity_descs:
                    seen_entity_descs.add(desc)
                    all_entities.append(ent)
        
        # 也用原始完整问题搜一次
        if len(sub_queries) > 1:
            result = self.search(question, sample_id, top_k=per_query_k)
            for evt in result.get("events", []):
                eid = evt["metadata"].get("event_id", "")
                if eid not in seen_event_ids:
                    seen_event_ids.add(eid)
                    all_events.append(evt)
            for ent in result.get("entities", []):
                desc = ent["metadata"].get("description", "")
                if desc not in seen_entity_descs:
                    seen_entity_descs.add(desc)
                    all_entities.append(ent)

        # 按 score 排序后截取
        all_events.sort(key=lambda x: x["score"], reverse=True)
        all_entities.sort(key=lambda x: x["score"], reverse=True)
        events = all_events[:top_k * 2]  # 多给一些上下文
        entities = all_entities[:top_k]

        context_parts = [
            "=== Spatio-Temporal Memory Context ===",
            f"Query: {question}",
            ""
        ]

        # 添加相关事件
        if events:
            context_parts.append("--- Relevant Events ---")
            for i, evt in enumerate(events, 1):
                meta = evt["metadata"]
                context_parts.append(
                    f"{i}. [{meta.get('event_type', '')}] {meta.get('summary', '')} "
                    f"(frames: {meta.get('frame_range', [])})"
                )
            context_parts.append("")

        # 添加相关实体信息
        if entities:
            context_parts.append("--- Relevant Entity States ---")
            for i, ent in enumerate(entities, 1):
                meta = ent["metadata"]
                context_parts.append(
                    f"{i}. {meta.get('tag', '')} - {meta.get('description', '')}"
                )
            context_parts.append("")

        # 添加实体摘要
        # 优先使用 tracker（build 模式下有数据），否则从磁盘统计实体索引数
        entity_summary = self.tracker.get_entity_summary()
        if entity_summary['total_entities'] > 0:
            context_parts.append(
                f"Scene contains {entity_summary['total_entities']} tracked entities "
                f"({entity_summary['static_entities']} static, "
                f"{entity_summary['dynamic_entities']} dynamic)"
            )
        else:
            # 查询模式：从磁盘统计
            entity_count = self.faiss_store.count_entity_keys(sample_id)
            events_size = self.faiss_store.get_index_size(sample_id, 'events')
            context_parts.append(
                f"Scene contains {entity_count} tracked entities, "
                f"{events_size} events recorded."
            )

        return "\n".join(context_parts)
