"""
离线构建时空图谱入口脚本

Usage:
    python -m scripts.build_stg \
        --scene_graph_path data/less_move/scene_graphs_target.json \
        --sample_id video_001 \
        --output_dir stg_output

    或使用测试数据:
    python -m scripts.build_stg \
        --scene_graph_path data/less_move/test_data/test_2frames.json \
        --sample_id test_video
"""

import argparse
import json
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stg.config import STGConfig, EmbeddingConfig, EntityMatchingConfig, TrajectoryConfig, BufferConfig
from stg.memory_manager import STGraphMemory


def parse_args():
    parser = argparse.ArgumentParser(description="Build Spatio-Temporal Graph Memory")
    
    parser.add_argument(
        "--scene_graph_path", type=str, required=True,
        help="Path to scene_graphs_target.json"
    )
    parser.add_argument(
        "--sample_id", type=str, default="video_001",
        help="Unique identifier for this video"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./stg_output",
        help="Output directory for FAISS indices and exports"
    )
    parser.add_argument(
        "--embedding_model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name for local embedding"
    )
    parser.add_argument(
        "--iou_threshold", type=float, default=0.3,
        help="IoU threshold for entity matching"
    )
    parser.add_argument(
        "--movement_threshold", type=float, default=10.0,
        help="Min pixel displacement to record trajectory"
    )
    parser.add_argument(
        "--buffer_size", type=int, default=5,
        help="Number of frames to buffer before flush"
    )
    parser.add_argument(
        "--score_filter", type=float, default=0.35,
        help="Minimum detection score to include an object"
    )
    parser.add_argument(
        "--export_graph", action="store_true",
        help="Export STG graph JSON after build"
    )
    parser.add_argument(
        "--export_registry", action="store_true",
        help="Export entity registry JSON after build"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed logs"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 构建配置
    config = STGConfig(
        scene_graph_path=args.scene_graph_path,
        output_dir=args.output_dir,
        faiss_dir=os.path.join(args.output_dir, "faiss"),
        embedding=EmbeddingConfig(
            use_local=True,
            local_model_name=args.embedding_model
        ),
        entity_matching=EntityMatchingConfig(
            iou_threshold=args.iou_threshold,
            score_filter=args.score_filter
        ),
        trajectory=TrajectoryConfig(
            movement_threshold=args.movement_threshold
        ),
        buffer=BufferConfig(
            buffer_size=args.buffer_size
        ),
        verbose=args.verbose
    )

    # 初始化并构建
    stg = STGraphMemory(config)
    
    print(f"\n{'='*60}")
    print(f"Starting STG Build")
    print(f"  Scene graph: {args.scene_graph_path}")
    print(f"  Sample ID:   {args.sample_id}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Embedding:   {args.embedding_model}")
    print(f"{'='*60}\n")

    result = stg.build(args.scene_graph_path, args.sample_id)

    # 可选导出
    if args.export_graph:
        stg.export_stg_graph(args.sample_id)

    if args.export_registry:
        stg.export_entity_registry(args.sample_id)

    # 简单的验证查询
    print(f"\n{'='*60}")
    print("Quick validation queries:")
    print(f"{'='*60}")

    test_queries = [
        "What is the player doing?",
        "Where is the basketball?",
        "Who is near the basketball hoop?",
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        search_result = stg.search(query, args.sample_id, top_k=3, threshold=0.0)
        events = search_result.get("events", [])
        if events:
            for evt in events[:3]:
                print(f"  [{evt['score']:.3f}] {evt['metadata'].get('summary', '')[:100]}")
        else:
            print("  No results found.")

    print(f"\n{'='*60}")
    print("Build complete! Use query_stg.py to run queries.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
