"""
在线查询时空图谱入口脚本

Usage:
    python -m scripts.query_stg \
        --sample_id video_001 \
        --output_dir stg_output \
        --query "What is the player doing near the basketball hoop?"

    交互模式:
    python -m scripts.query_stg \
        --sample_id video_001 \
        --output_dir stg_output \
        --interactive
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stg.config import STGConfig, EmbeddingConfig
from stg.memory_manager import STGraphMemory


def parse_args():
    parser = argparse.ArgumentParser(description="Query Spatio-Temporal Graph Memory")

    parser.add_argument(
        "--sample_id", type=str, default="video_001",
        help="Video/sample identifier (must match the one used in build)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./stg_output",
        help="Directory where FAISS indices were saved"
    )
    parser.add_argument(
        "--embedding_model", type=str, default="all-MiniLM-L6-v2",
        help="Sentence-transformers model name (must match build)"
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Single query to run"
    )
    parser.add_argument(
        "--top_k", type=int, default=10,
        help="Number of results to return"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Minimum similarity threshold"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive query mode"
    )
    parser.add_argument(
        "--context_mode", action="store_true",
        help="Output formatted context for LLM (get_context_for_qa)"
    )

    return parser.parse_args()


def run_single_query(stg: STGraphMemory, query: str, sample_id: str,
                      top_k: int, threshold: float, context_mode: bool):
    """执行单次查询"""
    if context_mode:
        context = stg.get_context_for_qa(query, sample_id, top_k=top_k)
        print(context)
        return

    result = stg.search(query, sample_id, top_k=top_k, threshold=threshold)

    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    events = result.get("events", [])
    if events:
        print(f"\n--- Events ({len(events)} results) ---")
        for i, evt in enumerate(events, 1):
            meta = evt["metadata"]
            print(
                f"  {i}. [{evt['score']:.3f}] [{meta.get('event_type', '')}] "
                f"{meta.get('summary', '')}"
            )
            frame_range = meta.get("frame_range", [])
            if frame_range:
                print(f"     Frames: {frame_range}")

    entities = result.get("entities", [])
    if entities:
        print(f"\n--- Entities ({len(entities)} results) ---")
        for i, ent in enumerate(entities, 1):
            meta = ent["metadata"]
            print(
                f"  {i}. [{ent['score']:.3f}] [{meta.get('tag', '')}] "
                f"{meta.get('description', '')}"
            )

    if not events and not entities:
        print("  No results found.")

    print()


def interactive_mode(stg: STGraphMemory, sample_id: str,
                      top_k: int, threshold: float, context_mode: bool):
    """交互式查询模式"""
    print(f"\n{'='*60}")
    print(f"Interactive Query Mode (sample_id: {sample_id})")
    print(f"Type 'quit' or 'exit' to stop")
    print(f"Type 'context' to toggle context mode")
    print(f"{'='*60}\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if query.lower() == "context":
            context_mode = not context_mode
            print(f"Context mode: {'ON' if context_mode else 'OFF'}")
            continue

        run_single_query(stg, query, sample_id, top_k, threshold, context_mode)


def main():
    args = parse_args()

    config = STGConfig(
        output_dir=args.output_dir,
        faiss_dir=os.path.join(args.output_dir, "faiss"),
        embedding=EmbeddingConfig(
            use_local=True,
            local_model_name=args.embedding_model
        ),
        verbose=False
    )

    stg = STGraphMemory(config)

    if args.interactive:
        interactive_mode(stg, args.sample_id, args.top_k,
                          args.threshold, args.context_mode)
    elif args.query:
        run_single_query(stg, args.query, args.sample_id,
                          args.top_k, args.threshold, args.context_mode)
    else:
        print("Please specify --query or --interactive")
        sys.exit(1)


if __name__ == "__main__":
    main()
