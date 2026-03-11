#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from stg import STGConfig, STGraphMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an STG memory from scene-graph JSON.")
    parser.add_argument("--scene_graph_path", required=True, help="Path to scene graph JSON.")
    parser.add_argument("--sample_id", required=True, help="Sample/video id.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument(
        "--embedding_backend",
        default="auto",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    args = parser.parse_args()

    config = STGConfig(output_dir=args.output_dir)
    config.embedding.backend = args.embedding_backend
    stg = STGraphMemory(config)
    stats = stg.build(scene_graph_path=args.scene_graph_path, sample_id=args.sample_id)

    output_root = Path(args.output_dir).resolve()
    sample_root = output_root / args.sample_id

    print("=== Build Summary ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("=== Artifacts ===")
    print(f"entity_registry: {sample_root / 'entity_registry.json'}")
    print(f"stg_graph: {sample_root / 'stg_graph.json'}")
    print(f"vector_store_dir: {output_root / 'store' / args.sample_id}")


if __name__ == "__main__":
    main()
