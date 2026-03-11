#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from stg import STGConfig, STGraphMemory


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scene_graph_path = root / "data" / "toy_scene_graphs.json"
    sample_id = "toy_video"

    config = STGConfig(output_dir=str(root / "outputs"))
    config.embedding.backend = "hashing"  # guaranteed local demo
    config.buffer.buffer_size = 3
    config.search.top_k = 6
    config.search.entity_top_k = 3
    config.search.similarity_threshold = 0.05

    stg = STGraphMemory(config)
    stats = stg.build(scene_graph_path=scene_graph_path, sample_id=sample_id)
    print("=== Demo Build Stats ===")
    print(stats)
    print()

    questions = [
        "What happened to the player?",
        "What happened to the basketball?",
        "Did the player and basketball move together?",
        "Which relations changed?",
    ]
    for question in questions:
        print("=" * 100)
        print(f"Question: {question}")
        print(stg.get_context_for_qa(question, sample_id=sample_id, top_k=6))
        print()


if __name__ == "__main__":
    main()
