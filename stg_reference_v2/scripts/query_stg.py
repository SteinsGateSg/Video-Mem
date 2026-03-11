#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from stg import STGConfig, STGraphMemory


def main() -> None:
    parser = argparse.ArgumentParser(description="Query an STG memory and return structured evidence.")
    parser.add_argument("--sample_id", required=True, help="Sample/video id.")
    parser.add_argument("--query", required=True, help="Natural language question.")
    parser.add_argument("--output_dir", default="./outputs", help="Output directory.")
    parser.add_argument("--top_k", type=int, default=8, help="Top-k retrieval count.")
    parser.add_argument(
        "--embedding_backend",
        default="auto",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    parser.add_argument("--json", action="store_true", help="Print structured evidence JSON.")
    args = parser.parse_args()

    config = STGConfig(output_dir=args.output_dir)
    config.embedding.backend = args.embedding_backend
    config.search.top_k = args.top_k
    stg = STGraphMemory(config)

    bundle = stg.retrieve_evidence(query=args.query, sample_id=args.sample_id, top_k=args.top_k)
    if args.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
    else:
        print(bundle["evidence_text"])


if __name__ == "__main__":
    main()
