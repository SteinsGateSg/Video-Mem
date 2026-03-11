#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from stg import STGConfig, STGraphMemory
from stg.llm_adapter import LLMAdapterError, OpenAICompatibleLLMAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieve STG evidence and run grounded QA with an OpenAI-compatible LLM."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--api_base", default="", help="OpenAI-compatible API base.")
    parser.add_argument("--api_key", default="", help="OpenAI-compatible API key.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument(
        "--embedding_backend",
        default="auto",
        choices=["auto", "sentence_transformers", "hashing"],
        help="Embedding backend.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print the grounded prompt and evidence without calling an LLM.")
    parser.add_argument("--json", action="store_true", help="Print the final answer payload as JSON.")
    args = parser.parse_args()

    config = STGConfig(output_dir=args.output_dir)
    config.embedding.backend = args.embedding_backend
    config.search.top_k = args.top_k
    stg = STGraphMemory(config)

    bundle = stg.retrieve_evidence(query=args.query, sample_id=args.sample_id, top_k=args.top_k)
    llm_evidence = stg.format_evidence_for_llm(bundle)
    prompts = stg.build_grounded_prompt(args.query, llm_evidence)

    if args.dry_run:
        payload = {
            "mode": "dry_run",
            "evidence": llm_evidence,
            "system_prompt": prompts["system_prompt"],
            "user_prompt": prompts["user_prompt"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    adapter = OpenAICompatibleLLMAdapter(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.model,
    )

    try:
        answer = adapter.answer(prompts, llm_evidence)
    except LLMAdapterError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        print("Hint: pass --dry_run to inspect the evidence and prompt without an API call.", file=sys.stderr)
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - external API failures
        print(f"[ERROR] LLM request failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    payload = {
        "query": args.query,
        "model": args.model,
        "evidence": llm_evidence,
        "answer": answer,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print("=== Evidence ===")
        print(bundle["evidence_text"])
        print()
        print("=== Grounded Answer JSON ===")
        print(json.dumps(answer, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
