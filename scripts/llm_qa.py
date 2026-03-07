"""
LLM + STG 联合问答示例脚本

Usage:
    python scripts/llm_qa.py \
        --sample_id video_001 \
        --output_dir stg_output \
        --api_base "https://api.openai.com/v1" \
        --api_key "sk-xxx" \
        --model "gpt-4" \
        --question "What is the player doing?"
"""
import argparse
from openai import OpenAI
from stg.config import STGConfig, EmbeddingConfig
from stg.memory_manager import STGraphMemory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_id", default="video_001")
    parser.add_argument("--output_dir", default="./stg_output")
    parser.add_argument("--api_base", required=True, help="LLM API base URL")
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--model", default="gpt-4")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()

    # 加载 STG
    config = STGConfig(
        output_dir=args.output_dir,
        faiss_dir=f"{args.output_dir}/faiss"
    )
    stg = STGraphMemory(config)

    # 检索
    context = stg.get_context_for_qa(args.question, args.sample_id, top_k=args.top_k)
    print("=== Retrieved Context ===")
    print(context)
    print()

    # 调用 LLM
    client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    prompt = f"""Based on the following spatio-temporal memory retrieved from video analysis, answer the question.

{context}

Question: {args.question}

Answer:"""

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a video understanding assistant. Answer questions based on the provided spatio-temporal context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    print("=== LLM Answer ===")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    main()

