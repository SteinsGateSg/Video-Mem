"""快速测试检索是否工作"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stg.config import STGConfig
from stg.memory_manager import STGraphMemory

cfg = STGConfig(output_dir='stg_output', faiss_dir='stg_output/faiss', verbose=False)
stg = STGraphMemory(cfg)

# 测试1: 搜索事件
print("=== Test 1: Search events ===")
r = stg.search('player shooting basketball', sample_id='video_001', top_k=5)
print(f"Events found: {len(r['events'])}")
for e in r['events']:
    print(f"  [{e['score']:.3f}] {e['metadata']['summary'][:120]}")

print(f"\nEntities found: {len(r['entities'])}")
for e in r['entities']:
    print(f"  [{e['score']:.3f}] {e['metadata']['description'][:120]}")

# 测试2: get_context_for_qa
print("\n=== Test 2: get_context_for_qa ===")
ctx = stg.get_context_for_qa(
    "What are these players doing? How many people appeared?",
    sample_id='video_001', top_k=15
)
print(ctx)
