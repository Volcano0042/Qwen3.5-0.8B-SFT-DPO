"""
extract_prompts.py — 从 PsyDTCorpus 抽取 prompt 截断点

策略:
  - 分层采样 + 最低配额（避免主流主题主导，保护长尾主题）
  - 每个对话抽 1 个截断点（保证 prompt 独立性）
  - 优先选对话中段的 assistant 轮
  - 保留完整上下文（system + 截断点前所有消息）

输出: data/prompts.jsonl
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from modelscope.msdatasets import MsDataset

# ============================================================
# 配置
# ============================================================
SEED = 42
OUTPUT_DIR = Path("/home/b311/data/zhangwei/zsq/Psy/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "prompts.jsonl"

# 每个主题的目标 prompt 数（手动调节后的最优分布，共1250条）
TARGET_QUOTA = {
    "婚恋": 200,
    "情绪": 150,
    "人际": 150,
    "家庭": 130,
    "治疗": 120,
    "成长": 100,
    "自我": 95,
    "行为": 90,
    "职场": 60,
    "社会": 55,
    "性心理": 50,
    "心理学知识": 50,
}
TOTAL_TARGET = sum(TARGET_QUOTA.values())

random.seed(SEED)


# ============================================================
# 截断点选择
# ============================================================
def pick_cut_index(dialog: list) -> int | None:
    """
    从对话中选一个 assistant 截断点。
    策略:
      - 必须是 assistant 轮
      - 跳过第一个 assistant 轮（开场白）
      - 优先从中段选（25%-75% 之间）
    """
    # 找出所有 assistant 的位置
    asst_indices = [i for i, m in enumerate(dialog) if m["role"] == "assistant"]

    # 跳过第一个 assistant 轮
    candidates = asst_indices[1:]

    if len(candidates) < 2:
        return None  # 候选太少，跳过这个对话

    # 取中段 50% 的索引
    n = len(candidates)
    mid_start = n // 4
    mid_end = (n * 3) // 4 + 1  # +1 保证至少有一个
    mid_candidates = candidates[mid_start:mid_end]

    if not mid_candidates:
        mid_candidates = candidates  # 兜底

    return random.choice(mid_candidates)


def build_prompt(dialog: list, cut_idx: int) -> dict | None:
    """构造一个 prompt 字典"""
    context = dialog[:cut_idx]  # 不包含 cut_idx 这一轮
    ground_truth = dialog[cut_idx]["content"]

    # 质量过滤
    if len(context) < 4:
        return None  # 上下文太短
    if len(ground_truth) < 30:
        return None  # ground truth 过短，可能数据有问题

    return {
        "context": context,
        "ground_truth": ground_truth,
        "cut_idx": cut_idx,
    }


# ============================================================
# 主流程
# ============================================================
print("=" * 70)
print("Step 1: 加载 PsyDTCorpus")
print("=" * 70)
ds = MsDataset.load("YIRONGCHEN/PsyDTCorpus", subset_name="default", split="train")
print(f"  数据集大小: {len(ds)}")

print("\n" + "=" * 70)
print("Step 2: 按主题分组")
print("=" * 70)
by_tag = defaultdict(list)
for idx, item in enumerate(ds):
    by_tag[item["normalizedTag"]].append((idx, item))

for tag in TARGET_QUOTA:
    available = len(by_tag.get(tag, []))  # 该标签在数据集中实际有多少条样本
    target = TARGET_QUOTA[tag]  # 你希望至少采集多少条
    coverage = target / available * 100 if available > 0 else 0
    print(f"  {tag:>15}: 可用 {available:>5}, 目标 {target:>4}, 覆盖率 {coverage:>5.1f}%")

print("\n" + "=" * 70)
print("Step 3: 分层采样")
print("=" * 70)
all_prompts = []  # 存放最终构建的所有 prompt 样本
stats = Counter()  # 统计每个主题实际成功采集的数量
skipped = Counter()  # 统计跳过的原因

for tag, target in TARGET_QUOTA.items():
    if tag not in by_tag:
        print(f"  ⚠️  {tag}: 数据集中没找到这个主题，跳过")
        continue

    dialogs = by_tag[tag]
    random.shuffle(dialogs)  # 打乱顺序，避免总取前 N 个

    collected = 0
    for orig_idx, item in dialogs:
        if collected >= target:
            break

        cut_idx = pick_cut_index(item["messages"])  # 只选择中段的 assistant 发言，跳过开场白
        if cut_idx is None:
            skipped[f"{tag}_no_cut"] += 1
            continue

        prompt_data = build_prompt(item["messages"], cut_idx)  # 构建 cut_idx 之前的 prompt 数据
        if prompt_data is None:
            skipped[f"{tag}_filtered"] += 1
            continue

        all_prompts.append({
            "prompt_id": f"{tag}_{orig_idx}_cut{cut_idx}",
            "tag": tag,
            "dialog_id": orig_idx,
            **prompt_data,
        })
        collected += 1

    stats[tag] = collected
    status = "✅" if collected == target else "⚠️"
    print(f"  {status} {tag:>15}: 收集 {collected:>4} / 目标 {target:>4}")

print("\n" + "=" * 70)
print("Step 4: 保存到 jsonl")
print("=" * 70)

# 打乱整体顺序，避免主题聚集（采样时不同主题交替更稳）
random.shuffle(all_prompts)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for p in all_prompts:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"\n✅ 已保存 {len(all_prompts)} 个 prompt 到 {OUTPUT_PATH}")
print(f"   文件大小: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

# 验证写入
with open(OUTPUT_PATH) as f:
    line_count = sum(1 for _ in f)
print(f"   验证: 文件实际行数 {line_count}")

print("\n" + "=" * 70)
print("Step 5: 统计 + 抽样检查")
print("=" * 70)

# 上下文长度分布
ctx_lens = [len(p["context"]) for p in all_prompts]
print(f"\n上下文轮数: min={min(ctx_lens)}, max={max(ctx_lens)}, mean={sum(ctx_lens) / len(ctx_lens):.1f}")

# 截断点位置分布（在对话的多大比例处）
cut_ratios = []
for p in all_prompts:
    item = ds[p["dialog_id"]]
    ratio = p["cut_idx"] / len(item["messages"])
    cut_ratios.append(ratio)
print(
    f"截断点位置（在对话的多大比例处）: min={min(cut_ratios):.2f}, max={max(cut_ratios):.2f}, mean={sum(cut_ratios) / len(cut_ratios):.2f}")

# 跳过统计
if skipped:
    print(f"\n跳过的样本（如果有）:")
    for reason, count in skipped.most_common():
        print(f"  {reason}: {count}")

# 抽样展示 3 个不同主题的 prompt
print("\n" + "=" * 70)
print("Step 6: 抽样展示 3 个 prompt")
print("=" * 70)
sample_tags = ["婚恋", "职场", "心理学知识"]
for tag in sample_tags:
    matching = [p for p in all_prompts if p["tag"] == tag]
    if not matching:
        continue
    p = matching[0]
    print(f"\n[{tag}] prompt_id={p['prompt_id']}")
    print(f"  上下文 ({len(p['context'])} 轮):")
    print(f"    [system]: {p['context'][0]['content'][:80]}...")
    print(f"    [{p['context'][-2]['role']}]: {p['context'][-2]['content'][:100]}...")
    print(f"    [{p['context'][-1]['role']}]: {p['context'][-1]['content'][:100]}...")
    print(f"  Ground truth (待生成的咨询师回复):")
    print(f"    {p['ground_truth'][:150]}...")

print("\n✅ Task 1.2 完成")
