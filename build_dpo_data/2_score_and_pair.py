"""
任务 1.4：对 4 个候选回复进行打分，并构建 chosen/rejected（优选/拒绝）样本对。

相比 v2 版本的改进（基于第二轮错误分析）：
  1. 放宽了长度下限要求：
       - 回复长度 <15 个字符 → 扣 2 分（原为 <20 才扣分）
       - 长度在 15~30 字之间 → 加 0.5 分（原为不加分）
     原因：像中文问句“那您当时感受是什么？”（共 16 字）就是一个非常优秀的开放式提问，
           不应因稍短而被惩罚。

  2. “优质提问”（good_question）现在可豁免“共情开头”（empathy_first）的惩罚：
     原因：好的问题本身已体现关注和引导，强行加共情反而显得冗余或不自然。

  3. 按标签（tag）设置不同的最小分数差距（MIN_GAP）：
       - 标签为“心理学知识”的样本，MIN_GAP = 3.5（其他标签仍为 4.0）
     原因：“心理学知识”类回复通常较少包含共情或提问元素，若使用统一高门槛（4.0），会导致大量有效长尾样本被过滤掉。

处理流程（Pipeline）：
  raw_samples.jsonl
      → [使用 7 维评分规则对每个候选回复打分]
      → [配对：选取同组中得分最高（chosen）与最低（rejected）的回复，
          但仅当两者分数差距 ≥ 当前标签对应的 MIN_GAP 时才保留该对]
      → [按标签分层采样：95% 训练集 / 5% 验证集]
      → 输出 dpo_train.jsonl + dpo_val.jsonl（符合 ms-swift DPO 训练格式）
"""

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================
# 配置
# ============================================================
DATA_DIR = Path("/home/b311/data/zhangwei/zsq/Psy/data")
RAW = DATA_DIR / "raw_samples.jsonl"
OUT_TRAIN = DATA_DIR / "dpo_train.jsonl"
OUT_VAL = DATA_DIR / "dpo_val.jsonl"
OUT_DEBUG = DATA_DIR / "scoring_debug.jsonl"

DEFAULT_MIN_GAP = 3.5
PER_TAG_MIN_GAP = {  # v2.2: with default 3.5, 心理学知识 needs even lower
    "心理学知识": 3.0,
}
VAL_RATIO = 0.05
SEED = 42


# ============================================================
# 7-dim
# ============================================================
# 共情表达
EMPATHY_WORDS = [
    "我理解", "我能感受到", "听起来", "看起来", "我注意到",
    "这一定", "想必", "或许", "也许", "似乎",
    "很正常", "可以理解", "是合理的", "是正常的",
    "不容易", "很辛苦", "很难受", "很痛苦",
    "难过", "焦虑", "无助", "沮丧", "困惑", "挫败",
    "愤怒", "委屈", "失落", "压力", "疲惫",
    "愿意", "可以多说", "想多了解", "想听你",
    "你的感受", "你的想法", "对你来说",
    "没关系", "不用着急", "慢慢来", "陪着你", "和你一起",
]

# 有害语言
HARMFUL_PATTERNS = [
    r"你应该", r"你必须", r"你不应该", r"你不能",
    r"你要记住", r"你得",
    r"不要难过", r"别难过", r"不要哭", r"想开点",
    r"这没什么", r"别想那么多",
    r"建议你立刻", r"赶紧去", r"马上停止",
    r"我不是专业的", r"我没法帮你",
]

# 模板化程度
TEMPLATE_INDICATORS = [
    ["我理解", "我能理解"],
    ["这是正常的", "很正常", "是合理的"],
    ["尝试", "试着", "可以试", "建议"],
]


def score_response(text: str) -> dict:
    s = {"empathy": 0.0, "length": 0.0, "harmful": 0.0,
         "empathy_first": 0.0, "repetition": 0.0,
         "good_question": 0.0, "template_penalty": 0.0}

    # 如果回答为空，总分为 -10
    if not text or not text.strip():
        s["total"] = -10.0
        return s

    # Dim 1: 检查文本是否包含预设的共情词汇，每命中一个词加 0.5 分，最多 4.0 分
    hits = sum(1 for w in EMPATHY_WORDS if w in text)
    s["empathy"] = min(hits * 0.5, 4.0)

    # Dim 2: 长度合理性
    L = len(text)
    if 50 <= L <= 300:
        s["length"] = 2.0
    elif 30 <= L < 50:
        s["length"] = 1.0
    elif 15 <= L < 30:
        s["length"] = 0.5
    elif L < 15:
        s["length"] = -2.0
    else:
        s["length"] = -1.0

    # Dim 3: 有害语言，每条减 1.5
    harmful_hits = sum(1 for p in HARMFUL_PATTERNS if re.search(p, text))
    s["harmful"] = -1.5 * harmful_hits

    # Dim 6: 是否为优质开放式提问
    has_q = "?" in text or "？" in text
    addresses_client = "你" in text or "您" in text
    is_good_question = False
    if has_q and addresses_client:
        if L < 80:
            s["good_question"] = 1.5
            is_good_question = True
        elif L < 150:
            s["good_question"] = 0.5

    # Dim 4: 共情优先启发式规则，只检查开头部分，除去开放式提问
    first_third = text[: max(1, len(text) // 3)]
    if any(w in first_third for w in EMPATHY_WORDS):
        s["empathy_first"] = 1.5
    elif is_good_question:        # 如果是开放式提问就不需要
        s["empathy_first"] = 0.0
    else:
        s["empathy_first"] = -1.0

    # Dim 5: 4-gram 重复惩罚，统计重复出现3次的
    if len(text) >= 4:
        grams = [text[i:i + 4] for i in range(len(text) - 3)]
        gram_counts = Counter(grams)
        repeated = sum(1 for c in gram_counts.values() if c >= 3)
        s["repetition"] = -0.5 * repeated

    # Dim 7: 模板化共情惩罚
    groups_hit = 0
    for group in TEMPLATE_INDICATORS:
        if any(ind in text for ind in group):
            groups_hit += 1
    if groups_hit >= 2:
        s["template_penalty"] = -1.0

    s["total"] = sum(v for k, v in s.items() if k != "total")   # 计算总分
    return s


# ============================================================
# 7-dim
# ============================================================
def build_pair(record: dict) -> dict | None:
    samples = record["samples"]
    scored = [(s, score_response(s)) for s in samples]
    scored.sort(key=lambda x: x[1]["total"], reverse=True)

    best_text, best_s = scored[0]
    worst_text, worst_s = scored[-1]
    gap = best_s["total"] - worst_s["total"]

    # 使用 MIN_GAP
    min_gap = PER_TAG_MIN_GAP.get(record["tag"], DEFAULT_MIN_GAP)
    if gap < min_gap:
        return None # 若差距很小，直接丢弃

    return {
        "prompt_id": record["prompt_id"],
        "tag": record["tag"],
        "context": record["context"],
        "chosen": best_text,
        "rejected": worst_text,
        "chosen_score": round(best_s["total"], 2),
        "rejected_score": round(worst_s["total"], 2),
        "gap": round(gap, 2),
        "all_scores": [round(s[1]["total"], 2) for s in scored],
    }

# 将构建好的偏好样本对转换为 ms-swift 框架所需的 DPO 训练格式
def to_msswift_format(pair: dict) -> dict:
    """
    Note: pair['context'] already contains the system message from PsyDTCorpus
    (the 738-char REBT prompt). We do NOT prepend a duplicate SYSTEM_PROMPT.
    """
    messages = list(pair["context"])  # already has [system, user, assistant, user, ...]
    messages.append({"role": "assistant", "content": pair["chosen"]})
    return {
        "messages": messages,
        "rejected_response": pair["rejected"],
    }


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    random.seed(SEED)

    records = []
    with open(RAW, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f" Loaded {len(records)} prompts × 4 samples = {len(records) * 4} responses")
    print(f" v2.1 scorer | DEFAULT_MIN_GAP={DEFAULT_MIN_GAP} | per-tag overrides: {PER_TAG_MIN_GAP}")

    pairs = []
    skipped = 0
    debug_rows = []
    for r in records:
        scored = [(s, score_response(s)) for s in r["samples"]]
        for text, s in scored:
            debug_rows.append({
                "prompt_id": r["prompt_id"],
                "tag": r["tag"],
                "text": text,
                **s,
            })

        pair = build_pair(r)
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)

    print(f" Built {len(pairs)} pairs (skipped {skipped})")

    chosen_scores = [p["chosen_score"] for p in pairs]
    rejected_scores = [p["rejected_score"] for p in pairs]
    chosen_lens = [len(p["chosen"]) for p in pairs]
    rejected_lens = [len(p["rejected"]) for p in pairs]
    print(f"  chosen score   avg = {sum(chosen_scores)/len(chosen_scores):+.2f}")
    print(f"  rejected score avg = {sum(rejected_scores)/len(rejected_scores):+.2f}")
    print(f"  avg gap = {sum(p['gap'] for p in pairs)/len(pairs):+.2f}")
    print(f"  chosen avg length   = {sum(chosen_lens)/len(chosen_lens):.1f} chars")
    print(f"  rejected avg length = {sum(rejected_lens)/len(rejected_lens):.1f} chars")
    print(f"  length diff (chosen - rejected) = {sum(chosen_lens)/len(chosen_lens) - sum(rejected_lens)/len(rejected_lens):+.1f} chars")

    tag_dist = Counter(p["tag"] for p in pairs)
    print(f"\n Pairs per tag:")
    for tag, n in sorted(tag_dist.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {n}")

    # Stratified train/val split
    by_tag = defaultdict(list)
    for p in pairs:
        by_tag[p["tag"]].append(p)
    train, val = [], []
    for tag, ps in by_tag.items():
        random.shuffle(ps)
        n_val = max(1, int(len(ps) * VAL_RATIO))
        val.extend(ps[:n_val])
        train.extend(ps[n_val:])
    random.shuffle(train)
    random.shuffle(val)
    print(f"\n Split: {len(train)} train + {len(val)} val")

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for p in train:
            f.write(json.dumps(to_msswift_format(p), ensure_ascii=False) + "\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for p in val:
            f.write(json.dumps(to_msswift_format(p), ensure_ascii=False) + "\n")
    with open(OUT_DEBUG, "w", encoding="utf-8") as f:
        for row in debug_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\n Wrote:")
    print(f"  {OUT_TRAIN}  ({OUT_TRAIN.stat().st_size / 1024:.1f} KB)")
    print(f"  {OUT_VAL}    ({OUT_VAL.stat().st_size / 1024:.1f} KB)")
    print(f"  {OUT_DEBUG}  ({OUT_DEBUG.stat().st_size / 1024:.1f} KB)")

    # Show 3 sample pairs
    print("\n" + "=" * 60)
    print(" Sample pairs (sorted by gap, showing 3):")
    print("=" * 60)
    pairs.sort(key=lambda x: -x["gap"])
    for p in pairs[:3]:
        print(f"\n--- prompt_id={p['prompt_id']} | tag={p['tag']} | gap={p['gap']} ---")
        print(f"  CHOSEN   ({p['chosen_score']}): {p['chosen'][:150]}")
        print(f"  REJECTED ({p['rejected_score']}): {p['rejected'][:150]}")
        print(f"  all 4 scores: {p['all_scores']}")


if __name__ == "__main__":
    main()