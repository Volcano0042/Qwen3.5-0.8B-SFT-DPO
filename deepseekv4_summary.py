"""
Phase 3c: 汇总 phase3_judgments.jsonl，计算 win rate + 多个分解视角。

输出：
- 终端打印总 win rate + tag 分解 + position bias 诊断
- data/phase3_results.json (供 README 引用)
"""

import json
from pathlib import Path
from collections import defaultdict, Counter

PROJECT_DIR = Path.cwd()
INPUT_FILE = PROJECT_DIR / "data" / "phase3_judgments.jsonl"
OUTPUT_FILE = PROJECT_DIR / "data" / "phase3_results.json"


def load_judgments():
    judgments = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            judgments.append(json.loads(line))
    return judgments


def winner_model(j):
    """从一条 judgment 推出谁赢了：'sft' / 'dpo' / 'tie' / None(失败)"""
    if not j["verdict"]:
        return None
    if j["verdict"] == "Tie":
        return "tie"
    # verdict 是 'A' 或 'B'，对应 a_model 或 b_model
    return j["a_model"] if j["verdict"] == "A" else j["b_model"]


def score_pair(j0, j1):
    """
    根据一对 (run=0, run=1) 评估算 DPO 的得分。
    j0: A=SFT, B=DPO
    j1: A=DPO, B=SFT
    返回 (dpo_score, category) ∈ {1.0, 0.75, 0.5, 0.25, 0.0}
    """
    w0 = winner_model(j0)  # 'sft' / 'dpo' / 'tie'
    w1 = winner_model(j1)

    if w0 is None or w1 is None:
        return None, "failed"

    # 两次都 DPO 赢 → 真胜
    if w0 == "dpo" and w1 == "dpo":
        return 1.0, "consistent_dpo_win"
    # 两次都 SFT 赢 → 真负
    if w0 == "sft" and w1 == "sft":
        return 0.0, "consistent_sft_win"
    # 两次都平 → 平
    if w0 == "tie" and w1 == "tie":
        return 0.5, "consistent_tie"

    # 一胜一负（不一致）→ position bias，记 0.5
    if {w0, w1} == {"sft", "dpo"}:
        return 0.5, "inconsistent_position_bias"

    # 一胜一平
    if w0 == "dpo" and w1 == "tie": return 0.75, "dpo_win_tie"
    if w0 == "tie" and w1 == "dpo": return 0.75, "dpo_win_tie"
    if w0 == "sft" and w1 == "tie": return 0.25, "sft_win_tie"
    if w0 == "tie" and w1 == "sft": return 0.25, "sft_win_tie"

    return 0.5, "other"


def main():
    judgments = load_judgments()
    print(f"[Data] loaded {len(judgments)} judgments\n")

    # 按 prompt_id 分组（每组 2 条）
    by_pid = defaultdict(dict)
    for j in judgments:
        by_pid[j["prompt_id"]][j["run_idx"]] = j

    # 算每个 prompt 的得分
    pair_results = []  # list of (prompt_id, tag, score, category)
    for pid, pair in by_pid.items():
        if 0 not in pair or 1 not in pair:
            print(f"⚠️  prompt {pid} 不完整，跳过")
            continue
        score, category = score_pair(pair[0], pair[1])
        pair_results.append({
            "prompt_id": pid,
            "tag": pair[0]["tag"],
            "score": score,
            "category": category,
        })

    valid = [r for r in pair_results if r["score"] is not None]
    print(f"[Pairs] total {len(pair_results)}, valid {len(valid)}\n")

    # ---- 1. 总 Win Rate ----
    total_score = sum(r["score"] for r in valid)
    win_rate = total_score / len(valid)
    print("=" * 60)
    print(f"📊 总 Win Rate: {win_rate*100:.2f}% ({total_score:.1f} / {len(valid)})")
    print("=" * 60)

    # ---- 2. 分类计数 ----
    cat_counter = Counter(r["category"] for r in valid)
    print("\n【一致性分解】")
    cat_order = [
        ("consistent_dpo_win", "DPO 完胜（两次都 DPO 赢）"),
        ("dpo_win_tie",        "DPO 一胜一平"),
        ("consistent_tie",     "两次都平"),
        ("inconsistent_position_bias", "不一致（position bias）"),
        ("sft_win_tie",        "SFT 一胜一平"),
        ("consistent_sft_win", "SFT 完胜（两次都 SFT 赢）"),
    ]
    for cat, desc in cat_order:
        n = cat_counter.get(cat, 0)
        pct = n / len(valid) * 100 if valid else 0
        print(f"  {desc:30s}: {n:3d} ({pct:5.1f}%)")

    # ---- 3. Position Bias 诊断 ----
    bias_n = cat_counter.get("inconsistent_position_bias", 0)
    bias_pct = bias_n / len(valid) * 100
    print(f"\n【Position bias 诊断】")
    print(f"  不一致率: {bias_pct:.1f}% (越低说明 judge 越稳)")
    if bias_pct < 15:
        print(f"  → judge 表现稳定 ✅")
    elif bias_pct < 30:
        print(f"  → judge 中等稳定（合理范围）")
    else:
        print(f"  → judge 不太稳，但 2-way 缓解已生效")

    # ---- 4. 按 tag 分解 ----
    print(f"\n【按 tag 分解 Win Rate】")
    by_tag = defaultdict(list)
    for r in valid:
        by_tag[r["tag"]].append(r["score"])
    for tag in sorted(by_tag.keys(), key=lambda t: -len(by_tag[t])):
        scores = by_tag[tag]
        wr = sum(scores) / len(scores) * 100
        print(f"  {tag:8s} (n={len(scores):2d}): {wr:5.1f}%")

    # ---- 5. 单边 win rate（不计 tie / bias）----
    pure_dpo_wins = cat_counter.get("consistent_dpo_win", 0)
    pure_sft_wins = cat_counter.get("consistent_sft_win", 0)
    pure_total = pure_dpo_wins + pure_sft_wins
    if pure_total > 0:
        pure_wr = pure_dpo_wins / pure_total * 100
        print(f"\n【纯净 Win Rate（去掉 tie 和 bias）】")
        print(f"  DPO {pure_dpo_wins} : {pure_sft_wins} SFT")
        print(f"  纯净 win rate: {pure_wr:.1f}%")

    # ---- 6. 写 JSON 结果文件 ----
    summary = {
        "win_rate": win_rate,
        "win_rate_pct": round(win_rate * 100, 2),
        "total_pairs": len(valid),
        "category_breakdown": dict(cat_counter),
        "by_tag": {
            tag: {
                "n": len(scores),
                "win_rate_pct": round(sum(scores) / len(scores) * 100, 2),
            }
            for tag, scores in by_tag.items()
        },
        "position_bias_pct": round(bias_pct, 2),
        "pure_win_rate_pct": round(pure_dpo_wins / pure_total * 100, 2) if pure_total > 0 else None,
        "pure_dpo_wins": pure_dpo_wins,
        "pure_sft_wins": pure_sft_wins,
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Results saved to {OUTPUT_FILE}")

    # ---- 7. 简历素材建议 ----
    print("\n" + "=" * 60)
    print("📝 简历素材（直接可用）")
    print("=" * 60)
    if win_rate >= 0.60:
        rating = "漂亮"
    elif win_rate >= 0.55:
        rating = "合格"
    else:
        rating = "边缘"
    print(f"评估等级: {rating}")
    print(f"\n建议简历 bullet:")
    print(f'  "Achieved {win_rate*100:.1f}% win rate vs SFT baseline using DeepSeek V4-Flash')
    print(f'   as LLM judge with 2-way position-bias mitigation, validated on {len(valid)}')
    print(f'   held-out prompts across {len(by_tag)} psychological topics."')


if __name__ == "__main__":
    main()