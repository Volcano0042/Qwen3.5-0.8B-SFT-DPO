"""
Phase 3b: 用 DeepSeek V4-Flash 对 202 个 (SFT vs DPO) 回复对做 pairwise 评估。
每对评估 2 次（交换 A/B 顺序）缓解 position bias。

输入：data/phase3_generations.jsonl (202 行)
输出：data/phase3_judgments.jsonl (≤404 行，每行一次评估)

支持断点续跑：启动时读已写入的 (prompt_id, run_idx)，跳过已完成的。
"""

import json
import os
import time
import requests
from pathlib import Path

# ============ 配置 ============
PROJECT_DIR = Path.cwd()
INPUT_FILE = PROJECT_DIR / "data" / "phase3_generations.jsonl"
OUTPUT_FILE = PROJECT_DIR / "data" / "phase3_judgments.jsonl"


API_KEY = os.environ.get("DEEPSEEK_API_KEY")
assert API_KEY, "未找到 DEEPSEEK_API_KEY 环境变量，请 source ~/.bashrc"


API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"  # 路由到 V4-Flash
TIMEOUT = 60
MAX_RETRIES = 3
SLEEP_BETWEEN = 0.1  # API 间隔，避免限流


# ============ Judge Prompt 模板 ============
JUDGE_SYSTEM_PROMPT = """你是一位资深的心理咨询督导专家，专精于理性情绪行为疗法（REBT）。
你的任务是评估两位实习咨询师对同一位来访者的回复，判断哪位咨询师的回复更好。

评估标准（按重要性排序）：
1. **共情与情绪理解**：是否准确捕捉来访者的情绪状态，让来访者感到被理解
2. **提问质量**：使用开放式提问引导来访者自我探索，避免诱导性、封闭式或带预设答案的提问
3. **避免不当回应**：不替来访者下负面归因（如"也许他觉得你不够好"），不替来访者做决定，不过早给建议
4. **REBT 一致性**：是否在合适时机帮助来访者识别非理性信念，但不强行套用框架

输出要求：仅输出 JSON 格式，不要任何其他文字。
{"verdict": "A" | "B" | "Tie", "reason": "30字以内的简短理由"}"""


JUDGE_USER_TEMPLATE = """【对话历史】
{context}

【来访者最新发言】
{last_user}

---

【咨询师 A 的回复】
{reply_a}

【咨询师 B 的回复】
{reply_b}

---

请评估哪位咨询师的回复更好。仅输出 JSON，不要其他内容。"""


# ============ 工具函数 ============
def format_context_for_judge(context, max_turns=4):
    """格式化对话历史给 judge 看。跳过 system，留最后 max_turns 轮，最后一句 user 单独提取"""
    non_system = [m for m in context if m["role"] != "system"]
    last_user_content = None
    for m in reversed(non_system):
        if m["role"] == "user":
            last_user_content = m["content"]
            break
    # 历史里去掉最后一条 user（已单独提取）
    history = non_system[:-1] if non_system and non_system[-1]["role"] == "user" else non_system
    # 截断到最后 max_turns*2 条
    if len(history) > max_turns * 2:
        history = history[-(max_turns * 2):]
        prefix = "（前面对话已省略）\n"
    else:
        prefix = ""
    lines = [prefix] if prefix else []
    for m in history:
        role_zh = "来访者" if m["role"] == "user" else "咨询师"
        lines.append(f"【{role_zh}】{m['content']}")
    return "\n".join(lines), last_user_content


def call_deepseek(messages, max_retries=MAX_RETRIES):
    """调用 DeepSeek API，带重试。返回 content string，失败返回 None。"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 200,
    }
    for attempt in range(max_retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=TIMEOUT)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                print(f"    [API] status {resp.status_code} attempt {attempt+1}/{max_retries}: {resp.text[:200]}")
        except Exception as e:
            print(f"    [API] exception attempt {attempt+1}/{max_retries}: {e}")
        time.sleep(2 ** attempt)  # 指数退避：1s, 2s, 4s
    return None


def parse_verdict(content):
    """从 judge 响应里解析 JSON。容错：去掉可能的 ```json``` 包裹。"""
    if not content:
        return None
    text = content.strip()
    # 去掉可能的 markdown code fence
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        obj = json.loads(text)
        verdict = obj.get("verdict", "").strip()
        if verdict in ("A", "B", "Tie"):
            return {"verdict": verdict, "reason": obj.get("reason", "")[:100]}
    except json.JSONDecodeError:
        pass
    return None


def judge_one_pair(context, last_user, reply_a, reply_b):
    """对一对回复做单次评估"""
    user_msg = JUDGE_USER_TEMPLATE.format(
        context=context,
        last_user=last_user,
        reply_a=reply_a,
        reply_b=reply_b,
    )
    content = call_deepseek([
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    return parse_verdict(content), content


def load_done_keys():
    """读已完成的 (prompt_id, run_idx)，断点续跑用"""
    done = set()
    if not OUTPUT_FILE.exists():
        return done
    with open(OUTPUT_FILE, encoding='utf-8') as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["prompt_id"], r["run_idx"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return done


# ============ 主流程 ============
def main():
    # 1. 读生成结果
    records = []
    with open(INPUT_FILE, encoding='utf-8') as f:
        for line in f:
            records.append(json.loads(line))
    print(f"[Data] loaded {len(records)} generation pairs")

    # 2. 构建评估任务列表（每对 2 次）
    tasks = []
    for r in records:
        ctx, last_user = format_context_for_judge(r["context"])
        # run_idx=0: A=SFT, B=DPO
        tasks.append({
            "prompt_id": r["prompt_id"], "tag": r["tag"], "run_idx": 0,
            "a_model": "sft", "b_model": "dpo",
            "context": ctx, "last_user": last_user,
            "reply_a": r["sft_reply"], "reply_b": r["dpo_reply"],
        })
        # run_idx=1: A=DPO, B=SFT
        tasks.append({
            "prompt_id": r["prompt_id"], "tag": r["tag"], "run_idx": 1,
            "a_model": "dpo", "b_model": "sft",
            "context": ctx, "last_user": last_user,
            "reply_a": r["dpo_reply"], "reply_b": r["sft_reply"],
        })
    print(f"[Tasks] total {len(tasks)} judge calls (={len(records)} pairs × 2 orders)")

    # 3. 断点续跑
    done = load_done_keys()
    if done:
        tasks = [t for t in tasks if (t["prompt_id"], t["run_idx"]) not in done]
        print(f"[Resume] {len(done)} already done, {len(tasks)} remaining")

    if not tasks:
        print("✅ 所有评估已完成")
        return

    # 4. 跑评估
    t_start = time.time()
    failed = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
        for i, t in enumerate(tasks, 1):
            verdict_obj, raw_content = judge_one_pair(
                t["context"], t["last_user"], t["reply_a"], t["reply_b"]
            )
            record = {
                "prompt_id": t["prompt_id"],
                "tag": t["tag"],
                "run_idx": t["run_idx"],
                "a_model": t["a_model"],
                "b_model": t["b_model"],
                "verdict": verdict_obj["verdict"] if verdict_obj else None,
                "reason": verdict_obj["reason"] if verdict_obj else None,
                "raw_content": raw_content if not verdict_obj else None,  # 失败时保留原始返回
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

            if not verdict_obj:
                failed += 1

            if i % 20 == 0 or i == len(tasks):
                elapsed = time.time() - t_start
                eta = elapsed / i * (len(tasks) - i)
                print(f"  [{i}/{len(tasks)}] elapsed={elapsed:.0f}s, eta={eta:.0f}s, failed={failed}")

            time.sleep(SLEEP_BETWEEN)

    total = time.time() - t_start
    print(f"\n✅ Done! Total time: {total:.0f}s ({total/60:.1f} min)")
    print(f"   Total tasks: {len(tasks)}, failed: {failed}")
    print(f"   Output: {OUTPUT_FILE}")
    if failed > 0:
        print(f"⚠️  {failed} 个任务失败，可重跑此脚本断点续跑（会重试这些）")
        print(f"   注意：失败的记录已写入文件但 verdict=null，重跑会跳过；")
        print(f"   如要强制重试，需手动从 jsonl 删除 verdict=null 的行")


if __name__ == "__main__":
    main()