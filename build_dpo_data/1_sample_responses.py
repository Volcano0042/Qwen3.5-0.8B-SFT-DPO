"""
sample_responses.py — Task 1.3: 批量采样

输入: data/prompts.jsonl  (1232 个 prompt)
输出: data/raw_samples.jsonl  (每行: prompt + 4 个候选回复)

特性:
  - 断点续传（自动跳过已完成的 prompt）
  - 每 25 个 prompt 增量保存
  - 错误隔离（单个 prompt 失败不影响其他）
  - 进度条显示 ETA
"""

import json
import time
import sys
from pathlib import Path
from tqdm import tqdm
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer, snapshot_download
from peft import PeftModel

# ============================================================
# 配置
# ============================================================
BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "/home/b311/data/zhangwei/zsq/Psy/my_output/psy-qwen-0.8b/v0-20260509-234522/checkpoint-375"
DATA_DIR = Path("/home/b311/data/zhangwei/zsq/Psy/data")
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"
OUTPUT_PATH = DATA_DIR / "raw_samples.jsonl"
FAILED_PATH = DATA_DIR / "failed_prompts_test.jsonl"
CHECKPOINT_EVERY = 25  # 每 25 个 prompt 保存一次

SAMPLES_PER_PROMPT = 4
TEMPERATURE = 0.9
TOP_P = 0.95
MAX_NEW_TOKENS = 300

# ============================================================
# Step 1: 加载已完成的 prompt（断点续传）
# ============================================================
done_ids = set()
if OUTPUT_PATH.exists():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                done_ids.add(json.loads(line)["prompt_id"])
            except Exception:
                pass
    print(f"✅ 检测到已完成 {len(done_ids)} 个 prompt（断点续传）")

# ============================================================
# Step 2: 加载所有 prompts，过滤掉已完成的
# ============================================================
all_prompts = []
with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        all_prompts.append(json.loads(line))

# 只保留那些 prompt_id 不在 raw_samples.jsonl 中的 prompt
todo_prompts = [p for p in all_prompts if p["prompt_id"] not in done_ids]
print(f"总 prompt: {len(all_prompts)}, 已完成: {len(done_ids)}, 待处理: {len(todo_prompts)}")

if not todo_prompts:
    print("✅ 全部 prompt 已采样完成，无需重跑")
    sys.exit(0)

# ============================================================
# Step 3: 加载模型
# ============================================================
print("\n加载模型中...")
base_dir = snapshot_download(BASE_MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(base_dir, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_dir,
    dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
print(f"✅ 模型加载完成. 显存占用: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# ============================================================
# Step 4: 采样函数
# ============================================================
def generate_n_samples(context: list, n: int = 4) -> list:
    """对一个 context 生成 n 个候选回复"""
    input_text = tokenizer.apply_chat_template(
        context, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,      # 最多生成 300 个新 token
            temperature=TEMPERATURE,            # 0.9 → 较高随机性
            top_p=TOP_P,                        # 0.95 → nucleus sampling
            do_sample=True,                     # 启用采样（非 greedy）
            num_return_sequences=n,             # 一次生成 n 条不同回复
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    responses = []
    input_len = inputs.input_ids.shape[1]
    for i in range(n):
        text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
        responses.append(text.strip())
    return responses


# ============================================================
# Step 5: 主循环（带 checkpoint）
# ============================================================
print(f"\n开始采样 ({SAMPLES_PER_PROMPT} samples/prompt, temp={TEMPERATURE})...")
batch_buffer = []   # 暂存成功生成的记录
failed_buffer = []  # 暂存失败的记录
start_time = time.time()

with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f, \
     open(FAILED_PATH, "a", encoding="utf-8") as fail_f:
    
    pbar = tqdm(todo_prompts, desc="Sampling", unit="prompt")
    for i, p in enumerate(pbar):
        try:
            samples = generate_n_samples(p["context"], n=SAMPLES_PER_PROMPT)
            
            record = {
                "prompt_id": p["prompt_id"],
                "tag": p["tag"],
                "dialog_id": p["dialog_id"],
                "cut_idx": p["cut_idx"],
                "context": p["context"],
                "ground_truth": p["ground_truth"],
                "samples": samples,     # 是生成的 n 条回复
            }
            batch_buffer.append(record)
            
            # 进度条显示采样到的回复长度（quick sanity check）
            avg_len = sum(len(s) for s in samples) / len(samples)
            pbar.set_postfix({"avg_len": f"{avg_len:.0f}", "ok": len(batch_buffer), "fail": len(failed_buffer)})
            
        except Exception as e:
            failed_buffer.append({
                "prompt_id": p["prompt_id"],
                "error": str(e)[:200],
            })
            pbar.set_postfix({"ok": len(batch_buffer), "fail": len(failed_buffer)})
            continue
        
        # 每 CHECKPOINT_EVERY 个 flush 一次
        if (i + 1) % CHECKPOINT_EVERY == 0:
            for r in batch_buffer:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for fr in failed_buffer:
                fail_f.write(json.dumps(fr, ensure_ascii=False) + "\n")
            out_f.flush()
            fail_f.flush()
            batch_buffer.clear()
            failed_buffer.clear()
    
    # 收尾：写出剩余 buffer
    for r in batch_buffer:
        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for fr in failed_buffer:
        fail_f.write(json.dumps(fr, ensure_ascii=False) + "\n")

# ============================================================
# Step 6: 总结
# ============================================================
elapsed = time.time() - start_time
print(f"\n{'='*60}")
print(f"✅ 采样完成")
print(f"  耗时: {elapsed/60:.1f} 分钟 ({elapsed/3600:.2f} 小时)")
print(f"  平均每 prompt: {elapsed/len(todo_prompts):.1f} 秒")

# 验证最终文件
total_records = 0
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    for _ in f:
        total_records += 1

print(f"\n  raw_samples.jsonl 总记录数: {total_records}")
print(f"  输出文件大小: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.1f} MB")

if FAILED_PATH.exists():
    fail_count = sum(1 for _ in open(FAILED_PATH))
    if fail_count:
        print(f"  ⚠️  失败 prompt 数: {fail_count}（详见 failed_prompts.jsonl）")

print(f"{'='*60}")
