#!/bin/bash
set -euo pipefail   # 任何命令失败立即退出，避免静默错误

# ----------- 路径配置 -----------
BASE_MODEL="/home/b311/data/zhangwei/.cache/modelscope/hub/models/Qwen/Qwen3___5-0___8B"
SFT_LORA="/home/b311/data/zhangwei/zsq/Psy/my_output/psy-qwen-0.8b/v0-20260509-234522/checkpoint-375"
TRAIN_DATA="/home/b311/data/zhangwei/zsq/Psy/data/dpo_train.jsonl"
VAL_DATA="/home/b311/data/zhangwei/zsq/Psy/data/dpo_val.jsonl"
OUTPUT_DIR="/home/b311/data/zhangwei/zsq/Psy/dpo_output/psy-qwen-dpo-beta01"

# SFT 时的 LoRA target_modules 正则（必须与 SFT 完全一致）
TARGET_REGEX='^(model(?=\.).*\.(in_proj_qkv|gate_proj|up_proj|in_proj_b|q_proj|out_proj|v_proj|in_proj_z|in_proj_a|down_proj|o_proj|k_proj))$'

# ----------- 训练超参 -----------
mkdir -p "${OUTPUT_DIR}"

SWANLAB_PROJECT="Psy-Qwen-DPO-LoRA/Dpo" \
SWANLAB_WORKSPACE="165340" \
swift rlhf \
    --rlhf_type dpo \
    --model "${BASE_MODEL}" \
    --adapters "${SFT_LORA}" \
    --ref_adapters "${SFT_LORA}" \
    \
    --dataset "${TRAIN_DATA}" \
    --val_dataset "${VAL_DATA}" \
    --max_length 1024 \
    --truncation_strategy left \
    \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --target_regex "${TARGET_REGEX}" \
    \
    --beta 0.1 \
    --loss_type sigmoid \
    \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    \
    --bf16 true \
    --gradient_checkpointing true \
    --torch_dtype bfloat16 \
    \
    --logging_steps 5 \
    --save_strategy epoch \
    --save_total_limit 3 \
    --eval_strategy epoch \
    --report_to none \
    --seed 42 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${OUTPUT_DIR}/logs" \
    --report_to swanlab