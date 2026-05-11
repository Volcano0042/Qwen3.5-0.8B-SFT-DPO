#!/bin/bash
# Psy-Qwen-DPO 环境恢复脚本
# 用途：DSW 实例重启后，/usr/local 被还原时一键恢复
# 用法：bash /mnt/workspace/psy-qwen-dpo/scripts/setup_env.sh

set -e  # 任一步骤失败就停

echo "========================================="
echo "Step 1/3: 升级 peft 0.17.1 -> 0.19.1"
echo "（解决 HybridCache 与 transformers 5.x 兼容问题）"
echo "========================================="
pip install --no-deps peft==0.19.1

echo ""
echo "========================================="
echo "Step 2/3: 安装 mergekit"
echo "（trl 0.24.0 callbacks.py eager import 需要）"
echo "========================================="
pip install mergekit

echo ""
echo "========================================="
echo "Step 3/3: 修复 trl 0.24.0 truthy-tuple bug"
echo "（_is_package_available 返回 tuple，需要取 [0]）"
echo "========================================="
TRL_FILE="/home/b311/data/zhangwei/miniconda3/envs/psy/lib/python3.10/site-packages/trl/import_utils.py"

# 备份原文件（如果还没备份过）
if [ ! -f "${TRL_FILE}.bak" ]; then
    cp "$TRL_FILE" "${TRL_FILE}.bak"
    echo "已备份原文件到 ${TRL_FILE}.bak"
fi

# 用 sed 给所有 _X_available = _is_package_available("X") 行末尾加 [0]
# 匹配模式：_xxx_available = _is_package_available("xxx")
# 替换为：_xxx_available = _is_package_available("xxx")[0]
sed -i -E 's/^(_[a-z_]+_available = _is_package_available\("[a-z_]+"\))$/\1[0]/' "$TRL_FILE"

# 验证：grep 应该看到 12 个修复
PATCHED_COUNT=$(grep -c "_is_package_available.*\[0\]$" "$TRL_FILE")
echo "已修复 ${PATCHED_COUNT} 处（期望 12）"

if [ "$PATCHED_COUNT" -ne 12 ]; then
    echo "⚠️ 警告：修复数量不对，请人工检查 $TRL_FILE"
    exit 1
fi

echo ""
echo "========================================="
echo "✅ 环境恢复完成！"
echo "========================================="
echo "验证命令：python -c \"from trl import DPOTrainer; print('trl OK')\""