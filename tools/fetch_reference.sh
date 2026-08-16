#!/usr/bin/env bash
# 从 git master 分支导出 minimind 标准实现文件到 reference/（仅导出文本文件，占零磁盘成本）。
# 用于手写实现后的逐行对照/数值对照。
set -euo pipefail
mkdir -p reference
git show master:model/model_minimind.py > reference/model_minimind.py
git show master:dataset/lm_dataset.py > reference/lm_dataset.py
git show master:trainer/train_pretrain.py > reference/train_pretrain.py 2>/dev/null || true
git show master:trainer/train_full_sft.py > reference/train_full_sft.py 2>/dev/null || true
git show master:scripts/convert_model.py > reference/convert_model.py 2>/dev/null || true
echo "reference files exported to reference/"
ls -la reference/