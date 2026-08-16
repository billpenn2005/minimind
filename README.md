# MiniMind3 · 手写实现教程

从空分支开始，用 git 分支链逐课手写构建 **minimind3** —— 一个与
[minimind](https://github.com/jingyaogong/minimind)（本仓库 master，标准实现）同架构的
小型因果语言模型，覆盖：**模型本身 → 预训练 → 全参 SFT → HF transformers 格式转换 → 测试**。

- 全程 **无需 GPU**（真实训练可选）、**不下载任何数据集/权重**（合成数据，代码内生成）；
- 一课一个分支，每课配有 **程序化验收**（`python verify.py`，逐课累积）；
- 完整设计见 [`tutorial/00-DESIGN.md`](tutorial/00-DESIGN.md)（master 分支上）。

## 环境要求

- Python ≥ 3.10，git
- `torch>=2.0`（CPU 版即可）、`transformers>=4.40`、`datasets>=2.19`
  （本项目验收环境：torch 2.11.0+cpu / transformers 4.57.6 / datasets 3.6.0）

## 一课一分支

| 课 | 分支 | 主题 | 产出 |
|---|---|---|---|
| 01 | `tutorial/01-skeleton` | 项目骨架与验收框架 | `minimind3/` 包、合成数据工具、`verify.py` |
| 02 | `tutorial/02-config` | 配置类 | `minimind3/config.py` |
| 03 | `tutorial/03-rmsnorm` | RMSNorm | `minimind3/rms_norm.py` |
| 04 | `tutorial/04-rope` | RoPE 位置编码 | `minimind3/rope.py` |
| 05 | `tutorial/05-attention` | 多头注意力（GQA） | `minimind3/attention.py` |
| 06 | `tutorial/06-feedforward` | SwiGLU 前馈 | `minimind3/feed_forward.py` |
| 07 | `tutorial/07-block` | 残差块 | `minimind3/block.py` |
| 08 | `tutorial/08-model` | 主体模型 | `minimind3/model_body.py` |
| 09 | `tutorial/09-causal-lm` | 因果 LM + 生成 | `minimind3/causal_lm.py` |
| 10 | `tutorial/10-data` | 分词器与数据集 | `minimind3/tokenizer_utils.py`、`datasets.py` |
| 11 | `tutorial/11-pretrain` | 预训练循环 | `minimind3/train_pretrain.py` |
| 12 | `tutorial/12-sft` | 全参 SFT | `minimind3/train_full_sft.py` |
| 13 | `tutorial/13-convert` | HF 格式转换 | `minimind3/convert.py` |
| 14 | `tutorial/14-final` | 端到端测试 | `minimind3/e2e.py`、全链验收 |

## 快速开始（第 01 课）

```bash
git checkout tutorial/01-skeleton
.venv/Scripts/python.exe tools/make_synthetic_data.py --out data --num 200
.venv/Scripts/python.exe verify.py
```

通过后进入下一课：`git checkout tutorial/02-config`。

可随时用 `git diff` 查看增量：`git diff tutorial/<kind> tutorial/<kind+1>`；
对照标准实现：`git diff master` 或 `tools/fetch_reference.sh`。

## 验收机制

- `python verify.py`：当前分支全部检查项；
- `python verify.py --fast`：跳过耗时训练类检查；
- 检查文件按课存放在 `verify/` 目录（`verify/NN_xxx.py`），逐课累积；
- 退出码非 0 即存在失败项。