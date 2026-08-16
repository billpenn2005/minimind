# MiniMind3 · 手写实现教程

> 🎯 **一句话**：像搭乐高一样，从**零**手写一个能"对话"的小型语言模型 **minimind3**。
> 不需要 GPU、不需要下载任何数据集、不需要任何深度学习基础——只需要 Python 和好奇心。

[tutorial/00-DESIGN.md](tutorial/00-DESIGN.md)（总设计）｜本页是入门手册：**我是谁、适合谁、怎么学、会遇到什么**。

---

## 1. 这个教程是什么？

minimind3 是一个与 [minimind](https://github.com/jingyaogong/minimind)（本仓库 `master` 分支上的**标准实现**，一个 64M 参数的开源小模型）**同架构**的小型因果语言模型。本教程带着你：

```
从空 git 分支开始
   → 逐课手写全部模型代码（配置/归一化/位置编码/注意力/前馈/残差块/主体/输出头）
   → 手写数据管道（分词器接入、问答数据集）
   → 手写两个训练循环（预训练 Pretrain + 指令微调 SFT）
   → 手写模型格式转换（导出成 HuggingFace transformers 标准目录）
   → 端到端测试（真训练、真对话、真验收）
```

14 课，一课一个 git 分支，每课配一个**程序化验收器**（`verify.py`，自动判分）。全部在 CPU 上、几十 MB 代码与合成数据内完成。

## 2. 适合谁？需要什么基础？

| 如果你…… | 结论 |
|---|---|
| 完全零基础，想搞懂 LLM（ChatGPT 之类）内部到底怎么工作 | ✅ 本教程就是为你写的：每课从"数学直觉"讲到"一行行代码" |
| 会用 Python（循环、类、函数）但没碰过 PyTorch | ✅ 会用到基础算子（矩阵乘、加法），第 01 课会给足铺垫 |
| 只会调用 transformers 的 API，想看看"黑盒"里面是什么 | ✅ 本教程 13 课手写实现，正是你要的答案 |
| 已经是深度学习老手 | ✅ 可以跳过理论，直接做"任务清单"并挑战验收 |

**硬性要求只有三个**：装了 Python ≥ 3.10 的电脑（CPU 即可）、git、以及遇到 bug 愿意读错误信息的耐心。😊

## 3. 环境准备（一次性，5 分钟）

```bash
# 1) 进入本教程的入口分支（孤儿分支，从零开始）
git checkout tutorial/01-skeleton

# 2) 创建虚拟环境并安装依赖（CPU 版 torch 就够）
python -m venv .venv
# Windows:
.venv\Scripts\python.exe -m pip install -r requirements.txt
# Linux/macOS:
.venv/bin/python -m pip install -r requirements.txt

# 3) 验证
.venv/Scripts/python.exe -c "import torch; print(torch.__version__)"
```

> 本仓库的验收环境实测：torch 2.11.0+cpu / transformers 4.57.6 / datasets 3.6.0。
> 依赖只有 torch / transformers / datasets 三个核心库（以及它们各自的依赖）。

## 4. 14 课课程表

| 课 | 分支 | 主题 | 手写产出 | 核心概念 |
|---|---|---|---|---|
| 01 | `tutorial/01-skeleton` | 骨架与验收框架 | 包结构、合成数据工具、verify | 语言模型范式、链式法则、合成数据 |
| 02 | `tutorial/02-config` | 配置类 | `config.py` | 超参数、派生尺寸、transformers 配置协议 |
| 03 | `tutorial/03-rmsnorm` | RMSNorm 归一化 | `rms_norm.py` | 归一化的意义、缩放不变性、fp32 精度 |
| 04 | `tutorial/04-rope` | RoPE 旋转位置编码 | `rope.py` | 位置编码的必要性、旋转矩阵、相对位置 |
| 05 | `tutorial/05-attention` | 多头注意力（GQA） | `attention.py` | QKV、缩放点积、因果掩码、KV 缓存、GQA |
| 06 | `tutorial/06-feedforward` | SwiGLU 前馈层 | `feed_forward.py` | 非线性、门控、参数量经济学 |
| 07 | `tutorial/07-block` | 残差块 | `block.py` | Pre-Norm 残差、梯度"高速公路" |
| 08 | `tutorial/08-model` | 模型主体 | `model_body.py` | Embedding、层堆叠、RoPE buffer、KV 编排 |
| 09 | `tutorial/09-causal-lm` | 因果 LM + 生成 | `causal_lm.py` | 权重绑定、shift 损失、贪心/采样生成、保存加载 |
| 10 | `tutorial/10-data` | 分词器与数据集 | `tokenizer_utils.py`、`datasets.py` | BPE、ChatML 模板、掩码标签（-100） |
| 11 | `tutorial/11-pretrain` | 预训练循环 | `train_pretrain.py` | 余弦退火、梯度裁剪/累积、断点续训 |
| 12 | `tutorial/12-sft` | 全参 SFT | `train_full_sft.py` | 监督微调、掩码损失、灾难性遗忘 |
| 13 | `tutorial/13-convert` | HF 格式转换 | `convert.py` | config.json、auto_map、remote code、round-trip |
| 14 | `tutorial/14-final` | 端到端测试 | `e2e.py`、全链验收 | 评估口径、流水线、一键全链验收 |

## 5. 怎么学？（学习节奏建议）

**每课的标准流程**（强烈建议照做）：

```bash
git checkout tutorial/NN-xxx        # 1. 切到本课分支（代码停在"上一课完成时"）
# 2. 读 tutorial/NN-xxx/README.md（先读"理论"再读"任务清单"）
# 3. 照着任务清单，在 minimind3/ 里手写你的实现
# 4. 自己写完了，再看仓库里已有的实现做对比（本仓库预置了"参考答案"）
.venv/Scripts/python.exe verify.py  # 5. 跑验收，直到所有检查 PASS
git diff tutorial/10-data tutorial/11-pretrain   # 6.（可选）看"本课增量"到底改了什么
```

> 💡 本仓库每课的 `minimind3/` 与 `verify/` 里**已经包含完整实现**——它们既是"参考答案"，也是"判卷标准"。
> 最有效的学法是：**先合上所有文件写一遍，再打开对照**。看不懂时就去看对应课的 README 理论部分。

## 6. 验收机制（判卷机怎么工作）

- **判卷机**：根目录 `verify.py` 自动扫描 `verify/` 目录下 `NN_xxx.py` 开头的文件，执行其中注册的断言函数；
- **判分**：PASS = 绿勾，FAIL = 红叉 + 错误详情；存在 FAIL 时退出码非 0；
- **快慢两档**：
  - `python verify.py` —— 全量（含真实训练类验收，约 1.5~3 分钟）；
  - `python verify.py --fast` —— 跳过训练类，数秒内完成，日常复查用；
- **逐课累积**：到第 14 课时全量验收共 70 项检查；14 个分支可以一键全查：

```bash
tools/check_all_branches.sh --fast    # 依次 checkout 14 个分支并各自验收（约 2 分钟）
```

## 7. 遇到问题？FAQ

**Q1: 报错 `ModuleNotFoundError` / 内存 / CUDA？**
全部验收用 CPU 即可；确认你是用 `.venv` 里的 python（`Python: Select Interpreter` 或显式路径）。

**Q2: 打印中文乱码？**
Windows 控制台默认 GBK，终端显示乱码 ≠ 代码错。验收器打印的是 ASCII 标记，不受影响。

**Q3: 卡在某一课 ≥ 2 小时？**
正常。先对照 `git diff` 看相邻分支差异，再对照 `master` 的标准实现（`tools/fetch_reference.sh` 导出到 `reference/`）。

**Q4: 我想看"真实大模型"长什么样？**
`master` 分支就是 minimind 标准实现：更复杂的训练循环（DDP、bf16、wandb）、MoE、LoRA、DPO/GRPO 等——教程每课的"对照标准实现"小节会指路。

**Q5: 学完能干什么？**
你可以把 `data/` 换成真实数据（`dataset/*.jsonl` 的 schema 完全一致），把 hidden_size 调大，用 GPU 训练——你的代码就变成了一个"真模型"的雏形。

## 8. 目录导航

```
verify.py                验收入口（判卷机）
minimind3/               你的手写实现（模型/数据/训练/转换/端到端）
verify/                  验收检查（每课一个 NN_xxx.py）
tools/                   工具（合成数据、参考实现导出、全链验收脚本）
tutorial/                文档（00-DESIGN 总设计 + 14 课 README）
data/                    合成数据（jsonl，可复现）
checkpoints/ out/  ...   训练产物（自动生成，已 gitignore）
```

祝玩得开心！🎉 从 `git checkout tutorial/01-skeleton` 开始吧。