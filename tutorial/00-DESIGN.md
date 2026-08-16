# MiniMind3 手写实现教程 · 总设计文档（00-DESIGN）

> **定位**：基于本仓库（[jingyaogong/minimind](https://github.com/jingyaogong/minimind)，即"标准实现"，
> 一个 64M 参数的开源小型语言模型），用 **git 分支链** 的方式，从空分支开始**逐课手写**构建一个
> **minimind3**（与 MiniMind 同架构的从零实现，覆盖"模型本身 → 预训练 → 全参 SFT →
> HF transformers 转换 → 端到端测试"全链路）。
> 每一课都有**程序化验收机制**（`verify.py` 自动判分）；全程**不依赖 GPU**（真实训练可选），
> **不下载任何数据集与模型权重**（数据用代码内合成）。
> 本文件是总设计 + 学习手册：回答"教程是什么、为什么这么设计、怎么学"。

---

## 0. 适合的读者与前置知识

| 前置知识 | 要求 | 说明 |
|---|---|---|
| Python | 会用：变量、函数、类、循环、`import` | 不确定的话先做本课任务 01 练手 |
| PyTorch | **零基础即可** | 每课上会解释用到的每个算子（`nn.Linear`、`add`、`softmax`……） |
| 机器学习 | **零基础即可** | 第 01 课就从"什么是语言模型"讲起 |
| git | 基本命令：`checkout` / `diff` / `log` | 本教程的"分支链"机制只需要这几条 | 
| 数学 | 会看"连乘、求和"符号即可 | 每处公式都有中文逐字解读，推导全部展开 |

**学习目标**（学完 14 课你将拥有）：
1. 能脱离任何库，**从零写出**一个可训练、可生成、可导出的因果语言模型（约 500 行核心代码）；
2. 理解 LLM 全链路：数据 → 模型 → 训练 → 微调 → 格式转换 → 服务化；
3. 读懂 minimind 标准实现乃至 Qwen3/Llama 生态的大部分代码；
4. 一份可扩展的代码库：换掉数据与配置即可"升级"成更大模型。

---

## 1. 教程目标与范围

**范围（本期 14 课）**：模型本身 + 预训练 + 全参 SFT + HF transformers 格式转换 + 最终测试。

**显式排除（作为进阶方向，写入各课"进阶"小节，不占本期分支）**：
LoRA、DPO/GRPO/PPO、数据蒸馏、MoE、量化、真实数据集、多卡/分布式、flash-attention 等算子优化。

**质量标准**：教程中的每一行代码都可与 minimind 标准实现（master 分支）逐模块比对；
所有验收断言可**程序化复现**（今天跑、明天跑、换台机器跑，结果一致）。

---

## 2. 工程约束（验收环境必须满足）

| 约束 | 说明 | 为什么 |
|---|---|---|
| **无 GPU** | 验收环境为 CPU（实测：torch 2.11.0+cpu / transformers 4.57.6 / datasets 3.6.0） | 让任何人零门槛复现 |
| **小存储** | 不下载任何数据集与权重；训练数据由 `tools/make_synthetic_data.py` 本地生成（约 56KB） | 教程增量 < 1MB |
| **可复现** | 合成数据 + 固定随机种子；验收断言用形状/数值/行为检查 | 告别"玄学通过率" |
| **快速** | 默认训练配置为微型（hidden=96~128、layers=2~4、词表 6400/512），全量验收约 1.5~3 分钟 | 每课可反复验证 |
| **手写** | 每课文档给出"任务清单"，要求**先手写**；分支内代码=参考答案；verify 只做行为/形状/数值断言，**不绑定实现细节** | 你的写法不同也能过验，学的是原理不是抄代码 |

> 这三条硬约束组合起来是本教程最大的特色：**一台普通电脑、一杯茶的时间、一次 E2E 的真实对话**。

---

## 3. 分支图谱（链式累积，一课一分支）

```
master  = minimind 标准实现（对照参考，永不合并进教程）
   │
   ● tutorial/01-skeleton    孤儿分支：项目骨架、合成数据工具、verify 框架         │ 5 项检查
   ● tutorial/02-config      MiniMindConfig（PretrainedConfig 子类 + 派生尺寸）   │ +5（累计10）
   ● tutorial/03-rmsnorm     RMSNorm（fp32 内部归一化）                          │ +5（累计15）
   ● tutorial/04-rope        RoPE：precompute_freqs_cis / apply_rotary_pos_emb  │ +6（累计21）
   ● tutorial/05-attention   GQA + QK-Norm + RoPE + KV-Cache + 因果掩码 + SDPA  │ +6（累计27）
   ● tutorial/06-feedforward SwiGLU（gate/up/down 三段式，无 bias）              │ +5（累计32）
   ● tutorial/07-block       MiniMindBlock：Pre-Norm 残差                      │ +5（累计37）
   ● tutorial/08-model       MiniMindModel：Embedding+堆叠+最终Norm+RoPE buffer │ +5（累计42）
   ● tutorial/09-causal-lm   MiniMindForCausalLM：lm_head/权重绑定/loss/generate│ +8（累计50）
   ● tutorial/10-data        Tokenizer 接入 + Pretrain/SFT 数据集（回答掩码）    │ +5（累计55）
   ● tutorial/11-pretrain    预训练循环：cosine LR、梯度裁剪、累积、续训         │ +4（累计59）
   ● tutorial/12-sft         全参 SFT：仅 assistant 片段计算损失                │ +3（累计62）
   ● tutorial/13-convert     pth ⇄ HF 互转 + AutoClass/remote-code 注册        │ +5（累计67）
   ● tutorial/14-final       端到端流水线 + 全部 Git 分支一键全链验收            │ +3（累计70）
```

**分支拓扑规则**（为什么这么设计）：
- `tutorial/01-skeleton` 是**孤儿分支**（无父提交）：用 `git checkout --orphan` + `git rm -rf .` 创建，
  保证"真正的空分支起步"——仓库历史里没有上一课的任何影子；
- 后续分支从上一分支**普通切出**：代码逐课累积，任意分支 `git checkout` 即得"学到该课为止"的完整可运行工程；
- **相邻分支 `git diff` = 本课增量**；`master` 与任一分支 `git diff` = 你的手写实现 vs 标准实现；
- 教程分支与 master **无共同祖先、永不合并**（结构上就不允许意外污染标准实现）。

---

## 4. 验收机制（verify.py 判卷机）

### 4.1 运行方式

```bash
.venv/Scripts/python.exe verify.py          # 全量（含训练类检查，约 1.5~3 分钟）
.venv/Scripts/python.exe verify.py --fast   # 跳过训练类，数秒
```

### 4.2 机制

1. 根 `verify.py` 扫描 `verify/` 下文件名以**两位数字开头**的 `NN_xxx.py`（保证执行顺序 = 课程顺序）；
2. 每个检查文件里有 `CHECKS = [(名称, 函数)]` 列表，由 `register("…")` 装饰器收集；
3. 逐项执行断言：PASS 绿勾 / FAIL 红叉+异常详情；**任一 FAIL 则进程退出码非 0**；
4. `--fast` 通过给检查函数一个带 `.fast=True` 的参数对象实现，慢训练类检查里读到 `.fast` 就提前 return。

### 4.3 检查风格（行为/数值/形状断言，绝不比对实现代码）

- 形状类：forward 输出各维度符合预期（如 `(B, S, hidden)`）；
- 数值类：与手写参考实现 allclose（容差默认 atol=rtol=1e-5）；
- 行为类：KV-cache 增量生成 == 全量生成；RoPE 旋转保持模长；掩码只作用于 assistant 片段；eos 早停……；
- 训练类（被 `--fast` 跳过）：真实跑一个微型 epoch，断言 loss 下降、权重文件存在、可严格加载；
- 转换类：round-trip 后 logits 一致 < 1e-3。

> 因为只验行为不验实现，你的解答和参考答案**代码可以完全不同**——只要数学上等价就能过验。
> 这正是"手写"的乐趣：机器只判对错，不判写法。

### 4.4 共享设施

- `verify/_common.py`：`TINY_CONFIG`（微型验收配置：hidden=96/layers=2/heads=4/kv=2/vocab=512/…）、
  `set_seed()`、`close()`（allclose 封装）；
- 单测辅助：数字前缀文件无法用点式 import，调试单个检查用
  `python -c "import importlib; importlib.import_module('verify.NN_xxx')"`。

---

## 5. 各课对照表（minimind3 产出 → minimind 标准实现）

| 课 | minimind3 产出 | 对照 minimind | 验收关键词 |
|---|---|---|---|
| 02 | `minimind3/config.py` → `MiniMindConfig` | `model/model_minimind.py` → `MiniMindConfig` | 默认值、派生尺寸、JSON 往返 |
| 03 | `minimind3/rms_norm.py` → `RMSNorm` | 同上 → `RMSNorm` | 缩放不变性、非中心化、fp32 |
| 04 | `minimind3/rope.py` → `precompute_freqs_cis / apply_rotary_pos_emb` | 同上 | 频率公式、模长保持、相对位置 |
| 05 | `minimind3/attention.py` → `Attention` | 同上 → `Attention` | GQA 展开、因果性、KV-cache 等价 |
| 06 | `minimind3/feed_forward.py` → `FeedForward` | 同上 → `FeedForward` | SwiGLU 公式、维度、梯度 |
| 07 | `minimind3/block.py` → `MiniMindBlock` | 同上 → `MiniMindBlock` | 残差恒等、pre-norm 展开等价 |
| 08 | `minimind3/model_body.py` → `MiniMindModel` | 同上 → `MiniMindModel` | buffer 不持久、KV 编排、全参数反传 |
| 09 | `minimind3/causal_lm.py` → `MiniMindForCausalLM(+generate)` | 同上 → `MiniMindForCausalLM` | 绑定、shift 损失、生成策略、往返 |
| 10 | `minimind3/tokenizer_utils.py + datasets.py` | `model/`(词表) + `dataset/lm_dataset.py` | round-trip、ChatML、掩码 |
| 11 | `minimind3/train_pretrain.py` | `trainer/train_pretrain.py` | loss 下降、断点续训 |
| 12 | `minimind3/train_full_sft.py` | `trainer/train_full_sft.py` | 掩码损失、内容级 CE 分离 |
| 13 | `minimind3/convert.py` | `scripts/convert_model.py` | 产物齐全、AutoModel 加载一致、脱离仓库可载 |
| 14 | `minimind3/e2e.py + tools/check_all_branches.sh` | `scripts/chat_api.py`/`eval_llm.py` | 端到端对话内容正确、全链验收 |

---

## 6. 学习流程（给学习者，每课通用）

1. `git clone` 本仓库 → `git checkout tutorial/01-skeleton`；
2. 读本课 `tutorial/NN-xxx/README.md`：**目标 → 理论 → 任务清单 → 验收 → 对照表 → 常见坑**；
3. **先合上所有代码文件**，按任务清单手写你的版本；写不动再偷看参考答案（同目录 `minimind3/`）；
4. 跑 `python verify.py` 直到 0 failed；
5. `git checkout tutorial/NN+1-xxx` 进入下一课；想复盘可用 `git diff`；
6. 全部完成后 `tools/check_all_branches.sh --fast` 一键全链验收。

> 💡 **手写是灵魂**：任务清单只给"类名/接口/行为要求"。强烈建议：写完 → 对照 `git diff`（本课增量）→
> 对照 `master`（标准实现）逐行复盘，三个层次全部对齐，才算真正吃透一课。

---

## 7. 理论覆盖矩阵（每课 README 内"理论"节的内容，均由浅入深展开）

| 课 | 理论主题（示例） |
|---|---|
| 01 | 语言模型范式（next-token prediction、链式法则）/ 为什么合成数据 / verify 机制 / git 分支工作流 |
| 02 | 什么是超参数 / intermediate_size 的 π 取整规则 / PretrainedConfig 机制与 bos/eos 坑 |
| 03 | 为什么归一化 / LayerNorm vs RMSNorm（省去均值中心化）/ fp32 内部精度 |
| 04 | 位置编码的必要性 / RoPE 旋转几何 / 频率公式推导 / 相对位置不变性 |
| 05 | 注意力几何（Q·K·V）/ 缩放点积 / 因果掩码低三角 / GQA 参数量权衡 / KV-Cache 的 O(T) 秘诀 |
| 06 | 非线性激活 / GELU vs SwiGLU / 门控机制 / 参数量经济学 |
| 07 | 残差连接的动机 / Pre-Norm vs Post-Norm / 梯度"高速公路" |
| 08 | Embedding 表 / 权重绑定的动机 / Parameter vs buffer / start_pos 与 KV 编排 |
| 09 | 下一个 token 损失（shift、ignore_index=-100）/ 训练与推理模式 / 采样策略全家桶 |
| 10 | BPE 分词原理 / ChatML 模板 / -100 掩码标签构造 / 截断与 padding |
| 11 | 优化器与学习率 / 余弦退火推导 / 梯度裁剪 / 梯度累积 / checkpoint vs 权重 |
| 12 | 监督微调的目标 / 为什么必须掩码（复读机病）/ 灾难性遗忘 / 学习率选择 |
| 13 | HF 目录格式 / auto_map / 动态加载（remote code）/ round-trip 一致性 / .pth 的"失忆"问题（sidecar） |
| 14 | 评估三种口径 / 流水线设计 / 全链验收脚本 / 进阶路线图 |

---

## 8. 存储与占用说明

- 教程增量（代码 + 文档 + 合成数据）**< 1MB**（实测：`data` 56K + `minimind3` 714K + `verify` 246K + 文档）；
- 不触发任何网络下载；不引入大权重产物（训练产物已 gitignore：`out/`、`checkpoints/`、`minimind3-hf/`、`*.pth`）；
- 仓库工作区散落的 `dataset/*.jsonl`（约 3GB，**未跟踪**）是 minimind 标准实现用的真实数据，
  与本教程无关；需要做大模型"真实训练（可选）"时可直接复用（schema 与本教程数据一致）。

---

## 9. 常见问题（FAQ）

**Q1：为什么不用 Jupyter Notebook？**
本教程强调"运行中的真实代码 + 自动化验收"，脚本式结构更贴近真实工程；每课的 README 承担讲解职责。

**Q2：为什么先模型后数据？**
模型类从第 2 课一路搭到第 9 课（边搭边验证），第 10 课用 tokenizer 把文本变成 token——零件齐了再引入数据流转更顺。
（想先看数据也完全可以，参考第 10 课的 README 独立阅读。）

**Q3：为什么用 git 分支而不是文件夹版本？**
分支让"上一课 vs 本课"变成一次 `git diff`，可复现、可回退、可对照；验收也是随分支累积的。

**Q4：课程里已有参考答案，会不会失去意义？**
答案与判卷标准都公开，这恰恰是它的意义：**对照式学习**。先写 → 验收 → 对答案 → 复盘，比看一遍视频深刻得多。

**Q5：全链验收跑多久？**
`--fast` 模式约 2 分钟遍历 14 分支；全量单分支约 1.5~3 分钟；最慢的端到端流水线约 1 分钟（CPU）。

---

## 10. 进阶方向（本期教程范围外，作为各课"进阶"指引）

MoE 架构（config 已预留字段）/ LoRA / DPO·GRPO·PPO / 思维链 SFT / 真实大语料预训练 / 多卡 DDP /
HF 生态兼容（Qwen3 格式转换，见 master `scripts/convert_model.py`）/ 量化（GGUF）/ vLLM 部署 /
YaRN 长上下文外推 / 更好的 tokenizer。

---

## 11. 维护指引

教程的维护背景、分支拓扑、验收框架约定、已知坑（Windows/CRLF、transformers bos/eos、
pyarrow 导入顺序、sidecar config、remote-code 缓存等）与发布流程，见仓库根维护手册
（pi 会在启动时自动加载 `AGENTS.md`）。新增/删除验收检查项时，记得同步该文档中的验收计数锚点表。