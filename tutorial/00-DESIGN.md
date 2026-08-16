# MiniMind3 手写实现教程 · 总设计文档

> 定位：基于本仓库（jingyaogong/minimind，即"标准实现"），用 **git 分支链** 方式，
> 从空分支开始逐课手写构建一个 **minimind3**（MiniMind 架构的从零实现）。
> 每个分支 = 一课；每一课都有 **程序化验收机制**（`verify.py`），
> 全程 **不依赖 GPU**（真实训练为可选），**不下载数据集与模型权重**（数据为代码内合成）。

---

## 0. 教程目标与范围

**范围（本期）**：模型本身 + 预训练 + 全参 SFT + HF transformers 格式转换 + 最终测试。

**显式排除（进阶扩展，不在本期分支内）**：LoRA、DPO/GRPO/PPO、数据蒸馏、MoE、量化、
真实数据集、多卡/分布式、flash-attention 原语优化。文档中以"进阶指引"给出方向。

**质量标准**：教程中每一行代码都可对照 minimind 标准实现（本仓库 master 分支）逐模块比对。

---

## 1. 工程约束（验收环境必须满足）

| 约束 | 说明 |
|---|---|
| 无 GPU | 验收环境为 CPU（本项目已验证：torch 2.11.0+cpu / transformers 4.57.6 / datasets 3.6.0） |
| 小存储 | 不下载任何数据集与权重；训练数据由 `tools/make_synthetic_data.py` 在本地生成（<200KB） |
| 可复现 | 全部合成数据 + 固定随机种子，验收断言使用数值/形状/行为检查，而非"训练曲线好看" |
| 快速 | 默认训练配置为微型（`hidden=96, layers=4` 级别），整篇验收几分钟内完成 |
| 手写 | 每课文档给出"任务清单"，要求学习者先手写；分支内代码为参考答案；`verify.py` 以行为/形状/数值验收，不绑定实现细节 |

---

## 2. 分支图谱（链式累积）

```
master  (minimind 标准实现 / 对照参考)
   │
   ● tutorial/01-skeleton   项目骨架：包结构、合成数据工具、verify 框架
   ● tutorial/02-config     MiniMindConfig（PretrainedConfig 子类 + 默认值推导）
   ● tutorial/03-rmsnorm    RMSNorm
   ● tutorial/04-rope       RoPE：precompute_freqs_cis / apply_rotary_pos_emb
   ● tutorial/05-attention  Attention：GQA + QK-Norm + RoPE + KV-Cache + 因果掩码
   ● tutorial/06-feedforward  SwiGLU FeedForward（gate/up/down 三段式）
   ● tutorial/07-block      MiniMindBlock：Pre-Norm 残差堆叠
   ● tutorial/08-model      MiniMindModel：Embedding + 层堆叠 + 最终 Norm + RoPE buffer
   ● tutorial/09-causal-lm  MiniMindForCausalLM：lm_head / 权重绑定 / 损失 / generate
   ● tutorial/10-data       Tokenizer 接入 + Pretrain/SFT 数据集（回答掩码）
   ● tutorial/11-pretrain   预训练循环：cosine LR、梯度裁剪、累积、checkpoint/续训
   ● tutorial/12-sft        全参 SFT：仅 assistant 片段计算损失
   ● tutorial/13-convert    torch(pth) ⇄ HF transformers 互转 + AutoClass 注册
   ● tutorial/14-final      端到端：初始化→微预训练→微 SFT→转换→生成采样→全量验收
```

规则：
- `tutorial/01-skeleton` 为 **孤儿分支**（无父提交，真正的"空分支"起步）；
- 后续分支依次从上一分支切出（普通分支），因此任意分支 `git checkout` 即得到该课为止的完整可运行工程；
- 相邻分支 `git diff` 即"本课增量"；master 与任一分支 `git diff` 即"手写实现 vs 标准实现"。

---

## 3. 每课的验收机制（verify.py 规范）

- 仓库根目录统一 `verify.py`，**逐课累积**：每课新增一节检查函数，旧检查保留（回归）。
- 运行方式：`python verify.py`（全部检查）；`python verify.py --fast`（跳过较慢的训练类检查）。
- 输出格式：逐检查项 `[PASS]/[FAIL]`，末尾汇总，任一 FAIL 则退出码非 0。
- 检查风格：形状/数值/行为断言（如 RoPE 旋转后模长不变、KV-cache 增量与全量等价、
  tokenizer round-trip、掩码只作用于 assistant 片段、训练 loss 在合成数据上可下降、
  转换回读 logits 一致 <1e-3），**不做死板实现比对**，因此学习者手写实现同样可过验。
- 配套 `tools/check_all_branches.sh`（于 lesson-14 提供）：遍历全部分支跑 verify，输出整体绿/红表。

---

## 4. 各课对照表（minimind3 文件 → minimind 标准实现）

| 课 | minimind3 产出 | 对照 minimind |
|---|---|---|
| 02 | `minimind3/config.py → MiniMindConfig` | `model/model_minimind.py → MiniMindConfig` |
| 03 | `minimind3/rms_norm.py → RMSNorm` | 同上 → `RMSNorm` |
| 04 | `minimind3/rope.py → precompute_freqs_cis/apply_rotary_pos_emb` | 同上 |
| 05 | `minimind3/attention.py → Attention` | 同上 → `Attention` |
| 06 | `minimind3/feed_forward.py → FeedForward` | 同上 → `FeedForward` |
| 07 | `minimind3/block.py → MiniMindBlock` | 同上 → `MiniMindBlock` |
| 08 | `minimind3/model_body.py → MiniMindModel` | 同上 → `MiniMindModel` |
| 09 | `minimind3/causal_lm.py → MiniMindForCausalLM(+generate)` | 同上 → `MiniMindForCausalLM` |
| 10 | `minimind3/tokenizer_utils.py + datasets.py` | `model/`(tokenizer 文件) + `dataset/lm_dataset.py` |
| 11 | `minimind3/train_pretrain.py` | `trainer/train_pretrain.py` |
| 12 | `minimind3/train_full_sft.py` | `trainer/train_full_sft.py` |
| 13 | `minimind3/convert.py` | `scripts/convert_model.py` |
| 14 | `minimind3/e2e.py + tools/check_all_branches.sh` | `scripts/chat_api.py`(生成侧)/`eval_llm.py` |

---

## 5. 学习流程（给学习者）

1. `git clone` 本仓库 → `git checkout tutorial/01-skeleton`；
2. 读 `tutorial/01-skeleton/README.md`（与每课的 `tutorial/NN-xxx/README.md` 同样结构）：
   目标 → 理论 → 任务清单 → 验收 → 对照表 → 常见坑；
3. 按任务清单**手写**实现（或先读参考答案再默写）；写完后 `python verify.py` 验收；
4. 通过后 `git checkout tutorial/NN+1` 进入下一课（也可用 `git diff tutorial/NN tutorial/NN+1` 查看本课增量）；
5. 全部完成后运行 `tools/check_all_branches.sh` 进行全链验收。

> 手写原则：文档中"任务清单"只给出类名/接口/行为要求；强烈建议先不看参考答案编写，
> 再对照 `git diff` 与 minimind 标准实现逐行复盘。

---

## 6. 理论覆盖矩阵（每课 README 内的"理论"节）

- 01 语言模型范式（next-token prediction）/ 项目结构 / git 分支工作流
- 02 超参数如何推导（intermediate_size 的 π 取整规则等）/ PretrainedConfig 机制
- 03 LayerNorm vs RMSNorm：为什么省去均值中心化 / fp32 内部精度
- 04 RoPE 的旋转几何：频率公式、位置内积相对性、KV 缓存与位置连续
- 05 GQA 与 MHA 的参数量权衡 / QK-Norm / 因果掩码的低三角结构 / SDPA
- 06 SwiGLU 的门控非线性 / 参数量对比 GELU 双线性
- 07 Pre-Norm 残差：收敛稳定性 / 残差连接的梯度传播
- 08 Embedding 表 / 参数共享(权重绑定)的动机 / buffer 与 parameter 的区别
- 09 下一个 token 损失（shift、ignore_index=-100）/ 采样生成：温度/顶 k / 顶 p / 重复惩罚
- 10 BPE 分词原理 / chat template / 回答掩码标签构造
- 11 预训练：余弦退火 LR、梯度裁剪、累积、checkpoint/resume、loss scale
- 12 SFT：监督微调目标、掩码的必要性（输出侧才监督）
- 13 HF 权重格式：config.json/pytorch_model.bin/safetensors/tokenizer / AutoClass 注册 / round-trip
- 14 综合：从零训练→微调→部署的最小闭环 & 评估口径

---

## 7. 存储与占用说明

- 教程增量（代码+文档+合成数据）新增 **< 1 MB**；
- 不触发任何网络下载；不引入大权重；
- 仓库工作区已有的 `dataset/*.jsonl`（约 3 GB，未跟踪）与本教程无关，仅可作"真实训练（可选）"数据源，
  不需要时可自行删除释放空间。

---

## 8. 进阶方向（教程范围内不做，文档中给指引）

MoE 架构 / LoRA / DPO·GRPO·PPO / 思维链 SFT / 真实大语料预训练 / 多卡 DDP / HF 生态兼容（Qwen3 格式）/
量化（GGUF） / vLLM 部署。

---

## 9. 维护指引

整套教程的维护背景、分支拓扑、验收框架约定、已知坑与发布流程，已沉淀为仓库根
[`AGENTS.md`](../AGENTS.md)（pi 启动时自动加载进上下文）。修改代码/验收/文档前先读它；
新增检查项记得同步其中 `floors` 锚点表。