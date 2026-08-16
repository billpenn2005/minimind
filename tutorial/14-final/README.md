# 第 14 课 · 端到端测试与全链验收

## 目标

- 手写 `minimind3/e2e.py`：把前 13 课的产物串成一条"从零到可用"的流水线；
- 提供 `tools/check_all_branches.sh`：一次跑完 14 个分支的程序化验收；
- 理解评估口径：什么算"模型学会了"。

## 理论：评估一个微型 LLM 的三种口径

| 口径 | 问题 | 本教程的用法 |
|---|---|---|
| 训练指标 | loss 是否下降、收敛值 | 每课 verify 的数值断言 |
| 概率口径 | 正确回答的 CE < 错误回答的 CE | lesson-12 的掩码有效性 |
| 行为口径 | 生成的文本是否"像样/正确" | 本课 e2e 的对话测试 |

三者逐步升级：loss 下降 ≠ 学到内容；内容级 CE 区分 ≠ 能开口说话；
只有**生成出来的文本内容正确**（行为口径）才算端到端闭环。
e2e 里我们问两个封闭事实问题（"什么是猫？"/"介绍一下狗。"），
贪心生成并要求回答命中事实关键词——这是本教程能给出的最强 CPU 验收。

## e2e 流水线五步

```
1. 合成数据   （data/tiny_*.jsonl，无网络）
2. 预训练     （e2e_pt_128.pth，1 epoch）
3. 全参 SFT   （e2e_sft_128.pth，4 epoch —— 12.2 探针验证过的可收敛配置）
4. HF 转换    （自包含 remote-code 目录 hf_128/）
5. 对话测试   （AutoModelForCausalLM 加载 → ChatML → generate → decode）
```

全程 ~1 分钟（CPU，hidden=128 / 2 层 / 6400 词表）。

## 全链验收脚本

```bash
tools/check_all_branches.sh --fast    # 每分支跑 verify --fast（几分钟）
tools/check_all_branches.sh --full    # 每分支全量 verify（约 10+ 分钟）
```

脚本语义：遍历 `tutorial/*` 分支 → checkout → `verify.py` → 汇总红绿表 → 恢复原分支。

## 常见坑

- e2e 的训练配置不能随意缩水：复用 12.2 探针验证过的
  `hidden=128 / acc=1 / epochs=4 / lr=5e-4`（内容级收敛的临界配置）；
- 生成质量检查用**包含关系**而非完全相等（不同平台浮点/分词可能有细微差异）；
- `check_all_branches.sh` 依赖 `sort -V` 排序分支名（01 < 02 < ... < 14）。

## 进阶方向（本教程范围之外，给方向）

- 更大模型/真实语料（把 hidden 调到 768、数据换成 dataset/*.jsonl，需 GPU）；
- MoE（use_moe=True 已预留字段）、LoRA 微调、DPO/GRPO 对齐；
- Qwen3 生态格式转换（复用 transformers 原生类，任何推理框架可加载）；
- 量化（GGUF/safetensors-int4）、vLLM 部署。

## 总验收

```bash
# 在 tutorial/14-final 上
.venv/Scripts/python.exe verify.py            # 全量（含 e2e，约 2~3 分钟）
.venv/Scripts/python.exe verify.py --fast     # 跳过训练类（数秒）
tools/check_all_branches.sh --fast            # 全 14 分支绿/红表
```