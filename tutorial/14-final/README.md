# 第 14 课 · 端到端测试与全链验收（收官）

> 难度：★★☆☆☆（内容平缓，成就感拉满）｜ 预计用时：2~3 小时 ｜ 前置：全部 13 课
> 学完本课你将拥有：**一条 1 分钟跑完的完整流水线**（从空权重到会说话的模型）+ **一键验收全部 14 分支**的脚本。

---

## 0. 本课目标

- [ ] 理解评估 LLM 的**三种口径**（训练指标 / 概率口径 / 行为口径）为什么逐级递进；
- [ ] 手写 `minimind3/e2e.py`：合成数据 → 微预训练 → 微 SFT → HF 转换 → AutoModel 对话；
- [ ] 亲手见证"模型背出事实"（行为级验收：答案命中关键词）；
- [ ] 手写 `tools/check_all_branches.sh` 全链验收脚本并跑通；
- [ ] 14 门课全部验收（累计 70 项）。

---

## 1. 理论：怎么判断"模型学会了"？

| 口径 | 问的问题 | 本教程用法 | 局限 |
|---|---|---|---|
| 训练指标 | loss 降了吗？ | 11/12 课数值断言 | loss 降 ≠ 学到内容 |
| 概率口径 | 正确回答的 CE 更低吗？ | 12.2 的 CE gap | 概率对 ≠ 生成对 |
| **行为口径** | **生成出来的文本对吗？** | 本课 e2e 对话测试 | 需要更大的模型才能泛化 |

三者递进：先用 loss 快筛，再用 CE 探针，**最后一定要"开口说话"来验收**。小模型在封闭世界
（数据就 300 条事实）可以做到行为正确——这正是收官的含义。

## 2. 流水线设计（e2e.py）

```
1. 合成数据（tools/make_synthetic_data，无网络）
2. 微预训练（e2e_pt_128.pth，1 epoch）
3. 全参 SFT （e2e_sft_128.pth，4 epoch —— 12.2 探针验证过的可收敛配置）
4. HF 转换  （自包含 remote-code 目录 hf_128/，13 课产物）
5. 对话测试 （AutoModelForCausalLM + tokenizer → ChatML → generate → decode）
```

**为什么训练配置不缩水？** `hidden=128 / layers=2 / SFT 4 epoch / lr 5e-4 / acc=1` 是 12.2 探针
验证的"CPU 上内容级收敛临界点"。任何参数被"优化"掉（比如 2 epoch），模型就背不出事实，
行为验收会红。**这是教程用真实实验校准过的配置，别轻易改**。（想加速？`:full` 模式跑的
`--skip-train` 复用已有权重。）

生成的验收答（贪心、无随机）实测：
```
Q: 什么是猫？ → A: 猫是一种会抓老鼠的动物。
Q: 介绍一下狗。 → A: 狗是人类忠诚的朋友。
```

## 3. 全链验收脚本（check_all_branches.sh）

```bash
bash tools/check_all_branches.sh --fast   # 依次 checkout 14 分支 + 各自 verify --fast（约 2 分钟）
bash tools/check_all_branches.sh --full   # 全量版（约 10 分钟）
```

脚本逻辑：`git for-each-ref` 枚举 `tutorial/*` 分支（`sort -V` 保证 01<02<…<14）→ 逐分支
`checkout` → 跑 `verify.py` → 汇总绿/红表 → 恢复原分支 → 任一失败退出码非 0。
用 `PY` 环境变量可换解释器：`PY=python3 bash tools/check_all_branches.sh`。

---

## 4. 手写任务清单

1. `e2e.py`：
   - 数据缺失时自动生成；
   - 用 `subprocess` 调 `train_pretrain / train_full_sft`（参数见 §2）；
   - 调 `convert_torch2transformers`（13 课）产出 hf 目录；
   - 对话测试：`AutoModelForCausalLM.from_pretrained(hf, trust_remote_code=True)` +
     `build_chat_prompt`（10 课）→ `generate`（09 课）→ decode，打印 `[E2E]` 标记行（验收解析用）；
2. `tools/check_all_branches.sh`；
3. 验收（全量，含 e2e，约 2~3 分钟）。

```bash
.venv/Scripts/python.exe verify.py
```

## 5. 验收解读（verify/14_e2e.py）

| 检查 | 验什么 |
|---|---|
| 14.1 | e2e 子进程全流程：五个阶段标记齐全；**两条回答命中事实关键词**（`猫`+`抓/老鼠`；`狗`+`忠诚/朋友`）；HF 产物存在 |
| 14.2 | 全链脚本存在、`bash -n` 语法通过、教程分支恰好 14 个、末分支 == tutorial/14-final |
| 14.3 | 检查模块完整性护栏：每个 `verify/NN_*.py` 非空；检查总数 ≥ 逐课锚点表（05→5, 10→10, …, 13 课 →67, 14 课 →70） |

> 14.3 是给未来维护者的"护栏"：任何人删空某课的检查，全链立刻红。锚点表在
> `verify/14_e2e.py` 的 `floors` dict——**未来新增检查项要同步更新它**。

## 6. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/e2e.py` | `scripts/chat_api.py` / `eval_llm.py`（推理侧） |

minimind 的推理端更丰富（OpenAI API 服务、WebUI、tool call、reasoning）；教程的 e2e 是**最小可验证闭环**，
服务化列进阶。

## 7. 常见坑

- **改坏训练配置**：内容级收敛点很脆（§2 强调），调参请以 12.2 的 CE gap 复现为准；
- **答案断言别用完全相等**：不同平台浮点/分词可能有细微差异，验收用**包含关系**（关键词）而非 ==；
- **bash 脚本在 Windows**：用 `git bash`/WSL 跑；脚本里对 `.venv/Scripts/python.exe` 的存在性做探测，
  `PY` 变量可覆盖；
- **`sort -V` 不可用**（老版 sort）：分支顺序会错，脚本断言末分支 == tutorial/14-final 兜底。

## 8. 总结：这 14 课你造了什么？

```
miniMind3 全家桶（约 500 行手写核心代码）：
  模型：config / RMSNorm / RoPE / GQA-Attention / SwiGLU / Block / Model / ForCausalLM
  数据：tokenizer 接入 / ChatML 数据集 / -100 掩码
  训练：预训练循环（cosine/clip/累积/续训）+ SFT（掩码微调）
  工程：pth⇄HF 转换（自包含 remote-code）+ e2e 流水线 + 70 项程序化验收
```

往里换三样就能进阶成大模型：更大的 hidden/layers（参数量 ↑）、真实语料（`dataset/*.jsonl`，
schema 完全兼容）、GPU（训练脚本直接支持）。进阶路线：MoE / LoRA / DPO·GRPO·PPO / 思维链 /
Qwen3 生态格式 / GGUF 量化 / vLLM 部署。

最后跑一遍总验收，然后…… 🎉 恭喜毕业！去 `master` 分支读 minimind 标准实现吧——你已经能看懂它了。