# 第 14 课 · 端到端流水线与毕业验收

> [上一课 13-convert](../13-convert/README.md) ← [目录](../../README.md)
>
> 难度：★★☆☆☆（收官课）｜ 预计用时 2 小时 ｜ 前置：全部 13 课

## 0. 本课目标

- [ ] 理解三级评估口径（训练指标/概率口径/行为口径）；
- [ ] 手写 `minimind3/e2e.py` 全流程流水线；
- [ ] 亲眼见证模型"背出事实"（行为级验收）；
- [ ] 毕业验收：`verify.py`（全部 71 项）。

## 1. 理论速览：怎么判断"学会了"？

| 口径 | 问的问题 | 用到哪 |
|---|---|---|
| 训练指标 | loss 降了吗？ | 11/12 课 |
| 概率口径 | 正确回答 CE 更低吗？ | 12.2（gap≈0.44） |
| **行为口径** | **生成出来的文本对吗？** | 本课：对话测试 |

三级递进，最后必须"开口说话"。封闭世界（300 条事实 + 探针校准的 128/2/4ep/5e-4 配置）里小模型也能行为正确。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/e2e.py`

**模块级**：
- `REPO_ROOT: Path` = `Path(__file__).resolve().parent.parent`；
- `run(cmd: list[str]) -> None`：在 REPO_ROOT 下 `subprocess.run`，returncode≠0 时 `sys.exit(f"[e2e] 失败: …")`；
- `stage(name: str) -> None`：打印 `\n========== [E2E] {name} ==========`（**验收解析这些标记**）。

**CLI 参数**：

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `--workdir` | str | `out/e2e` | 产物目录 |
| `--hidden` | int | 128 | 隐藏维（内容级收敛临界点，**勿缩水**） |
| `--layers` | int | 2 | 层数 |
| `--pt-epochs` | int | 1 | 预训练轮数 |
| `--sft-epochs` | int | 4 | SFT 轮数 |
| `--seq` | int | 96 | 序列长 |
| `--skip-train` | flag | False | 跳过训练（复用已有权重） |

**`main()` 五阶段**：
1. `stage("合成数据")`：`data/tiny_pretrain.jsonl` 缺失时 `python -m tools.make_synthetic_data --out data --num 300`；
2. `stage("预训练 …")`：子进程 `-m minimind3.train_pretrain --save_weight e2e_pt --save_dir workdir --epochs pt-epochs --batch_size 8 --max_seq_len seq --hidden_size hidden --num_hidden_layers layers --accumulation_steps 2 --log_interval 25 --save_interval 0 --seed 0`；
3. `stage("SFT …")`：`-m minimind3.train_full_sft --from_weight e2e_pt --save_weight e2e_sft --epochs sft-epochs --batch_size 8 --max_seq_len seq --hidden_size hidden --num_hidden_layers layers --accumulation_steps 1 --log_interval 25 --save_interval 0 --seed 0 --learning_rate 5e-4`；
4. `stage("HF 转换（自包含 remote-code）")`：inline `python -c "from minimind3.convert import convert_torch2transformers; convert_torch2transformers(r'<pth>', r'<hf_dir>')"`；
5. `stage("对话测试（AutoModelForCausalLM + 本机分词器）")` → `_chat_demo(hf_dir, hidden)`。

**`_chat_demo(hf_dir: Path, hidden: int) -> None`**：
- `AutoModelForCausalLM.from_pretrained(str(hf_dir), trust_remote_code=True).eval()` + `AutoTokenizer.from_pretrained(str(hf_dir))`（**先 `sys.path.insert(0, REPO_ROOT)`** 供 remote-code 解析）；
- 两个封闭问题 `["什么是猫？", "介绍一下狗。"]`：
  - `prompt = build_chat_prompt([{"role":"user","content":q}])` → `tokenizer(prompt, add_special_tokens=False).input_ids`；
  - `model.generate(torch.tensor([ids]), max_new_tokens=24, do_sample=False, top_k=0, top_p=1.0)`；
  - 解码 `out[0][len(ids):]`，去掉 `<|im_end|>` 与换行；
  - 打印 `[E2E] Q: {q}` 与 `[E2E] A: {answer}`（**验收解析 `[E2E] A:` 行**）。

## 3. 手写步骤

实现 §2.1 → `.venv/Scripts/python.exe verify.py 14`（全量约 2~4 分钟，含 e2e；`--fast` 跳过 14.1）。

## 4. 验收解读（verify/14_e2e.py）

| 检查（新） | 验什么 |
|---|---|
| 14.1 | e2e 全流程五阶段标记齐全；**两条回答命中事实关键词**（`猫`+`抓/老鼠`；`狗`+`忠诚/朋友`）；HF 产物存在 |
| 14.2 | 单分支自洽：`answers/` 三件套在；`check_all_branches.sh` 已移除；`verify.py 03 --fast` 恰好 16 项全过（按课过滤可用） |
| 14.3 | 检查模块 ≥14 个、总量 ≥70（本课经 01.x 新增后实为 71） |

> 答案断言用**关键词包含**而非全等（不同平台浮点/分词有细微差异）。

## 5. 参考答案

`answers/minimind3/e2e.py`（先写后对）。

## 6. 常见坑

- **改坏训练配置**：内容级收敛点很脆（hidden 128/4ep/5e-4/acc1 是探针校准值）；调参先复现 12.2 的 CE gap；
- **`[E2E]` 行格式**：验收按行解析，别改标记格式；
- **remote-code 加载失败**：`sys.path.insert(0, REPO_ROOT)` 必须在 import 模型前；
- **Windows 中文输出**：GBK 终端乱码正常；验收解析的是 ASCII 标记行。

## 7. 对照标准实现

| 你手写 | minimind 标准 |
|---|---|
| `minimind3/e2e.py` | `scripts/chat_api.py` / `eval_llm.py`（推理侧） |

minimind 推理端更丰富（OpenAI API 服务、WebUI、tool call）；教程 e2e 是最小可验证闭环。

## 8. 总结：14 章你造了什么？

```
minimind3/   ← 你手写的完整语言模型（约 500 行核心代码）
  模型：config / RMSNorm / RoPE / GQA-Attention / SwiGLU / Block / Model / ForCausalLM
  数据：分词器接入 / ChatML 数据集 / -100 掩码
  训练：预训练循环（cosine/clip/累积/续训）+ SFT（掩码微调）
  工程：pth⇄HF 转换（自包含 remote-code）+ e2e 流水线 + 71 项程序化验收
```

**毕业三步**：
```bash
.venv/Scripts/python.exe verify.py 14  --fast    # 快速（跳过 e2e 训练）
.venv/Scripts/python.exe verify.py              # 全量（含 e2e，约 2~4 分钟）== 71 passed, 0 failed
```

往大模型进阶只需三换：hidden/layers 增大（参数量↑）、真实语料（`dataset/*.jsonl`，schema 完全兼容）、GPU。进阶方向：MoE / LoRA / DPO·GRPO·PPO / 思维链 / Qwen3 生态格式 / GGUF 量化 / vLLM 部署。

🎉 恭喜毕业！回到根目录读 `model/`、`trainer/` 里的 minimind 标准实现——你已经完全能看懂它了。