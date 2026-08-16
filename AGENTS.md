# minimind3 手写教程 · 总设计与维护手册（AGENTS.md）

> 本文件是**仓库根 AGENTS.md**：pi 启动时自动加载并拼入上下文（见 pi 文档 Context Files）。
> 它是本教程的**单一权威文档**——既是面向学习者/维护者的**总设计**，也是 pi 自动加载的**维护手册**。
> 因内容会注入每个会话，请保持精炼；改动前先读对应章节，改动后按第 8 节「维护闭环」清单收尾。
> 教学文档是各章 `tutorial/NN-xxx/README.md`；本文件不替代它们，只承载"设计-验收-维护"元信息。

# 1. 项目定位与约束

仓库本体 = minimind 标准实现（master，作者 jingyaogong）。教程 = 以 minimind 为标准答案，从**空白的 `minimind3/` 工作区**逐课手写一个实现等价的 **minimind3**。

**结构（2026-08 改版：14 分支链 → 单分支 tutorial-main）**：
- 全部学习/验收在**单一分支 `tutorial-main`**（基于旧链末支 14-final 派生的普通分支）上进行；
- `answers/minimind3/` + `answers/data/` = **参考答案**（完整实现与规范数据，先写后对）；
- 工作区 `minimind3/` 由学习者从零创建（只有第 01 课产物 `__init__.py`）；`data/` 第 10 课起生成（gitignore）；
- 旧 14 分支链 `tutorial/01-skeleton`…`tutorial/14-final` 仍存在（历史归档，不再用于教学）。
- 约束：**零 GPU**（验收用 `.venv/Scripts/python.exe`，torch 2.11 CPU）、**零下载**（数据由工具生成、词表从 answers/ 拷贝，均为仓库内文件）、**教程增量 < 1MB**（当前 answers 770K + 教程文档 ~120K + verify ~250K）。

学习入口 = 根 `README.md`（章节导航）；总设计问答（为什么不用 notebook / 为什么先模型后数据 / 答案会不会毁学习）见根 README 与各章正文。

# 2. 验收机制（改动必读）

- 根 `verify.py`：自动发现 `verify/NN_xxx.py`（按文件名排序），`register(name)` 装饰器注册 `CHECKS`。
  **单分支新增按课过滤**：`verify.py NN` 只导入/执行课号 ≤ NN 的检查模块（`--fast` 可叠加，跳过训练/生成类检查，在被跳过的检查函数里 `if getattr(args, "fast", False): return` 提前返回）。无参数 = 全部 14 课。
- `verify/_common.py`：`TINY_CONFIG`（hidden 96/layers 2/vocab 512/…）、`set_seed`、`close`（allclose atol=rtol=1e-5）。
- 验收哲学：检查是"行为指纹"——RoPE 保范、KV 增量==全量、清参恒等（残差）、CE gap（学到没）、1e-5 round-trip（转换无损）；不绑定风格/措辞。
- **结构护栏**：`verify/01_skeleton.py` 的 01.1（骨架+参考答案就位）与 01.x（**起点留白**：`minimind3/` 只允许 `__init__.py`、`data/` 必须不存在）防"把答案拷回工作区"；`verify/14_e2e.py` 的 14.2（单分支自洽+`verify.py 03` 过滤恰好 15 项）与 14.3（检查模块 ≥14 个、检查总量 ≥70）。
- 失败排查：`.venv/Scripts/python.exe verify.py 05 2>&1 | grep FAIL`，或 `importlib.import_module('verify.05_attention')` 单测（数字前缀无法点式导入）。

# 3. 代码模块与课次对照（改动必读）

| 课 | 手写文件（minimind3/） | 内容要点 | 对照 minimind 标准实现 |
|---|---|---|---|
| 01 | `__init__.py` | 包入口：docstring + `__version__ = "0.1.0"` | — |
| 02 | `config.py` | MiniMindConfig（model_type="minimind3"、派生尺寸、YaRN、estimate_parameter_count） | model/model_minimind.py |
| 03 | `rms_norm.py` | RMSNorm（fp32 内部归一化后 type_as） | 同上 |
| 04 | `rope.py` | precompute_freqs_cis / rotate_half / apply_rotary_pos_emb（GPT-NeoX 双半表） | 同上 |
| 05 | `attention.py` | GQA + QK-Norm + RoPE + KV-Cache + 因果掩码 + SDPA 快路径 + repeat_kv | 同上 |
| 06 | `feed_forward.py` | SwiGLU（gate/up/down，无 bias，ACT2FN["silu"]） | 同上 |
| 07 | `block.py` | MiniMindBlock（pre-norm 残差，命名 input_layernorm/post_attention_layernorm） | 同上 |
| 08 | `model_body.py` | MiniMindModel（embed+blocks+final norm+**persistent=False** RoPE buffer+KV 编排+aux_loss） | 同上 |
| 09 | `causal_lm.py` | MiniMindForCausalLM（权重绑定、shift 损失、自实现 generate） | 同上 |
| 10 | `tokenizer_utils.py` + `datasets.py` | AutoTokenizer 惰性加载、ChatML、generate_labels 掩码、Pretrain/SFT Dataset（**词表目录 `tokenizer/` 从 answers/ 拷贝**） | model/ 词表 + dataset/lm_dataset.py |
| 11 | `train_utils.py` + `train_pretrain.py` | get_lr 余弦、setup_seed、checkpoint/权重存取（含 config sidecar） | trainer/trainer_utils.py、train_pretrain.py |
| 12 | `train_full_sft.py` | SFT 循环（换数据+掩码损失+低 lr；复用 11 课的 _batches/_weight_path） | trainer/train_full_sft.py |
| 13 | `convert.py` | pth⇄HF（sidecar 优先推断配置 + 自包含 remote-code 拼接 + 反向） | scripts/convert_model.py |
| 14 | `e2e.py` | 全流程流水线（合成数据→预训练→SFT→转换→对话） | — |

# 4. 已知坑（维护时最容易踩，全部有教训记录）

- **Windows/GBK 控制台**打印中文乱码属正常，勿当 bug；decode 到含 U+010A(`\n`) 的文本 print 会 UnicodeEncodeError，测试输出用 ASCII 标记（如 `[E2E] Q:`/`loss X`）。
- **transformers>=4.45** 的 `PretrainedConfig.__init__` 会把 bos/eos/pad_token_id 当作显式参数覆盖子类默认值 → 子类必须 `kwargs.pop('bos_token_id')` 后显式传 `super().__init__(bos_token_id=…)`；minimind 自己的 MiniMindConfig 同样有此潜在 None（不是 bug）。
- **Windows 上必须先 import datasets 再 import torch**（pyarrow DLL 冲突）；`tools/make_synthetic_data.py` 生成 `data/`（根锚定 `/data/` 已 gitignore；`answers/data/` 是跟踪文件）。
- **repeat_kv 用 expand+reshape**（顺序 [kv0,kv0,kv1,…]），不是 `repeat`。
- attention：KV 拼接发生在 apply_rotary_pos_emb **之后**（历史 K 已旋转）；因果掩码只加在 `scores[..., -seq_len:]` 新列上。
- generate：top-p 需要 `mask[..., 1:] = mask[..., :-1].clone()` 移位；增量模式只喂 `input_ids[:, past_len:]`；权重绑定在 `post_init()` 之前做。
- 训练脚本坑：`save_interval=0` 时 `%` 运算的短路顺序；resume 跳过全部 batch 时 `step` 需预初始化；checkpoint 无条件每 epoch 尾保存、权重保存才受 save_interval 控制。
- 转换/加载坑：`convert.py` 的 standalone 拼接 **config.py 必须放 _SRC_FILES 首位**且文件头加 `from __future__ import annotations`；transformers 会缓存 remote code 于 `~/.cache/huggingface/modules/transformers_modules/hf/`，改了建模文件需清理；`.pth` 不含非张量超参 → 权重旁必须写 `*.config.json` sidecar（train_utils.save_weights 自动写），infer 优先读它、兜底 q_norm.weight 取 head_dim（gcd 兜底不可靠）。
- 文档同步坑：`answers/minimind3/` 是工作区实现的"规范抄本"，改动实现必须同步 answers（或反之）；教程分支已归档，勿再在旧链上改文档。
- 本机 git 在 Windows 提示 "LF will be replaced by CRLF" 属正常（core.autocrlf）。

# 5. 收敛配置与实验事实（CPU 临界值，勿随意缩水）

- 12.2 探针/e2e：hidden 128 / layers 2 / SFT 4 epoch / lr 5e-4 / accum 1 / seq 96 → 内容级 CE 分离（好 0.93 vs 坏 1.37）；e2e 贪心生成可背出"猫是一种会抓老鼠的动物。""狗是人类忠诚的朋友。"；CE gap 即"学到没"的机器判据。
- 默认 tiny 配置（hidden 96/2 层）预训练 1 epoch 收敛；SFT 内容级需要 hidden 128（96 只在 loss 层收敛）。
- 真实 GPU：预训练 lr 5e-4、SFT 1e-5~2e-5、更长 max_seq_len（教程仅演示原理）。

# 6. 环境与资源

- 验收解释器：仓库根 `.venv/Scripts/python.exe`（torch 2.11+cpu、transformers 4.57.6、datasets 3.6.0、tokenizers 0.22.2）；系统默认 python **没有 torch**。
- 产物目录（out/、checkpoints/、minimind3-hf/）均已 gitignore。
- **严禁引入网络下载**。数据：`python -m tools.make_synthetic_data --out data --num 300`（确定性；`answers/data/` 为规范拷贝）。词表：第 10 课把 `answers/minimind3/tokenizer/` 拷进工作区（`git show`/`cp -r` 均可——教程分支是孤儿，看不见 master 的 model/）。
- 仓库根散落的 `dataset/*.jsonl`（约 3GB）是标准实现用的真实数据，**未跟踪**，与教程无关。
- Git 身份 billpenn2005，origin = https://github.com/billpenn2005/minimind；发布 = `git push origin master tutorial-main`（必要时 `git push origin refs/heads/tutorial/*:refs/heads/tutorial/*` 同步归档分支）。

# 7. 维护闭环（每次改动后）

1. 小步验证：`.venv/Scripts/python.exe verify.py --fast`（全部但跳过训练；单课快查用 `verify.py NN`）；
2. 若动了训练类检查 → 全量 `verify.py`（含 e2e，约 2~4 分钟）；若动了模型源码 → 重跑 lesson-13 相关检查并确认 answers/minimind3 与产物同步；
3. QA 用"答案回拷法"：临时 `cp -r answers/minimind3 minimind3/` 跑全量，确认验收框架对完整实现全绿，然后 `rm -rf minimind3` 恢复留白；
4. 改动文档（README / AGENTS.md / 各章 README）后：同步 `answers/README.md` 对照表与根 README 导航（章号/检查数）；
5. 更新本文件已过时的事实（验收数字、护栏、新坑）。
6. 先 verify 后 commit，勿链式 `verify; git commit`（失败会误提交）。

# 8. 常见任务速查

- **给某课加检查**：改 `verify/NN_x.py` 注册新 fn → 同步 `verify/14_e2e.py` 14.3 的总量下限与根 README 累计数（如有变化）。
- **开新章**：在 `tutorial/NN-xxx/README.md` 按统一模板（导航/目标/理论/任务要求/验收/参考答案/坑/对照）撰写；创建对应 `verify/NN_xxx.py` 并把文件加入 01.1 清单（如需）。
- **重跑端到端**：`.venv/Scripts/python.exe -m minimind3.e2e`（~1 分钟，产物 out/e2e/）。
- **与标准实现对比**：`git diff tutorial-main master -- minimind3/…` 或 `bash tools/fetch_reference.sh`。
- **发布**：`git push origin master tutorial-main`。

# 9. 进阶方向（超出本教程范围）

MoE（config 已含字段）/ LoRA / DPO·GRPO·PPO / 思维链 / Qwen3 生态格式（minimind 的 convert_model.py 即此路线）/ GGUF 量化 / vLLM 部署；换大模型只需改 hidden/layers + 真实语料（dataset/*.jsonl，schema 完全兼容）+ GPU。