# minimind3 手写教程 · 总设计与维护手册（AGENTS.md）

> 本文件是**仓库根 AGENTS.md**：pi 启动时自动加载并拼入上下文（见 pi 文档 Context Files）。
> 它是本教程的**单一权威文档**——既是面向学习者/维护者的**总设计**，也是 pi 自动加载的**维护手册**。
> 因内容会注入每个会话，请保持精炼；改动前先读对应章节，改动后按第 8 节「维护闭环」清单收尾。
> 每课教学文档是独立的 `tutorial/NN-xxx/README.md`（课程内容），本文件不替代它们，只承载"设计-验收-维护"元信息。

# 1. 项目定位与约束

仓库本体 = minimind 标准实现（master，作者 jingyaogong）。教程 = 以 minimind 为标准答案，从**空分支**逐课手写一个实现等价的 **minimind3**。三大约束：**零 GPU**（验收只用 `.venv/Scripts/python.exe`，torch 2.11 CPU）、**零下载**（数据代码内合成、词表从 git 对象导出）、**教程增量 < 1MB**（当前 data 56K + minimind3 ~714K + verify ~246K + 文档 ~110K）。

总设计 = 本文件。标准实现对照：`tools/fetch_reference.sh` 从 master 导出到 `reference/` 供逐行对比。

## 1.1 读者画像与学习目标

- 读者：零基础起步（只要 Python≥3.10 + git）；期望经此教程看懂 minimind 标准实现。
- 核心目标：从空白分支逐课手写 config→RMSNorm→RoPE→Attention→FFN→Block→Model→ForCausalLM→数据→预训练→SFT→pth⇄HF 转换→端到端验收，最终能用一句话解释"大语言模型为什么能生成"。
- 学习节奏建议：模型核心 8 课（01-08）每课 1~2 小时，`09-causal-lm` 与 `10-data` 稍重，训练两课含真实 CPU 训练。卡课超过 2 小时 → 先读 `minimind3/` 预写实现（它就是"标准答案与评分标准"），读懂后照抄并跑验收。

## 1.2 为什么是"文档 + 分支"结构（设计问答）

- **为什么不用 notebook**：notebook 无法程序化验收、无法做分支级回归；verify.py + git 分支是可持续的验收闭环。
- **为什么先模型后数据**：模型（01-09）是纯计算结构，CPU 可即时验证；数据/训练（10-14）依赖前面结构。
- **为什么每课一个分支**：相邻分支 diff = 该课增量（可 git diff 查看"这课改了什么"）；分支即 checkpoint。
- **预写答案会让人不看代码就抄吗**：答案与 verify 同仓，是"脚手架 + 评分标准"；文档要求先自己写再对照。
- **估算时长**：模型核心每课 1~2h，训练与转换 2~3h，总约 25~35h。

# 2. 学习路径与分支拓扑（不可破坏）

- 孤儿分支 `tutorial/01-skeleton` **没有父提交**（`git checkout --orphan` + `git rm -rf .` 创建，工作区未跟踪文件保留）。
- `tutorial/02-config` … `tutorial/14-final` 每个都是前一个分支的普通子分支 → 代码逐课累积，相邻分支 diff = 该课增量。
- 教程分支与 master **无共同祖先，永不合并**。master 只承载本文件 + README 入口。
- 分支命名：`tutorial/NN-题名`（NN 两位）。14 课：skeleton/config/rmsnorm/rope/attention/feedforward/block/model/causal-lm/data/pretrain/sft/convert/final。
- 每分支的根文档（README.md、AGENTS.md）是**从 master 拷来的快照**（教程分支是孤儿，看不见 master 文件——与 lesson 10 的 tokenizer 拷贝同模式）；改动时每分支需单独同步。

## 2.1 分支图（含每分支累积验收数锚点）

```
01-skeleton (5) → 02-config (10) → 03-rmsnorm (15) → 04-rope (21) → 05-attention (27)
→ 06-feedforward (32) → 07-block (37) → 08-model (42) → 09-causal-lm (50) → 10-data (55)
→ 11-pretrain (59) → 12-sft (62) → 13-convert (67) → 14-final (70)
```

（锚点即 verify/14_e2e.py `floors` 字典；新增/删除检查必须同步。）

# 3. 验收机制（改动必读）

- 根 `verify.py`：静态入口，自动发现 `verify/NN_xxx.py`（按文件名排序），`register(name)` 装饰器注册 `CHECKS = [(name, fn)]`；传入参数对象带 `.fast`（`--fast` 跳过慢训练检查——在被跳过的检查函数里 `if getattr(args, "fast", False): return` 提前返回）。失败即 exit 非 0，输出 `== ... passed, N failed ==`。
- `verify/_common.py`：`TINY_CONFIG`（hidden 96/layers 2/vocab 512/…）、`set_seed`、`close`（allclose atol=rtol=1e-5）。
- 验收哲学：检查是"行为指纹"而非实现绑定——RoPE 保范、KV 增量==全量、清参恒等（残差）、CE gap（学到没）、1e-5 round-trip（转换无损）。训练检查用微型配置（hidden 96~128、2 层、seq 64~96）CPU 秒级到分钟级完成。
- 失败排查：`.venv/Scripts/python.exe verify.py 2>&1 | grep FAIL`；或 `importlib.import_module('verify.05_attention')` 单测（数字前缀无法点式导入）。
- 全链验收：`bash tools/check_all_branches.sh --fast`（14 分支依次 checkout 跑 verify，约 2 分钟；`--full` 约 10 分钟；`PY=` 可换解释器）。

# 4. 模块实现要点（minimind3/ 包，纯手写）

| 文件 | 内容 | 对照标准实现 |
|---|---|---|
| config.py | MiniMindConfig（含键值/派生尺寸、YaRN、estimate_parameter_count，model_type="minimind3" 避免与标准实现注册冲突） | model/model_minimind.py |
| rms_norm.py | RMSNorm（fp32 内部归一化后 type_as） | 同上 |
| rope.py | precompute_freqs_cis / apply_rotary_pos_emb（GPT-NeoX 双半表） | 同上 |
| attention.py | GQA + QK-Norm + RoPE + KV-Cache + 因果掩码 + SDPA 快路径 | 同上 |
| feed_forward.py | SwiGLU（gate/up/down，无 bias） | 同上 |
| block.py | MiniMindBlock（pre-norm 残差） | 同上 |
| model_body.py | MiniMindModel（embed+blocks+final norm+persistent=False RoPE buffer+KV 编排+aux_loss） | 同上 |
| causal_lm.py | MiniMindForCausalLM（权重绑定、shift 损失、自实现 generate 含 temperature/top_k/top_p/repetition_penalty/KV 增量） | 同上 |
| tokenizer_utils.py | 懒加载 AutoTokenizer、ChatML 渲染、generate_labels 掩码 | model/ 词表 + dataset/lm_dataset.py |
| datasets.py | PretrainDataset / SFTDataset（pad 标签 -100） | dataset/lm_dataset.py |
| train_utils.py | get_lr 余弦、setup_seed、checkpoint/权重存取 | trainer/trainer_utils.py |
| train_pretrain.py / train_full_sft.py | 训练循环（CLI 参数、累积、clip、resume、**随权重写 config sidecar**） | trainer/train_pretrain.py、train_full_sft.py |
| convert.py | pth⇄HF（sidecar 优先推断配置 + 自包含 remote-code 拼接） | scripts/convert_model.py |
| e2e.py | 全流程流水线（合成数据→预训练→SFT→转换→对话） | — |

# 5. 已知坑（维护时最容易踩，全部有教训记录）

- **Windows/GBK 控制台**打印中文乱码属正常，勿当 bug；decode 到含 U+010A(`\n`) 的文本 print 会 UnicodeEncodeError，测试输出用 ASCII 标记（如 `[E2E] Q:`/`loss X`）并由 verify 解析。
- **transformers>=4.45** 的 `PretrainedConfig.__init__` 会把 bos/eos/pad_token_id 当作显式参数覆盖子类默认值 → 子类必须 `kwargs.pop('bos_token_id')` 后显式传 `super().__init__(bos_token_id=…)`；minimind 自己的 MiniMindConfig 同样有此潜在 None（不是 bug）。
- **Windows 上必须先 import datasets 再 import torch**（顶部顺序），否则 pyarrow DLL 冲突。
- **repeat_kv 用 expand+reshape**（顺序 [kv0,kv0,kv1,…]），不是 `repeat`。
- attention：KV 拼接发生在 apply_rotary_pos_emb **之后**（历史 K 已旋转）；因果掩码只加在 `scores[..., -seq_len:]` 新列上。
- generate：top-p 需要 `mask[..., 1:] = mask[..., :-1].clone()` 移位；增量模式只喂 `input_ids[:, past_len:]`；权重绑定在 `post_init()` 之前做。
- 训练脚本坑：`save_interval=0` 时 `%` 运算的短路顺序（`args.save_interval > 0 and (...)` 放前面）；resume 跳过全部 batch 时 `step` 需预初始化、末尾残差梯度块要靠 `start_step + 1 <= step` 防越界；checkpoint 无条件每 epoch 尾保存、权重保存才受 save_interval 控制。
- 转换/加载坑：`convert.py` 的 standalone 拼接 **config.py 必须放 _SRC_FILES 首位**且文件头加 `from __future__ import annotations`；改模型后要重新生成 modeling_minimind3.py；transformers 会缓存 remote code 于 `~/.cache/huggingface/modules/transformers_modules/hf/`，改了建模文件需清缓存或改目录名；`.pth` 不含非张量超参（rope_theta 等）→ 权重旁必须写 `*.config.json` sidecar，infer 优先读它、兜底从 q_norm.weight 维度取 head_dim（gcd 兜底不可靠）。
- 文档快照坑：教程分支各自持有 AGENTS.md/README 快照，改了 master 的根文档**不会自动**进分支；每分支需单独同步（见 §8 闭环第 5 步）。
- `tools/check_all_branches.sh` 依赖 `sort -V` 排序分支名；默认 `--fast`，可用 `PY=python` 覆盖解释器；脚本结束会切回原分支。
- 本机 git 默认在 Windows 下提示 "LF will be replaced by CRLF" 属正常（core.autocrlf），不影响内容。

# 6. 收敛配置与实验事实（CPU 临界值，勿随意缩水）

- 12.2 探针/e2e 用 hidden 128 / layers 2 / SFT 4 epoch / lr 5e-4 / accum 1 / seq 96 → 内容级 CE 分离（好 0.93 vs 坏 1.37）；e2e 贪心生成可背出"猫是一种会抓老鼠的动物。""狗是人类忠诚的朋友。"；该 CE gap 正是"掩码 SFT 学到了"的机器判据。
- 默认 tiny 配置（hidden 96/2 层）预训练 1 epoch 可收敛；SFT 内容级需要 hidden 128（96 只在 loss 层收敛）。
- 真实 GPU 训练建议：预训练 lr 5e-4、SFT 1e-5~2e-5、更长 max_seq_len；教程仅演示原理。

# 7. 环境与资源

- 验收解释器：仓库根 `.venv/Scripts/python.exe`（torch 2.11+cpu、transformers 4.57.6、datasets 3.6.0、tokenizers 0.22.2）；系统默认 python **没有 torch**。产物目录（out/、checkpoints/、minimind3-hf/）均已 gitignore。
- **严禁引入网络下载**（数据集/权重）。数据用 `python -m tools.make_synthetic_data --out data --num 300` 生成（确定性、可复现），data/*.jsonl 是**已跟踪**文件（.gitignore 特意不管它）。
- 仓库根散落的 `dataset/*.jsonl`（约 3GB）是标准实现用的真实数据，**未跟踪**，与教程无关，别动别提交。
- 词表来源：`model/tokenizer.json` + `tokenizer_config.json` 仅在 master 上有（教程分支是孤儿无 model/），lesson 10 已用 `git show master:model/...` 拷贝进 `minimind3/tokenizer/`（已跟踪）。
- Git 身份 billpenn2005，origin = https://github.com/billpenn2005/minimind；发布 = `git push origin master` + `git push origin refs/heads/tutorial/*:refs/heads/tutorial/*`。

# 8. 维护闭环（每次改动后）

1. 小步验证：`.venv/Scripts/python.exe verify.py`（全量约 1.5~2.5 分钟；改代码快查可用 `--fast`）；
2. 若动了训练类检查 → 全量跑；若动了模型源码 → 重跑 lesson-13 相关检查并确认 standalone 产物更新；
3. 提交（先 verify 后 commit，勿链式 `verify; git commit`——失败会误提交）；课程分支变化 → 在分支上提交；框架/文档 → master；
4. 大改动后跑 `tools/check_all_branches.sh --fast`（全 14 分支约 2 分钟）；
5. 改动根文档（README / 本文件 / 课程 README）后：按分支图逐分支 `git show master:…` 拷贝同步（快照模式），并同步 00-DESIGN 已删除、锚点为 AGENTS.md；
6. 更新本文件已过时的事实（验收数字、floors、新坑）。

# 9. 常见任务速查

- **给某课加一个检查**：改 `verify/NN_x.py` 注册新 fn → 更新 14.3 floors → 全量 verify。
- **改课程文档**：改对应分支 `tutorial/NN-x/README.md` → 该分支 verify --fast → 提交 → 从该分支把新文档同步到后续分支（快照拷贝：`git show refs/heads/tutorial/NN:路径` 逐分支写回）。
- **重跑端到端**：`.venv/Scripts/python.exe -m minimind3.e2e`（~1 分钟，产物 out/e2e/）。
- **查看相邻分支差异**：`git diff refs/heads/tutorial/11-pretrain refs/heads/tutorial/12-sft`（= 第 12 课增量）；与标准实现对比：`git diff refs/heads/tutorial/14-final master -- minimind3/…`。
- **发布**：master + 全部 tutorial 分支 push（见 §7）。
- **磁盘占用**：`git count-objects -vH`（孤儿链每个分支持有全部历史→对象库会较大，正常）。

# 10. 进阶方向（超出本教程范围，供扩展）

MoE（config 已含字段）/ LoRA / DPO·GRPO·PPO / 思维链 / Qwen3 生态格式（minimind 的 convert_model.py 即此路线）/ GGUF 量化 / vLLM 部署；换大模型只需改 hidden/layers + 真实语料（dataset/*.jsonl，schema 完全兼容）+ GPU。