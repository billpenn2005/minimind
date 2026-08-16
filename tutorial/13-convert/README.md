# 第 13 课 · HF transformers 格式转换

## 目标

手写 `minimind3/convert.py`：`pth ⇄ HF 目录` 双向转换，
产出可被 `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` 加载的模型目录。

## 理论

### 1. HF 模型目录是什么

一个"可分发"的 transformers 模型 = 一个目录：

```
config.json            # 模型蓝图（目标：加载时能凭它重建结构）
pytorch_model.bin      # 权重（也可 safetensors）
tokenizer.json / tokenizer_config.json
modeling_minimind3.py  #（自包含版）自定义结构源码
```

`AutoModelForCausalLM.from_pretrained(dir)` 的解析路径：
读 config.json → 按 `model_type`/`auto_map` 找模型类 → 实例化 → 载权重。

### 2. 两种实现路线

| 路线 | 机制 | 适用 |
|---|---|---|
| AutoClass 注册（minimind 路线） | `register_for_auto_class()` 写入 `auto_map: {"AutoConfig": "minimind3.config.MiniMindConfig", ...}` | 同仓库/同环境 |
| 自包含 remote-code（本课新增） | 把全部模型源码生成为目录内 `modeling_minimind3.py`，auto_map 指向它 | 拷贝到任何机器 |

自包含版的价值：**转换产物 = 可独立分发的成品**，不依赖本仓库代码。
实现技巧：按依赖顺序拼接各模块源码、去掉相对导入、头部加
`from __future__ import annotations`（类体注解先定义 config）、config 片段放最前。

### 3. 配置的"自描述"问题（本课最大的坑）

`.pth` 权重只含张量，**不含 `rope_theta`、`max_position_embeddings` 等"非张量"超参**。
若转换时用默认值重建（rope_theta 1e4 vs 1e6）→ RoPE 频率不同 → logits 对不上
（验收 13.3/13.5 的 logits 一致性会精确到 1e-5 级别抓住它）。

解法（最佳实践）：训练脚本每次保存权重时**同时写 sidecar `*.config.json`**；
转换优先读 sidecar，缺失时才从张量形状推断（`q_norm.weight` 维度 = head_dim
是最可靠信号）。

## 任务清单（先手写再看答案）

1. `infer_config_from_state_dict(sd, sidecar_path)`：sidecar 优先 + 张量推断兜底；
2. `convert_torch2transformers`：加载/对齐/注册/保存（config.json + bin + tokenizer）；
3. `_write_standalone_modeling`：源码拼接生成远程代码文件 + 改写 auto_map；
4. `convert_transformers2torch`：反向转换。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：张量推断、产物文件齐全（含 auto_map 指向本地模块）、
AutoModel 加载 logits 与原始模型一致、**脱离仓库目录可加载并生成**、反向 round-trip。

## 对照标准实现

`master:scripts/convert_model.py`（minimind 的 torch→transformers：注册 + Qwen3 生态兼容
双路线；教程实现自包含 remote-code 版；Qwen3 生态转换留作进阶）。

## 常见坑

- 忘了 sidecar → 非默认 `rope_theta` 训练出的模型转换后 logits 对不上；
- remote-code 拼接时**类型注解会立即求值**（`config: MiniMindConfig`），config 必须在最前；
- `register_for_auto_class` 写在保存前，`from_pretrained` 要 `trust_remote_code=True`；
- `pad_token_id` 不设会缺省为 None，生态工具（padding/chat）会抱怨；
- transformers 会把 remote code 缓存到 `~/.cache/huggingface/modules/...`，
  改了建模文件后需清缓存或换目录名。