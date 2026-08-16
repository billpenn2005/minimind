# 第 13 课 · pth ⇄ HF transformers 格式转换

> [上一课 12-sft](../12-sft/README.md) ← [目录](../../README.md) → [下一课 14-final](../14-final/README.md)
>
> 难度：★★★☆☆ ｜ 预计用时 2~3 小时 ｜ 前置：09+12 课

## 0. 本课目标

- [ ] 理解 HF 模型目录格式、auto_map/remote-code、`.pth 失忆` 与 sidecar 解法；
- [ ] 手写 `minimind3/convert.py` 双向转换（torch ⇄ HF）；
- [ ] 验收 68 项累计（本课新增 13.1~13.5；13.3/13.5 是 1e-5 级高精度断言）。

## 1. 理论速览

- **可分发目录**：`config.json`（蓝图）+ `pytorch_model.bin`（权重）+ tokenizer 两件 +（自包含版）`modeling_minimind3.py`。
- **加载解析**：`AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` 读 config（`model_type`/`auto_map`）→ 定位类 → 实例化 → strict 载权重。
- **路线 A（AutoClass 注册）**：`MiniMindConfig.register_for_auto_class()` + `MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")` → save 时写入 auto_map；局限：依赖环境/仓库。
- **路线 B（自包含 remote-code，教程绝活）**：按依赖序拼接全部模型源码到 `modeling_minimind3.py`，auto_map 指向本地模块 → 产物目录拷到任何机器都能独立加载。两个关键：**config.py 放首位 + 头部 `from __future__ import annotations`**；去掉 `from .x import` 相对导入。
- **⚠️ `.pth 失忆`**：`.pth` 只存张量；`rope_theta/max_position_embeddings` 等非张量超参不在里面 → 转换时用错默认值（rope_theta 1e6 错成 1e4）RoPE 频率全变 → 1e-5 一致性崩。**解法：sidecar `*.config.json`**（11/12 课 `save_weights(config=…)` 已自动写），转换优先读它；兜底从 `q_norm.weight` 维度读 head_dim（gcd 兜底不可靠：gcd(96,48)=48 会把 4头/2KV 猜成 2头/1KV）。
- **其它约定**：`pad_token_id=0` 只在转换时设置；`safe_serialization=False`（写 bin）；改建模文件后要清 `~/.cache/huggingface/modules/transformers_modules/hf/`。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/convert.py`

**模块级**：`import json, os, re, sys`；`from pathlib import Path`；`import torch`；
- `_REPO_ROOT: Path` = `Path(__file__).resolve().parent.parent`；
- `_SRC_FILES: list[str]` = `["config.py","rms_norm.py","rope.py","attention.py","feed_forward.py","block.py","model_body.py","causal_lm.py"]`（**顺序不可改**）。

### 2.2 函数与流程

**`infer_config_from_state_dict(state_dict: dict, sidecar_path=None) -> MiniMindConfig`**：
- sidecar 存在 → `MiniMindConfig(**json.load(sidecar))`；
- 否则由张量形状推断：vocab/hidden 取 `model.embed_tokens.weight.shape`；层数 `max(…q_proj.weight)`+1；`head_dim = q_norm.weight.shape[0]`（无 q_norm 时 gcd 兜底）；`heads = hidden//head_dim`；`kv = k_dim//head_dim`；intermediate 取 `gate_proj.weight.shape[0]`。

**`_gcd(a: int, b: int) -> int`**：欧几里得。

**`convert_torch2transformers(torch_path, transformers_path, dtype=torch.float32, standalone=True) -> str`**：
1. `state_dict = torch.load(torch_path, map_location="cpu")`；`sidecar = torch_path.replace(".pth", ".config.json")`；
2. `lm_config = infer_config_from_state_dict(state_dict, sidecar)`；`lm_config.pad_token_id = 0`；
3. 注册 AutoClass 两条；`model = MiniMindForCausalLM(lm_config)`；`miss, extra = load_state_dict(..., strict=False)`；**`assert not miss and not extra`**（键名必须全对上）；
4. `.to(dtype)` → `save_pretrained(path, safe_serialization=False)` + `load_tokenizer().save_pretrained(path)`；
5. `standalone=True` 时调 `_write_standalone_modeling`；print `[convert] …`；返回路径。

**`_write_standalone_modeling(out_dir: str) -> None`**：
- 头部写 `from __future__ import annotations` + 说明注释；按 `_SRC_FILES` 逐个读 `minimind3/<name>` 源码，`re.sub(r"^from \.\w+ import .*$", "", src, flags=re.M)` 去掉包内相对导入，拼成 `modeling_minimind3.py`；
- 改 `config.json`：`cfg["auto_map"] = {"AutoConfig": "modeling_minimind3.MiniMindConfig", "AutoModelForCausalLM": "modeling_minimind3.MiniMindForCausalLM"}`。

**`_load_json(path) -> dict`**、**`_dump_json(path, data) -> None`**：json 读/写（utf-8、indent=2）。

**`convert_transformers2torch(transformers_path, torch_path) -> str`**：`_load_from_hf` → `torch.save({k: v.float().cpu()…})`。

**`_load_from_hf(transformers_path)`**：`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`。

**`__main__` 示例**：`convert_torch2transformers("out/full_sft_96.pth", "minimind3-hf/full_sft_96")` 后加载打印类型。

## 3. 手写步骤

实现 §2.2 → `.venv/Scripts/python.exe verify.py 13`（13.3/13.5 含 subprocess 远端加载，约 30~60s）。

## 4. 验收解读（verify/13_convert.py）

| 检查（新） | 验什么 |
|---|---|
| 13.1 | 随机 `.pth`（含 sidecar）推断 config == 期望（96/2/512/4/2/128） |
| 13.2 | 产物文件齐全（config/bin/tokenizer×2/modeling_minimind3.py）；auto_map 指向**本地模块**；pad=0 |
| 13.3 | **AutoModel 加载后 logits == 原模型（1e-5）** |
| 13.4 | **脱离仓库**（cwd=临时目录、仓库不在 sys.path）也能 remote-code 加载并生成 8 个 id |
| 13.5 | 反向转换后 strict 重载，logits == 原（1e-5）→ round-trip 闭环 |

> 1e-5 意味着任何权重/超参/前向不一致都会被抓住（开发期 max diff 1.3e-1 即 sidecar 缺 rope_theta 的现场）。

## 5. 参考答案

`answers/minimind3/convert.py`（先写后对；重点核对 sidecar 优先与 `_SRC_FILES` 顺序）。

## 6. 常见坑

- **sidecar 缺失**：非默认 rope_theta 模型转换后静默错乱（13.3 抓）；
- **拼接顺序错**：config 放最后 → 类体注解 NameError（顺序 + `__future__ annotations` 双保险）；
- **register_for_auto_class 忘在 save 前**：config.json 无 auto_map；
- **忘 `trust_remote_code=True`**：报"找不到模型类"；
- **remote code 缓存残留**：`~/.cache/huggingface/modules/transformers_modules/hf/` 清缓存或换目录名；
- **`strict=False` 静默**：用 assert 显式暴露 miss/extra。

## 7. 对照标准实现

| 你手写 | minimind 标准 |
|---|---|
| `minimind3/convert.py` | `scripts/convert_model.py` |

minimind 的目标更野：映射成 **Qwen3/Qwen3-MoE 原生类**（生态任意框架可加载，含 LoRA 合并）；教程主路 = 自包含 remote-code（等价能力、零外部依赖）；Qwen3 生态映射列为进阶。

## 8. 小结

✅ 双向转换 + 1e-5 round-trip 保证 + 可独立分发的产物（含 sidecar 对治 `.pth 失忆`）。
**下一课（收官）**：端到端流水线——从空权重到能背出事实的模型，一键跑完，毕业验收！