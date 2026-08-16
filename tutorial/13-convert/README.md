# 第 13 课 · HF transformers 格式转换（convert.py）

> 难度：★★★☆☆ ｜ 预计用时：2~3 小时 ｜ 前置：09 课（模型类）+ 12 课（训练产物）
> 学完本课你应能回答：**一个"可分发"的模型目录里有什么？auto_map 是什么？为什么 .pth 会"失忆"？**

---

## 0. 本课目标

- [ ] 理解 HF 模型目录格式（config.json / pytorch_model.bin / tokenizer）与加载解析路径；
- [ ] 理解两种路线：AutoClass 注册（minimind 路线） vs **自包含 remote-code**（本课新招：产物可脱离仓库运行）；
- [ ] 理解 `.pth` 的"失忆"问题与 **sidecar config** 解法；
- [ ] 手写 `minimind3/convert.py` 双向转换（torch ⇄ HF）；
- [ ] 验收 5 项新检查（13.1~13.5，累计 67 项全过）。

---

## 1. 理论

### 1.1 一个可分发的模型目录里有什么？

训练的 `.pth` 只有一堆数字，孤岛一个。业界分发标准（HF）是一个**目录**：

```
config.json            ← 模型蓝图（结构尺寸 + 超参数）
pytorch_model.bin      ← 权重（也可 safetensors）
tokenizer.json + tokenizer_config.json
modeling_minimind3.py  ←（自包含版）自定义结构的源码
```

`AutoModelForCausalLM.from_pretrained(dir)` 的解析流程：
1. 读 `config.json`（含 `model_type` / `auto_map`）；
2. 按它找/加载模型类；
3. 实例化 → 载入权重（`strict=True` 校验键名）；
4. 返回可推理的模型。

### 1.2 路线 A：AutoClass 注册（minimind 的做法）

```python
MiniMindConfig.register_for_auto_class()                       # auto_map 里写 AutoConfig
MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
model.save_pretrained(dir)    # 会把 auto_map 写进 config.json
```

之后**同一环境**里 `AutoModelForCausalLM.from_pretrained(dir, trust_remote_code=True)` 能加载。
局限：auto_map 指向"模块路径"，换个没有该模块的环境就死——**依赖仓库**。

### 1.3 路线 B：自包含 remote-code（本课新增的绝活）

把全部模型源码**按依赖顺序拼接**成一个 `modeling_minimind3.py` 放进产物目录，并改写 auto_map：

```json
{"auto_map": {"AutoConfig": "modeling_minimind3.MiniMindConfig",
              "AutoModelForCausalLM": "modeling_minimind3.MiniMindForCausalLM"}}
```

转换成了**目录级 remote code**：该目录拷到任何机器（哪怕没有本仓库），`trust_remote_code=True`
即可加载——**产物 = 可独立分发的成品**。拼接的两个关键：
1. 按依赖顺序排序源码片段（config 在最前），头部加 `from __future__ import annotations`
   （类体注解 `config: MiniMindConfig` 会立即求值，config 必须先定义）；
2. 去掉 `from .x import ...` 相对导入，让所有类落在同一模块。

### 1.4 ⚠️ 核心坑：`.pth` 会"失忆"

`.pth` 只含**张量**。像 `rope_theta`（1e6）、`max_position_embeddings`、`rms_norm_eps` 这些
**非张量超参**根本不进 state_dict！若转换时用默认值重建（比如 rope_theta 错成 1e4），
RoPE 频率全变 → 模型输出完全不同 → 13.3/13.5 的 logits 一致性（1e-5）立刻红叉。

**解法（最佳实践，本课已内置）**：训练脚本每次存权重时**同时写一个 sidecar `*.config.json`**
（11/12 课保存逻辑已改）；转换时**优先读 sidecar**，缺失时才从张量形状推断结构尺寸
（`q_norm.weight` 的维度 = head_dim，是最可靠的信号；gcd 兜底不可靠——gcd(96,48)=48 会让
4 头 2 KV 变成 2 头 1 KV）。

### 1.5 本课其它约定与细节

- `pad_token_id = 0` 只在**转换时**设置（不污染类默认值）：生态工具（padding/chat 模板）需要它；
- 反向转换 `convert_transformers2torch`：HF → `.pth`（float32），可与 11/12 课权重互换；
- `safe_serialization=False` → 写 `pytorch_model.bin`（标准实现同款；safetensors 列进阶）；
- transformers 会把 remote code **缓存**到 `~/.cache/huggingface/modules/transformers_modules/...`——
  改了建模文件必须清缓存或换目录名。

---

## 2. 手写任务清单

1. `infer_config_from_state_dict(sd, sidecar_path)`：sidecar 优先 + 形状推断兜底；
2. `convert_torch2transformers`：加载（strict 校验）→ 注册 AutoClass → save_pretrained + tokenizer；
3. `_write_standalone_modeling`：拼接源码 + 改写 auto_map（路线 B）；
4. `convert_transformers2torch` + `_load_from_hf`（反向）；
5. `__main__` 示例：`out/full_sft_96.pth → minimind3-hf/full_sft_96`；
6. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 3. 验收解读（verify/13_convert.py）

| 检查 | 验什么 |
|---|---|
| 13.1 | 从随机 .pth 推断的 config == 期望（96/2/512/4/2/128：hidden/layers/vocab/heads/kv/intermediate） |
| 13.2 | 产物文件齐全：config.json / pytorch_model.bin / tokenizer 两件 / modeling_minimind3.py；auto_map 指向**本地模块**；pad_token_id == 0 |
| 13.3 | **AutoModelForCausalLM 加载后 logits == 原模型（1e-5）** |
| 13.4 | **脱离仓库**：cwd 切到临时目录（仓库不在 sys.path）也能 remote-code 加载并生成 8 个 id |
| 13.5 | 反向转换 transformers2torch 后再 strict 加载，logits == 原（1e-5）→ **round-trip 闭环** |

> 13.3/13.5 是整个教程精度最高的断言：`1e-5` 意味着**任何一处权重/超参/前向逻辑不一致都会被抓住**。
> 当年开发时，sidecar 缺 rope_theta 的 bug 就是这样被 max diff 1.3e-1 定位的。

## 4. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/convert.py` | `scripts/convert_model.py` |

minimind 的转换目标更野：直接映射成 **Qwen3/Qwen3-MoE 原生类**（生态任意框架可加载），且有
LoRA 合并等。教程主路 = 自包含 remote-code（等价能力、零外部依赖）；Qwen3 生态映射列为进阶。

## 5. 常见坑（全部真实踩过）

- **sidecar 缺失** → 非默认 rope_theta 的模型转换后静默错乱（13.3 抓）；
- **拼接顺序错**：config 片段放最后 → 类体注解 NameError（依赖顺序 + `__future__ annotations` 双保险）；
- **`register_for_auto_class` 忘在 save 前调用**：config.json 没有 auto_map；
- **`from_pretrained` 忘 `trust_remote_code=True`**：直接报"找不到模型类"；
- **remote code 缓存残留**：改了建模文件后旧缓存坑你（删 `~/.cache/huggingface/modules/transformers_modules/hf/` 或换目录）；
- **strict=False 静默**：宁可 `assert not miss and not extra` 显式暴露键名差异。

## 6. 小结 & 下一课

- ✅ HF 目录 = config + 权重 + tokenizer（+可选的源码）；
- ✅ 路线 A 注册式（依赖环境） vs 路线 B 自包含式（可分发）；
- ✅ .pth 失忆 → sidecar config 拯救；
- ✅ 1e-5 round-trip = "转换无损"的机器保证。

**下一课（14-final）**：收官！把 13 课产物串成一条 **端到端流水线**（合成数据 → 微预训练 → SFT →
转换 → AutoModel 对话），检验"整个教程真的教会了一个模型说话"（内容级：它要能背出
"猫是一种会抓老鼠的动物。"）。再加一个**一键全链验收脚本**，10 分钟内跑完 14 个分支。