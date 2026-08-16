# 第 02 课 · 配置类 MiniMindConfig

> [上一课 01-skeleton](../01-skeleton/README.md) ← [目录](../../README.md) → [下一课 03-rmsnorm](../03-rmsnorm/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 1~2 小时 ｜ 前置：01 课

## 0. 本课目标

- [ ] 理解"模型蓝图"：所有尺寸/超参集中一处，任何模块从 config 取数；
- [ ] 理解派生尺寸（head_dim、intermediate_size）的计算；
- [ ] 手写 `minimind3/config.py` 的 `MiniMindConfig`；
- [ ] 验收 11 项累计（本课新增 02.1~02.5）。

## 1. 理论速览

- **为什么需要 config**：模型结构（层数/维度/头数/词表）散落各处会失控；把一切收进一个类，序列化即 `config.json`（第 13 课转换要读它）；
- **为什么继承 `transformers.PretrainedConfig`**：白送 `to_dict()` / 序列化 / `from_pretrained` / `**kwargs` 容错，让模型类能接入 HF 生态；
- **派生尺寸**：
  - `head_dim = hidden_size // num_attention_heads`（每头维度）；
  - `intermediate_size = ceil(hidden_size × π / 64) × 64`：minimind 的"π 取整"惯例，让 MLP 宽度≈3.2×hidden，且 64 对齐。
- **⚠️ transformers 新版的坑**：`PretrainedConfig.__init__` 把 `bos_token_id`/`eos_token_id` 当**显式命名参数**（默认 None）——子类里先设置的值会被覆盖。解法：用 `kwargs.pop` 取值并**显式传参**给 `super().__init__(bos_token_id=…, eos_token_id=…)`。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/config.py`

**模块级**：import `math` 与 `from transformers import PretrainedConfig`。

### 2.2 `class MiniMindConfig(PretrainedConfig)`

| 成员 | 类型 | 值/默认 | 说明 |
|---|---|---|---|
| `model_type`（类属性） | `str` | `"minimind3"` | **必须**，避免与 minimind 的 `"minimind"` 注册冲突（两库同进程时） |

**`__init__(self, **kwargs)` 中设置的实例属性（全部经 `kwargs.get(key, 默认值)` 取得）：**

| 属性名 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `hidden_size` | `int` | `768` | d_model，embed 维度 |
| `num_hidden_layers` | `int` | `8` | 解码块数 |
| `dropout` | `float` | `0.0` | 训练随机失活率 |
| `vocab_size` | `int` | `6400` | 词表大小（对齐 minimind 词表） |
| `bos_token_id` | `int` | `1` | **必须 `kwargs.pop`** 后显式传给 super |
| `eos_token_id` | `int` | `2` | 同上 |
| `flash_attn` | `bool` | `True` | 是否走 SDPA 快路径（第 05 课用） |
| `num_attention_heads` | `int` | `8` | Q 头数 |
| `num_key_value_heads` | `int` | `4` | KV 头数（GQA） |
| `head_dim` | `int` | `hidden_size // num_attention_heads` | 每头维度（显式化，便于 flash 兼容） |
| `max_position_embeddings` | `int` | `32768` | RoPE 表长度上限（第 04/08 课用） |
| `hidden_act` | `str` | `"silu"` | MLP 激活名（第 06 课经 ACT2FN 查询） |
| `intermediate_size` | `int` | `math.ceil(hidden_size * math.pi / 64) * 64` | MLP 中间维度 |
| `rms_norm_eps` | `float` | `1e-6` | 归一化稳定项（第 03 课用） |
| `rope_theta` | `float` | `1e6` | RoPE 频率基数（第 04 课用） |
| `tie_word_embeddings` | `bool` | `True` | embed 与 lm_head 权重绑定（第 09 课用） |
| `inference_rope_scaling` | `bool` | `False` | 是否启用 YaRN 长上下文 |
| `rope_scaling` | `dict \| None` | YaRN dict 或 `None` | `True` 时 = `{"type":"yarn","beta_fast":32,"beta_slow":1,"factor":16,"original_max_position_embeddings":2048,"attention_factor":1.0}` |
| `use_moe` | `bool` | `False` | MoE 开关（本教程不用，保留字段对齐 minimind） |
| `num_experts` | `int` | `4` | MoE 专家数 |
| `num_experts_per_tok` | `int` | `1` | 每 token 激活专家数 |
| `moe_intermediate_size` | `int` | `= intermediate_size` | MoE 专家宽度 |
| `norm_topk_prob` | `bool` | `True` | MoE 路由概率归一化 |
| `router_aux_loss_coef` | `float` | `5e-4` | MoE 辅助损失系数 |

**`__init__` 结尾必须**：`super().__init__(bos_token_id=self.bos_token_id, eos_token_id=self.eos_token_id, **kwargs)`。

### 2.3 方法 `estimate_parameter_count(self) -> int`

按结构公式估算参数量（教学用）：

```
每层 = 注意力 + MLP
attn = hidden*heads*head_dim          # q
     + 2 * hidden*kv_heads*head_dim   # k,v
     + heads*head_dim*hidden          # o
mlp  = 2*hidden*intermediate          # gate,up
     + intermediate*hidden            # down
emb  = vocab_size * hidden
head = 0 若 tie_word_embeddings 否则 vocab_size*hidden
返回 per_layer*num_hidden_layers + emb + head
```

| 要求 | 约束 |
|---|---|
| 返回类型 | `int` |
| 亲测值 | 默认配置 ≈ 64,790,400（教学断言用公式即可，不强求具体数） |

## 3. 手写步骤

1. 抄规格到 `minimind3/config.py`（属性顺序无所谓，**默认值必须一致**）；
2. 特别注意 `bos/eos` 的 pop + 显式传参；
3. `verify.py 02`。

## 4. 验收解读（verify/02_config.py）

| 检查（新） | 验什么 |
|---|---|
| 02.1 | `MiniMindConfig()` 默认值全套正确（hidden 768/layers 8/vocab 6400/heads 8/kv 4/rope_theta 1e6/…） |
| 02.2 | 派生尺寸：TINY_CONFIG 下 `head_dim = hidden//heads`、`intermediate_size = ceil(h·π/64)·64` |
| 02.3 | **bos/eos 不被覆盖**：`config.bos_token_id == 1`、`eos_token_id == 2`（专门抓 transformers 新版坑） |
| 02.4 | `model_type == "minimind3"`；`to_dict()` 可序列化 |
| 02.5 | `estimate_parameter_count()` 返回正 int 且公式自洽 |

## 5. 参考答案

`answers/minimind3/config.py`（先写后对；逐行比默认值/派生式/pop 顺序）。

## 6. 常见坑

- **bos/eos 被覆盖成 None**：症状 `config.bos_token_id is None` → 用 `kwargs.pop` + 显式传参；
- **`intermediate_size` 直接写死 2432**：规格要求用公式（`hidden=768→2432`），换 hidden 要能自动变；
- **忘了 `model_type`**：第 13 课 AutoClass 注册时与 minimind 冲突；
- **传参漏 `**kwargs`**：多余未知键会报 TypeError。

## 7. 对照标准实现

| 你手写 | minimind 标准实现 |
|---|---|
| `minimind3/config.py` | `model/model_minimind.py` 的 `MiniMindConfig` |

差异：本教程显式化 `head_dim`、新增 `estimate_parameter_count()`、`model_type` 改名避免冲突；其余默认值逐项对齐（hidden 768 / layers 8 / heads 8 / kv 4 / rope_theta 1e6 / max_pos 32768 / tie True…）。

## 8. 小结

✅ 蓝图类完成，默认值与 minimind 逐项对齐；✅ 学会了"派生尺寸"与"transformers 参数覆盖"两个关键机制。
**下一课**：`RMSNorm`——第一个真正做数学的模块。