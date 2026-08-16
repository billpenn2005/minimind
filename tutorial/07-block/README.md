# 第 07 课 · MiniMindBlock：Pre-Norm 残差解码块

## 目标

手写 `minimind3/block.py` 的 `MiniMindBlock`，把第 05/06 课的 Attention 与 MLP
组装成可堆叠的 Transformer 解码块。

## 理论

### 1. 残差连接（Residual / Skip Connection）

$$h' = h + f(h)$$

让梯度 $\partial h'/\partial h = 1 + \partial f/\partial h$：即使 $\partial f/\partial h$ 很小，
梯度也能"抄近道"直达早期层，这是深层网络不消失的关键。verify 07.2 的检查法：
把块内参数全部清零，`f` 输出必为 0，此时必须 `block(x) == x`。

### 2. Pre-Norm vs Post-Norm

- Post-Norm（原始 Transformer）：`h' = LN(h + f(h))`——梯度要穿过 LN，训练不稳；
- Pre-Norm（现代 LLM）：`h' = h + f(LN(h))`——残差路径恒等、干净。

minimind 采用 Pre-Norm，且 LM 头前的**最终归一化单独放在主体模型末尾**。

### 3. 块的组装顺序

```
h  --LN1--> Attention --+--> --LN2--> MLP --+--> h'
└───────────────────────┘  └────────────────┘
```

两个 RMSNorm 都是 `hidden_size` 维；`position_embeddings` 与 `past_key_value` 透传给 Attention。

## 任务清单（先手写再看答案）

1. `MiniMindBlock(layer_id, config)`：self_attn / input_layernorm / post_attention_layernorm / mlp；
2. `forward`：残差两个加法，返回 `(out, present_kv)`；
3. 保持子模块命名与 minimind 一致（权重转换依赖）。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：形状与命名、**残差恒等**、Pre-Norm 手工展开一致、梯度回传、cache 透传。

## 对照标准实现

`master:model/model_minimind.py → MiniMindBlock`。

## 常见坑

- 别在残差加法里忘掉 `residual` 变量（先存后加）；
- 两个 LN 的 `eps` 用 `config.rms_norm_eps`（1e-6），不是 RMSNorm 默认 1e-5；
- `use_cache=False` 时 `present_key_value` 为 None，要原样透传（verify 07.5）。