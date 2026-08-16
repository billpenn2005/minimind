# 第 02 课 · MiniMindConfig：模型的"蓝图"

## 目标

手写 `minimind3/config.py` 中的 `MiniMindConfig`（继承 `transformers.PretrainedConfig`），
与 minimind 标准实现的默认值完全对齐，并理解每个超参数的含义与推导规则。

## 理论

### 1. 为什么需要 Config 类

一个模型有几十个超参数。把它们集中在一个可序列化的对象里：
- 一处定义、处处引用（模型结构、训练脚本、转换脚本共享同一份蓝图）；
- 随权重一起保存（HF 的 `config.json`），保证"权重 + 配置"自洽；
- `PretrainedConfig` 提供 `save_pretrained / from_pretrained / to_dict / from_dict`，
  并支持 `AutoConfig` 注册与 `trust_remote_code` 加载。

### 2. 关键参数（minimind 取值）

| 参数 | 默认 | 说明 |
|---|---|---|
| hidden_size | 768 | d_model，所有子层的工作维度 |
| num_hidden_layers | 8 | Transformer 解码块数量 |
| num_attention_heads / num_key_value_heads | 8 / 4 | MHA→GQA，KV 头减半省显存 |
| head_dim | hidden // heads = 96 | 每头维度，显式化便于 SDPA |
| intermediate_size | ceil(768·π/64)·64=2432 | minimind 特色取整（接近 3.17·d_model，与 LLaMA 的 3·d_model 同量级） |
| max_position_embeddings | 32768 | RoPE 预计算表长度 |
| rms_norm_eps | 1e-6 | 防除零 |
| rope_theta | 1e6 | RoPE 基频（越大低频越密，利于长文本） |
| tie_word_embeddings | True | Embedding 与 lm_head 共享权重 |
| vocab_size | 6400 | 跟随仓库自训练 BPE 词表 |

### 3. 派生参数

- `head_dim = hidden_size // num_attention_heads`
- `intermediate_size`：`math.ceil(hidden_size * pi / 64) * 64`（让 MLP 中间维度是 64 的倍数）
- `rope_scaling`：由 `inference_rope_scaling` 开关派生（YaRN 长上下文方案，本教程默认关闭）

## 任务清单（先手写再看答案）

1. 继承 `PretrainedConfig`，`model_type = "minimind3"`；
2. 实现上面"关键参数"全部字段的默认值（与 minimind 一致）；
3. 实现两个派生参数的计算；
4. 调用 `super().__init__(**kwargs)` 收尾；
5. （扩展）实现 `estimate_parameter_count()`，按结构公式估算参数量。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：默认值一致（含 `intermediate_size==2432`）、微型配置推导、
JSON round-trip、参数估算公式、YaRN 开关。

## 对照标准实现

`master:model/model_minimind.py → MiniMindConfig`。
差异说明：`model_type` 改为 `"minimind3"`（避免两套类在同一进程内注册冲突）；
新增 `estimate_parameter_count()`（教学扩展）；其余字段与默认值逐一对应。

## 常见坑

- `super().__init__(**kwargs)` 必须放在全部属性赋值之后，且把未知 kwargs 传下去，
  否则 `from_pretrained` 会报未知字段错误；
- 派生参数要**在赋值时计算**，不要先存再算，避免多构造路径不一致；
- 别忘 `num_key_value_heads` 默认是 4（GQA），不是 8。