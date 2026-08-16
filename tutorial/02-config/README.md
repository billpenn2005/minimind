# 第 02 课 · 配置类 MiniMindConfig

> 难度：★☆☆☆☆ ｜ 预计用时：1~2 小时 ｜ 前置：第 01 课（知道"语言模型=预测下一个 token"即可）
> 学完本课你应能回答：**模型的"身材参数"从哪来？transformers 的 PretrainedConfig 是什么？为什么 bos/eos 会掉进一个经典坑？**

---

## 0. 本课目标

- [ ] 理解**超参数（hyperparameter）**概念和 MiniMind 的每个默认值含义；
- [ ] 理解"派生尺寸"：为什么 `head_dim`、`intermediate_size` 不用手填；
- [ ] 理解 `transformers.PretrainedConfig` 的职责（序列化、与 AutoConfig 集成）；
- [ ] 手写 `minimind3/config.py` 的 `MiniMindConfig`；
- [ ] 让 5 项新检查（02.1~02.5）全部 PASS（累计 10 项）。

---

## 1. 理论：什么是"配置"，为什么要先写它？

### 1.1 一个模型 = 蓝图（config）+ 参数（weights）

想象盖房子：**蓝图**决定"几层楼、几间房、多高的天花板"，**材料**（砖头水泥）对应参数。
训练前，我们必须先把蓝图定下来——因为参数张量的**形状**完全由蓝图决定。

训练代码里只有一种写法成立：

```
config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, ...)  # 蓝图
model  = MiniMindForCausalLM(config)                                # 按蓝图施工
```

配置类集中回答模型的"身材问题"：

| 超参数 | 含义 | MiniMind 默认值 | 直观理解 |
|---|---|---|---|
| `hidden_size` | 每个 token 的向量维度（d_model） | 768 | 每个词用 768 个数描述 |
| `num_hidden_layers` | Transformer 层数 | 8 | 8 层"加工车间" |
| `vocab_size` | 词表大小 | 6400 | 词典里 6400 个词条 |
| `num_attention_heads` | 注意力头数（Q 头） | 8 | 8 个"观察角度" |
| `num_key_value_heads` | KV 头数（GQA） | 4 | 4 个共享 KV 的观察角度（第 05 课详解） |
| `head_dim` | 每个头的维度 | hidden//heads = 96 | 每个观察角度的分辨率 |
| `intermediate_size` | 前馈层中间维度 | ceil(768·π/64)·64 = 2432 | "管道最粗处" |
| `max_position_embeddings` | 最大序列长度 | 32768 | 最多能处理多长的上下文 |
| `rope_theta` | RoPE 频率基数 | 1e6 | 位置编码的"波长"调节钮（04 课） |
| `rms_norm_eps` | 归一化防除零项 | 1e-6 | 安全垫 |
| `tie_word_embeddings` | 是否共享词嵌入与输出层 | True | 省约 490 万参数的技巧（09 课） |
| `dropout` | 随机丢弃比例 | 0.0 | 防止过拟合（小模型几乎不用） |

> 为什么默认值是这些数字？它们来自 minimind 标准实现的真实配置；第 08 课你亲手搭模型时会体会到
> 768×8 这个组合在"参数量/训练速度/效果"上的平衡（64M 参数，对标 MobileLLM 的研究结论：
> 小模型区间"深而窄"优于"宽而浅"，但 hidden<512 又会明显变差）。

### 1.2 派生尺寸：让机器替你算

有些尺寸**不是独立选择的**，而是由别的参数**规定**的：

- `head_dim = hidden_size // num_attention_heads`（768 // 8 = 96）：所有头的维度加起来恰好等于隐藏维度，要求 `hidden % heads == 0`；
- `intermediate_size`：minimind 特色的取整公式

$$
\text{intermediate} = \left\lceil \frac{\text{hidden} \times \pi}{64} \right\rceil \times 64
$$

（768 × π ÷ 64 ≈ 37.7 → 向上取整 38 → ×64 = 2432。）取 64 的倍数是为利用硬件对齐（内存对齐、矩阵乘加速）；乘 π 只是"选一个不是 2 的整数幂的数"，让中间维度与输入输出维度互素、信息不冗余——这是工程设计上的小心思，理解即可。

### 1.3 transformers 的 PretrainedConfig：为什么继承它？

我们的配置类继承 `transformers.PretrainedConfig`，让我们免费获得生态能力：

- **序列化**：`config.to_diff_dict()` 转 dict、`save_pretrained(dir)` 写 `config.json`、`from_pretrained(dir)` 读回——第 02/13 课验收都在用；
- **AutoConfig 集成**：第 13 课注册 `register_for_auto_class` 后，`AutoModelForCausalLM.from_pretrained` 能凭 config 自动重建模型；
- **插件字段**：`model_type` = `"minimind3"`（与标准实现的 `"minimind"` 区分，防止两个类同时 import 时注册冲突）。

### 1.4 ⚠️ 本课最大的坑：bos/eos 被父类"吃掉"

`PretrainedConfig.__init__` 把 `bos_token_id / eos_token_id / pad_token_id` 当作**显式命名参数**（默认 None）。
如果你在子类里先写了 `self.bos_token_id = kwargs.get("bos_token_id", 1)` 再把整个 kwargs 传给 `super().__init__(**kwargs)`，
transformers 会用它的命名参数（None！）**覆盖**你的设置——而且如果你真把 `bos_token_id` 留在 kwargs 里，会直接抛
`TypeError: got multiple values for keyword argument`。

正确姿势（就是本分支参考答案的做法）：

```python
# 既要取值、又要从 kwargs 里"弹走"，避免传给 super 时冲突：
self.bos_token_id = kwargs.pop("bos_token_id", 1)
self.eos_token_id = kwargs.pop("eos_token_id", 2)
super().__init__(**kwargs)   # bos/eos 已不在 kwargs 里，父类的显式参数不会覆盖你
```

> 冷知识：minimind 标准实现自己的 MiniMindConfig 同样有这个潜在问题（它的 `bos_token_id` 也常是 None，
> 训练时绕道用 tokenizer 的 bos）。本教程选择在这里"修复"它，并在文档里讲清楚——这正是"手写教程"比"抄代码"
> 多出来的价值。

---

## 2. 阅读参考答案（minimind3/config.py）

按顺序读三个部分：

1. **`__init__`**：全是 `self.xxx = kwargs.get("xxx", 默认值)`——"外部传了就听外部，没传就用默认"。注意 `kwargs.pop` 的两个 token。
2. **`rope_scaling` 派生**：`inference_rope_scaling=True` 时生成 YaRN 缩放字典（factor=16、beta 参数……），否则为 None。
   这是"把可选能力做成 config 字段"的经典模式：平时不占内存，需要时一键开启（04 课 RoPE 会消费它）。
3. **`estimate_parameter_count`**（教程扩展，minimind 没有）：手算参数量公式：

$$
\text{params} = \text{vocab} \times h + \text{layers} \times (4 h^2 + 3 h \cdot \text{int} + \dots) - \mathbb{1}[\text{tie}] \cdot \text{vocab} \times h
$$

（`tie_word_embeddings=True` 时 lm_head 复用 embed 表，省掉 `vocab × h`。）验收 02.4 会逐项核对这个公式。

---

## 3. 手写任务清单

1. 确定 8 个核心默认值（对照 §1.1 表格，务必与 minimind 一致：hidden 768 / layers 8 / vocab 6400 / heads 8 / kv 4）；
2. 实现派生尺寸：`head_dim`、`intermediate_size`（用 `math.ceil` + π）；
3. 处理特殊 token：`kwargs.pop` 之后再把剩余 kwargs 传给 `super().__init__`（剩余 kwargs 是未知字段，父类会妥善吸收，保证巨量已存 config.json 兼容）；
4. 支持可选字段：`rope_scaling`、`use_moe`（MoE 相关字段留好，本教程只用 Dense，字段存在即可）；
5. 实现 `estimate_parameter_count()`。
6. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 4. 验收解读（verify/02_config.py）

| 检查 | 验什么 | 怎么验 |
|---|---|---|
| 02.1 | 默认值与 minimind 完全一致（含 intermediate==2432、head_dim==96） | 直接比较属性 |
| 02.2 | 微型配置能正确派生（TINY_CONFIG 96/4 头 → head_dim 24） | 属性计算 |
| 02.3 | 序列化往返：`to_diff_dict` → `from_dict` 属性相等；`save_pretrained` → `PretrainedConfig.from_pretrained` 且 `model_type == "minimind3"` | 落盘重读 |
| 02.4 | `estimate_parameter_count` 公式正确（tie=False 时多加 vocab×h） | 数值对照 |
| 02.5 | `inference_rope_scaling=True` 时生成合法 YaRN dict（必备键齐全） | 键集合 |

---

## 5. 对照标准实现

| minimind3 | minimind 标准实现（master） |
|---|---|
| `minimind3/config.py` → `MiniMindConfig` | `model/model_minimind.py` → `MiniMindConfig` |

差异点（本教程设计）：`model_type="minimind3"`；`head_dim` 显式化；修复 bos/eos 默认值；
新增 `estimate_parameter_count()` 教学扩展；MoE 字段保留但默认 `use_moe=False`。

## 6. 常见坑

- **忘记了 `% heads == 0`**：head_dim 整除失败会在第 05 课 attention 里炸出奇怪的形状错误——配置课就应抱住约束；
- **bos/eos 留在 kwargs 里**：`TypeError: got multiple values...`（直接用 `pop` 解决）；
- **`super().__init__` 位置**：必须放在**所有** `self.xxx` 赋值之后，且此时 kwargs 已清空未知项；
- **忘记 `import math`**：`math.ceil` 会 NameError；
- **派生尺寸写在赋值顺序前面**：`head_dim` 计算依赖 `hidden_size`，务必先赋 `hidden_size`。

## 7. 小结 & 下一课

- ✅ 蓝图（config）先行：参数形状全由它决定；
- ✅ 派生尺寸与 minimind 特色取整规则；
- ✅ transformers 生态（PretrainedConfig）与 AutoConfig 的接入方式；
- ✅ bos/eos 经典坑与 `kwargs.pop` 解法。

**下一课（03-rmsnorm）**：第一个真正的"计算零件"——RMSNorm。你会学到"为什么要归一化"，
以及为什么现代 LLM 都抛弃了经典的 LayerNorm（均值中心化）而改用 RMSNorm。