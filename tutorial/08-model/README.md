# 第 08 课 · 模型主体 MiniMindModel

> 难度：★★★☆☆ ｜ 预计用时：2~3 小时 ｜ 前置：07 课残差块
> 学完本课你应能回答：**embedding 表怎么查？RoPE 表为什么是"隐形"的 buffer？KV-Cache 的编排层在统筹什么？**

---

## 0. 本课目标

- [ ] 理解 Embedding（词表查找表）与 `nn.Embedding` 的用法；
- [ ] 理解 `Parameter` vs `buffer`（尤其 `persistent=False` 的妙用）；
- [ ] 手写 `minimind3/model_body.py` 的 `MiniMindModel`；
- [ ] 验收 5 项新检查（08.1~08.5，累计 42 项全过）。

---

## 1. 理论

### 1.1 Embedding 表：从 token id 到向量

语言模型收到的输入是 token id（整数，如 1968）。`nn.Embedding(vocab_size, hidden_size)` 维护一张
`[6400, 768]` 的查找表：`embed(input_ids)` = 按 id 取行，得到 `(B, S, 768)` 的向量序列。
**训练时这张表也是可学习参数**——模型学到的词义就分布在表里（注意：词义表征是全局学习的，不是查字典）。

### 1.2 主体模型的骨架

```python
class MiniMindModel(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)   # 最终归一化
        # RoPE 频率表：注册为"非持久 buffer"
        freqs = precompute_freqs_cis(...)
        self.register_buffer("freqs_cos", freqs.cos(), persistent=False)
        self.register_buffer("freqs_sin", freqs.sin(), persistent=False)
```

### 1.3 `Parameter` vs `buffer` vs `persistent=False`

| 类型 | 存什么 | 会被优化器更新？ | 会进 state_dict（保存）？ |
|---|---|---|---|
| `nn.Parameter` | 可学习权重（如 W_q） | ✅ | ✅ |
| 普通 buffer（`persistent=True`） | 运行时的"状态"（如 BN 均值） | ❌ | ✅ |
| **非持久 buffer**（`persistent=False`） | **只算一次的常量**（如 RoPE 表） | ❌ | **❌ 不保存** |

RoPE 表是"由 config 算出、永远不变的常量"——**保存它是纯浪费磁盘**（23MB 的 32768×768 float32 表！）。
每次构造模型时现场重算即可。`persistent=False` = "带着它跑，但不存档"。

### 1.4 forward 与 KV-Cache 编排

```
embed_tokens(x) → dropout → 8×MiniMindBlock（逐层透传 KV）→ norm → (hidden, presents, aux_loss)
```

- **`start_pos`**＝若传入 `past_key_values`，从第一层缓存的长度推得当前新 token 的起点；
  用 `freqs_cos[start_pos : start_pos + seq_len]` 对**新 token** 切片——
  位置编号连续，这就是 KV-Cache 与 RoPE 的衔接点；
- **单轮无缓存**（训练/预填充）：`start_pos=0`，整段切片；
- **逐层 KV 收集**：每层返回自己的 `present_kv`，汇总成 `presents` 元组（长度 = 层数）；
- **`aux_loss`**：MoE 负载平衡损失的汇总位（本教程 Dense 恒为 0，字段保留以对齐接口）；
- **meta 设备防御**：`meta` 设备（未实例化参数）时重算 buffer，避免从预训练 ckpt 加载结构时无法建参数。

### 1.5 为什么最终的 Norm 是"裸的"？

输出 `norm` 没有残差——它是模型最后一道"整理"，让表示回到温和尺度后再交给 lm_head（09 课）
做词表打分。这是 Llama/Qwen 系的标准收尾。

---

## 2. 手写任务清单

1. `embed_tokens` / `dropout` / `layers`（ModuleList）/ `norm`；
2. RoPE 表：`precompute_freqs_cis` + `cos()`/`sin()` 注册为 `persistent=False` buffer；
3. `forward(input_ids, position_ids=None, past_key_values=None, use_cache=False, output_attentions=False)`：
   - embedding → dropout；
   - `start_pos` 派生（从 past KV 长度）；
   - 逐层 forward，收集 `presents` 与 `aux_loss`；
   - 最终 norm；返回 `(hidden_states, presents, aux_loss)`；
4. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 3. 验收解读（verify/08_model.py）

| 检查 | 验什么 | 直觉 |
|---|---|---|
| 08.1 | 输出形状 `(B,S,hidden)`、dtype float32 | 基本契约 |
| 08.2 | `freqs_cos/sin` **不在 state_dict 里** | persistent=False 生效 |
| 08.3 | **模型级增量 KV == 全量**（分 1+2+3 段喂 vs 一次喂完，1e-4） | 编排正确（跨层一致） |
| 08.4 | 每层 past 形状 `(B, kv_heads, S, head_dim)`、层数与 config 一致 | 缓存结构 |
| 08.5 | 一次 forward 后**所有参数**都有梯度 | 全链路反传连通 |

> 08.3 是"缓存编排层"的终极考试：单块正确（05.4）不代表 8 块衔接正确——这里的增量推理必须与
> 朴素全量逐 token 一致，任何 start_pos 或拼接错误都会让结果分叉。

## 4. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/model_body.py` → `MiniMindModel` | `model/model_minimind.py` → `MiniMindModel` |

延续逐模块对齐；辅助类 `MoeModelOutput`/`ModelOutput` 直接复用 transformers 的 `ModelOutput`
（本课是 tutorial 首次引出 transformers 生态类，第 09/13 课会更深依赖）。

## 5. 常见坑

- **buffer 忘了 `persistent=False`**：state_dict 里凭空多 23MB×2，验收 08.2 抓；
- **`start_pos` 从错的地方取**：应取第一层 past 的序列长度（`past[0][0].shape[2]`）而不是词表之类；
- **RoPE 切片位置**：新 token 用 `start_pos:start_pos+seq_len`，不是从头切（否则位置编号错乱）；
- **meta 设备防御缺失**：加载 ckpt 时的 `torch.load(..., weights_only=True / map_location)` 场景下
  参数处于 meta 设备，直接建 buffer 会报 "Cannot copy out of meta tensor"；
- **`ModuleList` 误用成 `nn.Sequential`**：需要逐层取 `presents`，Sequential 会吞返回值。

## 6. 小结 & 下一课

- ✅ Embedding = 词表查找表（可学习）；
- ✅ `persistent=False` buffer = 只跑不算的常量（省 23MB state_dict）；
- ✅ 主体 = embed → 块堆叠 → 终 norm；KV 编排与 start_pos 在这里统一；
- ✅ meta 设备防御保证与 transformers 加载协议兼容。

**下一课（09-causal-lm）**：**点石成金的一课**。把主体输出映射回词表（lm_head），加上**权重绑定**
（embed 与 lm_head 共享、省 490 万参数）、**shift 损失**（预测下一个 token）、你自己的 **generate**
（温度 / top-k / top-p / 重复惩罚 / KV 增量 / HF 保存加载）。做完这一课，你就拥有了一个
**能保存、能加载、能自动对话的完整语言模型**。