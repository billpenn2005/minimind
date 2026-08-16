# 第 07 课 · 残差块 MiniMindBlock（Pre-Norm）

> 难度：★★☆☆☆ ｜ 预计用时：1~2 小时 ｜ 前置：05+06 课（注意力与前馈）
> 学完本课你应能回答：**残差连接解决什么问题？为什么现在都用 Pre-Norm 而不是 Post-Norm？"梯度高速公路"是什么？**

---

## 0. 本课目标

- [ ] 理解残差连接（Residual Connection）的动机；
- [ ] 对比 Post-Norm 与 Pre-Norm，理解当代 LLM 的选择；
- [ ] 手写 `minimind3/block.py` 的 `MiniMindBlock`（含 KV 透传约定）；
- [ ] 验收 5 项新检查（07.1~07.5，累计 37 项全过）。

---

## 1. 理论

### 1.1 深层的诅咒：梯度消失

网络越深越好（表达力强），但反向传播时梯度要**逐层相乘**一路传回去：
$\partial L / \partial x_0 = \partial L / \partial x_L \cdot \prod_{l} W_l$。层数一多（哪怕 8 层），
连乘会让梯度指数级缩水或爆炸——浅层学不动了。

### 1.2 残差连接的魔法：给梯度一条"高速公路"

残差连接让每一层学习"**增量**"而不是"完整变换"：

$$
x_{l+1} = x_l + f(x_l)
$$

其中 $f$ 是层内计算（Attention 或 FFN + Norm）。反向传播时，输出对输入的偏导是：

$$
\frac{\partial x_{l+1}}{\partial x_l} = 1 + \frac{\partial f(x_l)}{\partial x_l}
$$

那个 **`+1`** 就是"高速公路"：即使深层把 $\partial f$ 压到极小，梯度也能**恒等直通**到最底层。
于是 100 层也敢训。这是 ResNet（2015，CV）和现代 Transformer 共同的基石。

### 1.3 Pre-Norm vs Post-Norm

| | 顺序 | 决策 |
|---|---|---|
| **Post-Norm**（2017 原始 Transformer） | `x + Attention(Norm(x))` 之后才 Norm | 对归一化时机更挑剔，深模型训练不稳 |
| **Pre-Norm**（当代标配） | `x + Attention(Norm(x))`，**入口归一化** | 训练稳定，天然支持"梯度直通" |

我们的块是 Pre-Norm，且是**双子 Pre-Norm**（注意力、前馈各有一个 Norm）：

```python
class MiniMindBlock(nn.Module):
    def __init__(self, config, layer_id):
        self.layer_id = layer_id
        self.self_attn = Attention(config, layer_idx=layer_id)   # 命名与 minimind 完全一致（13 课转换要靠它）
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def forward(self, hidden_states, attention_mask, past_key_value=None, use_cache=False, output_attentions=False):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)          # 1. 归一化
        attn_out, present_kv = self.self_attn(hidden_states, ...)    # 2. 注意力
        hidden_states = residual + attn_out                          # 3. 残差直通
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) # 4. 再归一化
        hidden_states = self.mlp(hidden_states)                      # 5. 前馈
        hidden_states = residual + hidden_states                     # 6. 再残差
        return hidden_states, present_kv
```

注意命名 `input_layernorm` / `post_attention_layernorm` 是 **transformers 生态的通用约定**
（Llama/Qwen 都叫这个），minimind 也不例外——13 课转换权重时 state_dict 键名要对上。

### 1.4 KV 透传：块级接口约定

`forward` 接受 `past_key_value` 并返回 `present_kv`；`use_cache=False` 时返回 None。
这一层"透传"让 08 课的主体模型可以统一编排多层缓存（每层各存各的 K/V）。

---

## 2. 手写任务清单

1. `__init__(config, layer_id)`：两个 RMSNorm、一个 Attention、一个 FeedForward（顺序、命名如上）；
2. `forward`：实现 Pre-Norm 残差两步（attn 一步 + mlp 一步）；
3. 透传 `past_key_value` / `present_kv`；
4. 保持与 minimind 的 state_dict 命名一致；
5. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 3. 验收解读（verify/07_block.py）

| 检查 | 验什么 | 直觉 |
|---|---|---|
| 07.1 | 输出形状正确；子模块命名符合 minimind 约定 | 结构 + 将来的权重转换 |
| 07.2 | **残差恒等**：把所有参数清零后 `block(x) == x`（1e-6） | 参数=0 时 f(x)=0，只有残差直通——最优雅的残差验证 |
| 07.3 | 手写等价式 `x + attn(ln1(x)) + mlp(ln2(residual))` 一致（1e-5） | Pre-Norm 展开合法 |
| 07.4 | 梯度同时流过注意力和 MLP 两个分支 | 残差不是"短路"了学习 |
| 07.5 | `use_cache=False` 时 present_kv 为 None，反之有值 | KV 透传约定 |

> 07.2 是本节最漂亮的检查：如果实现里残差写错了（比如忘了加 x），参数清零后输出就是 0 而非 x，
> 断言立刻现形。数学与代码在此缝合成一个"行为指纹"。

## 4. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/block.py` → `MiniMindBlock` | `model/model_minimind.py` → `MiniMindBlock` |

结构逐行对齐；minimind 的 MoE 分支（`mlp.aux_loss`）以扩展形式保留在本课接口里（07.5 之后的 08 课
会把 `aux_loss` 汇总进输出），Dense 路径完全等价。

## 5. 常见坑

- **Post-Norm 写反**：输出质量与训练稳定性都下降，且验收 07.3 的展开式不匹配；
- **忘了残差直通**：07.2 立刻红叉；
- **`layer_id` 未传**：attention 里 `layer_idx` 用于调试/未来 MoE，缺失会在怪异的地方报错；
- **`output_attentions` 参数缺位**：第 09 课 GenerationMixin 或调试工具会需要它（保持接口完整）。

## 6. 小结 & 下一课

- ✅ 残差 = 学习增量 + 梯度高速公路（`1 + ∂f` 直通）；
- ✅ Pre-Norm = 入口归一化，训练稳定；
- ✅ 子模块命名对齐生态（transformers 约定），为 13 课奠基；
- ✅ KV 透传把缓存编排留给主体模型。

**下一课（08-model）**：把 8 个这样的块 + Embedding 表 + 最终 RMSNorm + RoPE buffer 组装成
**完整主体模型 `MiniMindModel`**。你会学到 embedding 表、`persistent=False` 的 buffer 技巧，
以及 KV-Cache 的"编排层"如何统筹所有层。