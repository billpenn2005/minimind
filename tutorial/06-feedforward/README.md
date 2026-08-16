# 第 06 课 · SwiGLU 前馈层（FeedForward）

> 难度：★★☆☆☆ ｜ 预计用时：1~2 小时 ｜ 前置：05 课注意力
> 学完本课你应能回答：**Transformer 为什么要一个"前馈层"？SwiGLU 的"门"在门什么？为什么中间维度反而更短？**

---

## 0. 本课目标

- [ ] 理解注意力与前馈层的分工（"交换信息" vs "加工信息"）；
- [ ] 理解激活函数的作用与 SiLU、GELU、SwiGLU 的关系；
- [ ] 手写 `minimind3/feed_forward.py` 的 `FeedForward`（gate/up/down 三段式）；
- [ ] 验收 5 项新检查（06.1~06.5，累计 32 项全过）。

---

## 1. 理论

### 1.1 注意力的局限：只会"搬运"，不会"思考"

注意力做的事是**按权重混合**内容：输出是输入的加权和（组合但不改变"信息单元"本身）。
一个只有注意力的网络，等价于一个**线性加权器**——它无法学习"如果 X 且 Y 则 Z"这类**非线性**规则。

前馈层（MLP）补上这一环：**对每个 token 独立做一次非线性变换**（忘却 token 间的交叉，专注"这一个
token 该被加工成什么"）。两个部件交替堆叠（07 课组装），就形成了 Transformer 的完整骨架：

```
注意力层：  全局信息交换（token ↔ token）
前馈层：    局部信息加工（token 自身，非线性抉择）
```

### 1.2 经典 MLP 与"中间维度"的经济学

经典实现是两层线性 + 一个激活（如 GELU）：

$$ \text{FFN}(x) = \text{GELU}(x W_{up}) W_{down}, \qquad d_{mid} \approx 4 \times d_{model} $$

瓶颈：`W_up` 的形状是 `(d, 4d)`——中间维度 $4d$ 直接吃下**模型 ~2/3 的参数量**。这是"代价最高的零件"。

### 1.3 SiLU（也叫 Swish）：门控激活

$$
\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

它把输入"自己当门控"：负值被压向 0，正值几乎线性。比 ReLU 平滑（处处可导，无拐点），
比 GELU 表达力更强。minimind 用 `hidden_act="silu"`（config 里可换任何 `ACT2FN` 注册的激活）。

### 1.4 SwiGLU：显式三门控

$$
\text{SwiGLU}(x) = \underbrace{\text{SiLU}(x W_{gate})}_{\text{门}} \odot \underbrace{(x W_{up})}_{\text{内容}} \cdot W_{down}
$$

`gate` 决定"放多少内容过去"（0~1 之间的软开关），`up` 提供内容，`down` 合成输出。
相比经典 MLP（GELU），SwiGLU 的**门控是显式学习**的，实证效果更好。

**参数量经济学**：同样效果下，SwiGLU 只需 $2/3$ 的中间维度——所以 minimind 把 `intermediate_size`
设为 `ceil(h·π/64)·64 ≈ 2432`（≈ 3.2×d），而不是经典的 4×d：**更少参数、更好效果**（03 课那个"
π 取整"谜题在这里揭晓）。

### 1.5 为什么"无 bias"？

现代大模型（Llama、Qwen、MiniMind）的线性层一律 `bias=False`。原因：有 RMSNorm 先行归一化，
bias 的贡献冗余；去掉还能省参、对齐生态（权重格式转换更干净，13 课受益）。这就是为什么
`nn.Linear(d, d, bias=False)` 三件套。

---

## 2. 阅读参考答案（minimind3/feed_forward.py）

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn    = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

命名（gate/up/down）与 minimind 完全一致——这是**故意的**：第 13 课转换权重时，
`state_dict` 的键名必须能对上标准实现。

## 3. 手写任务清单

1. 两个 `nn.Linear(..., bias=False)` 上行 + 一个下行（顺序按 minimind 命名）；
2. `act_fn = ACT2FN[config.hidden_act]`（transformers 的激活注册表）；
3. forward 一行：`down(silu(gate(x)) * up(x))`；
4. （可选加分）支持 `config.intermediate_size` 覆盖——验收 06.4 用一个不同数值的配置来测。
5. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 4. 验收解读（verify/06_feedforward.py）

| 检查 | 验什么 |
|---|---|
| 06.1 | 上行输出 `(B,S,intermediate)`、下行 `(B,S,hidden)`；三投影无 bias |
| 06.2 | 结果 == 手写 `down(silu(gate)xup)` 公式（1e-5） |
| 06.3 | `silu(x) == x*sigmoid(x)`；且与 `ACT2FN["silu"]` 一致 |
| 06.4 | `intermediate_size` 覆盖生效（开小尺寸也不报错） |
| 06.5 | 梯度连通 + 零输入输出 0（SiLU(0)=0，无残留） |

## 5. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/feed_forward.py` → `FeedForward` | `model/model_minimind.py` → `FeedForward` |

代码等价；minimind 额外支持 MoE 分支（`MLPMoe` + router），本教程 07~09 课保留 `aux_loss` 字段的
接线（`use_moe=False` 时为零），把 MoE 完整实现留给进阶。

## 6. 常见坑

- **gate/up 混淆**：`silu(gate) * up`——把 gate 和 up 写反了效果显著变差（两种写法都"能跑"，验收 06.2 会抓）；
- **默认 `intermediate_size=2432` 心里没数**：hidden=768 时它来自 π 取整规则（02 课）；
- **忘了 `ACT2FN`**：直接 `nn.SiLU()` 也等价，但失去 config 驱动的灵活性；
- **给线性层加 bias**：不报错但风格偏离生态（13 课 strict 加载会报 key 不匹配）。

## 7. 小结 & 下一课

- ✅ 注意力 = 交换，前馈 = 加工；
- ✅ SwiGLU = SiLU(gate) ⊙ up → down，显式门控；
- ✅ 中间维度缩短 1/3（4d → 3.2d）是 SwiGLU 的参数量红利。

**下一课（07-block）**：把注意力 + 前馈 + 两个 RMSNorm + 残差连接组装成 `MiniMindBlock`。
你会理解 **Pre-Norm 残差**——当代 LLM 的训练稳定密码，以及"梯度高速公路"为什么让深层网络可以训练。