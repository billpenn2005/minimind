# 第 03 课 · RMSNorm 归一化

> 难度：★☆☆☆☆ ｜ 预计用时：1~2 小时 ｜ 前置：第 02 课（知道 hidden_size 是向量维度）
> 学完本课你应能回答：**为什么要把向量"归一化"？RMSNorm 和 LayerNorm 差在哪？为什么要在 fp32 里算？**

---

## 0. 本课目标

- [ ] 理解归一化（normalization）在神经网络中的作用；
- [ ] 对比 LayerNorm 与 RMSNorm（后者是当代 LLM 的标配）；
- [ ] 手写 `minimind3/rms_norm.py` 的 `RMSNorm`（含 fp32 内部计算）；
- [ ] 验收 5 项新检查（03.1~03.5，累计 15 项全过）。

---

## 1. 理论：为什么要归一化？

### 1.1 问题：数值"失控"和训练"不稳"

深度学习训练时，每一层的输入分布持续变化（后一层吃的是前一层的输出，而前一层参数一直在变），
这叫 **internal covariate shift（内部协变量偏移）**。后果：
- 某些维度数值巨大、某些趋近 0，梯度要么爆炸要么消失，训练震荡；
- 激活函数（如 silu）在高绝对值区域饱和，梯度极小而停滞。

**归一化**的答案是：在每层入口处，把向量"拉回"到一个温和的尺度，让模型看到"形状相同、幅度可控"的输入。

### 1.2 两种归一化的公式对比

**LayerNorm（经典）**：每元素减去均值（中心化），再除以标准差：

$$
\text{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta
\qquad
\mu = \frac{1}{d}\sum x,\quad \sigma^2 = \frac{1}{d}\sum (x - \mu)^2
$$

**RMSNorm（当代）**：**不做均值中心化**，直接用均方根（root mean square）缩放：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \odot \gamma
\qquad
\text{mean}(x^2) = \frac{1}{d}\sum x_i^2
$$

差别只有一点：RMSNorm **不减均值**。它没有可学习的偏置 β，只有缩放权重 γ（初始化为 1）。

### 1.3 为什么"不减均值"反而赢了？

两行代码讲完核心直觉：
1. RMSNorm 省掉了 `x - μ` 这一步，GPU 上更省算子、更快；
2. Transformer 里每个 token 的归一化是**独立**做的（对最后一维），均值这个信息其实"信息量很低"——
   减掉它几乎无损，RMS 缩放才是主要作用。**删掉可有可无的计算，换来速度与稳定**，这就是工程。

数学上有个好性质：**缩放不变性（scale-invariance）**：

$$
\text{RMSNorm}(c \cdot x) = \text{RMSNorm}(x) \quad \text{对任意正标量 } c
$$

（分子分母同时被 c 放大，抵消了。）而 LayerNorm 做的是"去均值 + 缩放"，并不具有这个性质——
验收 03.2 就会用这个性质做"行为指纹"：**如果你的实现偷偷减了均值，缩放不变性就会失败**。妙！

### 1.4 为什么要在 fp32 里算？

`x.pow(2).mean()` 会把小数值平方——在 fp16（半精度）下，小数的平方会超出可表示精度，产生噪声甚至 NaN。
业界惯例：**归一化统计量（这里指 mean(x²)）必须在 fp32 里计算**，算完再回到输入的 dtype：

```python
x_fp32 = x.float()                       # 提升精度
mean_sq = x_fp32.pow(2).mean(-1, keepdim=True)
x_norm = x_fp32 * torch.rsqrt(mean_sq + eps)
return x_norm.type_as(x) * self.weight   # 回到原 dtype 再乘可学习权重
```

`torch.rsqrt` = `1/sqrt` 的快速实现。`eps`（=1e-6）防止除零。

---

## 2. 阅读参考答案（minimind3/rms_norm.py）

- `class RMSNorm(nn.Module)`：一个 `weight`（可学习参数，初值 1，形状 `(dim,)`）；
- `forward(x)`：按 §1.4 的三行实现；
- 输入形状 `(..., dim)` 任意——`mean(-1)` 只作用于最后一维，所以 token 批次的任何排布都能用。

## 3. 手写任务清单

1. 定义 `RMSNorm(nn.Module)`，`__init__(dim, eps)`；
2. 创建 `nn.Parameter(torch.ones(dim))` 作为 `self.weight`；
3. 实现 forward：fp32 内部计算 mean(x²) → `rsqrt` → 缩放 → `type_as(x)` 回原精度 → 乘 weight；
4. 思考（不写也行）：为什么 `keepdim=True`？——因为要在广播时保持"每个 token 一个缩放因子"的位置；
5. 验收。

```bash
.venv/Scripts/python.exe verify.py
```

## 4. 验收解读（verify/03_rmsnorm.py）

| 检查 | 验什么 | 直觉 |
|---|---|---|
| 03.1 | 输出形状正确、weight 初值为 1 | 最基础 |
| 03.2 | **缩放不变性** + 输出特征均值非零（对比 LayerNorm） | 行为指纹：不许偷偷减均值 |
| 03.3 | fp16 输入也能正确计算（内部 fp32），输出 dtype 与输入一致 | 精度设计 |
| 03.4 | 梯度能流回 weight 与输入 | 反传连通性 |
| 03.5 | 全零输入返回全零（0/√0+ε=0，无 NaN） | 数值稳定性 |

> 03.2 的细节：验证器先断言 `rms(c*x) ≈ rms(x)`（任意 c），再检查输出的特征均值**不是 0**
> （LayerNorm 版实现每一维均值都被清零，这个指纹立刻露馅）。

## 5. 对照标准实现

| minimind3 | minimind 标准实现 |
|---|---|
| `minimind3/rms_norm.py` → `RMSNorm` | `model/model_minimind.py` → `RMSNorm` |

两者的公式完全一致（fp32 内部 + rsqrt + type_as）。

## 6. 常见坑

- **忘记 `keepdim=True`**：形状变成 `(B,S)` 而非 `(B,S,1)`，广播错误 → 立刻改；
- **在 fp16 里直接算 mean(x²)**：小批次下偶发 NaN，验收 03.3 抓住你；
- **`weight` 忘了初始化/初始化成 0**：模型输出直接清零，梯度消失；
- **把 LayerNorm 的 β 也抄进来**：03.2 会以"均值非零失败"提醒你。

## 7. 小结 & 下一课

- ✅ 归一化解决"数值失控、训练不稳"；
- ✅ RMSNorm = 去均值中心化的极简版，缩放不变 + 省算子；
- ✅ fp32 内部计算的工程细节；
- ✅ "行为指纹"式验收：性质断言比形状断言更能抓住实现偏差。

**下一课（04-rope）**：语言模型处理的是**序列**，但注意力本身"不关心顺序"——你需要给每个 token
打上**位置牌**。你会从零推导 RoPE 旋转位置编码，并亲手验证它的核心魔法：**相对位置不变性**。