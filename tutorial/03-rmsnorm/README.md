# 第 03 课 · RMSNorm：归一化的最小实现

## 目标

手写 `minimind3/rms_norm.py` 的 `RMSNorm`，数值行为与 minimind 标准实现一致。

## 理论

### 1. 为什么需要归一化

深层网络的中间特征尺度漂移会导致梯度爆炸/消失。归一化把每层输入的特征
"拉回"到可控尺度，是训练稳定性的基石。

### 2. LayerNorm vs RMSNorm

- LayerNorm（Ba et al. 2016）：
  $$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$
  需要计算均值 $\mu$ 与方差 $\sigma^2$（两次归约，且要算均值差）。
- RMSNorm（Zhang & Sennrich 2019）：
  $$\hat{x} = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$$
  只算均方根，**没有均值中心化**，省一次归约；实验表明对 Transformer 足够。

**尺度不变、非中心化**：RMSNorm 会把输入的整体缩放“归化”掉：
$\mathrm{RMSNorm}(c x) = \mathrm{RMSNorm}(x)$（尺度不变）；并且它**不做均值中心化**：
输出沿特征维的均值一般不为 0（LayerNorm 输出均值为 0）。
verify 03.2 专门抓这两个数学性质，任何“顺手减去均值”的实现都会失败。

### 3. 实现要点

- `weight` 是可学缩放参数，初值为 1（$\beta$ 偏置被省略）；
- `torch.rsqrt`（$x^{-1/2}$）比 `1/sqrt` 更快更稳；
- **内部用 fp32 计算**，输出转回输入精度：低精度（fp16/bf16）下
  $\sum x^2$ 容易溢出/下溢，fp32 兜底；
- `eps` 放在根号内防除零。

## 任务清单（先手写再看答案）

1. `RMSNorm(dim, eps=1e-5)`，持有可学 `weight`（初值全 1）；
2. `norm(x)`：均方根归一化（用 `rsqrt`）；
3. `forward(x)`：`weight * norm(x.float())` 后 `type_as(x)` 回原精度。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

检查点：形状/初值/数值、尺度不变与非中心化、fp32 内部精度、梯度回传、防除零。
验收通过后，本课的 RMSNorm 将成为注意力 QK-Norm 与最终层归一化的公共件。

## 对照标准实现

`master:model/model_minimind.py → RMSNorm`（逐行等价；标准实现省略偏置、默认 eps=1e-5、
内部 fp32、`type_as` 回原精度）。

## 常见坑

- 忘了 `keepdim=True`：`mean(-1)` 会把最后一维消掉，无法广播；
- `rsqrt(x + eps)` 的括号写错会先加后开方（其实顺序无所谓，但别写成 `rsqrt(x)+eps`）；
- 在 fp16 输入上直接逐元素运算，数值易偏差（verify 03.3 会抓）。