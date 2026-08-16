# 第 03 课 · RMSNorm 归一化

> [上一课 02-config](../02-config/README.md) ← [目录](../../README.md) → [下一课 04-rope](../04-rope/README.md)
>
> 难度：★★☆☆☆ ｜ 预计用时 1 小时 ｜ 前置：02 课

## 0. 本课目标

- [ ] 理解归一化动机（内部协变量偏移）与 LayerNorm/RMSNorm 的差别；
- [ ] 手写 `minimind3/rms_norm.py` 的 `RMSNorm`；
- [ ] 验收 16 项累计（本课新增 03.1~03.5）。

## 1. 理论速览

- **归一化做什么**：每层的输入分布漂移会放大训练不稳定；把每个特征向量"拉回"单位尺度再放缩，梯度更平稳。
- **LayerNorm**：$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma$（先减均值、再除标准差）。
- **RMSNorm**：去掉均值中心化，只除均方根：
  $$\text{RMS}(x) = \sqrt{\frac{1}{d}\sum_i x_i^2}, \qquad \hat{x} = \frac{x}{\text{RMS}(x)+\epsilon} \odot \gamma$$
  直觉：均值中心化对 LLM 收益有限但成本高（需统计全局均值），RMS 省一步还足够稳。
- **scale-invariance（第 03.2 的行为指纹）**：RMSNorm 对输入缩放不变：$\text{RMSNorm}(c\cdot x) = \text{RMSNorm}(x)$（LayerNorm 因减均值**不具备**此性质）——用它抓"偷偷实现成 LayerNorm"的错误。
- **fp32 内部计算**：`mean(x²)` 在低精度下易下溢/溢出，先 `x.float()` 算完再转回原 dtype。

## 2. 任务要求（精确规格）

### 2.1 新建 `minimind3/rms_norm.py`

**模块级**：`import torch`、`from torch import nn`。

### 2.2 `class RMSNorm(nn.Module)`

| 成员 | 类型 | 说明 |
|---|---|---|
| `eps` | `float` | 稳定项（构造参数 2 传入） |
| `weight` | `nn.Parameter` | 可学习缩放向量，初始 `torch.ones(dim)`，形状 `(dim,)` |

**构造 `__init__(self, dim: int, eps: float = 1e-5)`**：`dim`=特征维度；`eps` 默认 1e-5（调用方会传 `config.rms_norm_eps`=1e-6）。

| 方法 | 参数（类型） | 返回（类型） | 行为 |
|---|---|---|---|
| `norm(self, x: torch.Tensor) -> torch.Tensor` | x 形状 `(…, dim)` | 形状同 x | `x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)`（rsqrt 比 1/sqrt 快且稳；**keepdim=True 必须有**） |
| `forward(self, x: torch.Tensor) -> torch.Tensor` | x `(…, dim)` | 形状同 x、dtype 同 x | `(self.weight * self.norm(x.float())).type_as(x)`（内部 fp32，输出还原精度） |

## 3. 手写步骤

```python
# minimind3/rms_norm.py
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5): ...
    def norm(self, x): ...
    def forward(self, x): ...
```
写完跑 `.venv/Scripts/python.exe verify.py 03`。

## 4. 验收解读（verify/03_rmsnorm.py）

| 检查（新） | 验什么 |
|---|---|
| 03.1 | 输出形状 `(B,S,hidden)`；`weight` 形状 `(hidden,)` 且可学习 |
| 03.2 | **scale-invariance**：`rmsnorm(x*5) ≈ rmsnorm(x)`（atol 1e-5），且均值残留 > 0.1（证明**没有**做减均值中心化） |
| 03.3 | 与手写公式 `x/√(mean(x²)+eps)` 逐元素一致（1e-5） |
| 03.4 | 与 `torch.nn.functional` 手写 RMSNorm 等价；fp32 内部计算 |
| 03.5 | 梯度流经 `weight` 与输入 x（backward 后二者有 grad） |

## 5. 参考答案

`answers/minimind3/rms_norm.py`（22 行；先写后对）。

## 6. 常见坑

- **丢 `keepdim=True`**：形状塌掉 → 广播错误；
- **fp16 输入直接 `mean(x²)`**：数值溢出/下溢 → 先 `.float()`；
- **把输入也做均值中心化**：03.2 立刻红叉（scale-invariance 就是为抓它设计的）；
- **`weight` 初始化为 0**：输出恒 0，训练必死。

## 7. 对照标准实现

`answers/minimind3/rms_norm.py` 与 minimind `model/model_minimind.py` 的 RMSNorm 逐行等价（f32 内部、rsqrt、type_as）。

## 8. 小结

✅ 归一化基元完成；✅ 学会"行为指纹"验收法（缩放不变性反证无均值中心化）。
**下一课**：RoPE——给 token 注入位置信息的旋转魔法。