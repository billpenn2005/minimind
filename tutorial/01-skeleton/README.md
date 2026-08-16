# 第 01 课 · 项目骨架与验收框架

## 目标

从真正空的分支开始，搭好 minimind3 的工程骨架：
包结构、合成数据工具、程序化验收框架，并理解"语言模型到底是什么"。

## 理论：语言模型范式

语言模型做的事情只有一件：**根据上文预测下一个 token 的概率分布**。

$$P(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} P(x_t \mid x_1, \dots, x_{t-1})$$

- 输入：token 序列；输出：每个位置的 `vocab_size` 维概率分布；
- 训练 = 最大化（真实序列的）对数似然 = 最小化交叉熵；
- 推理/生成 = 从分布中采样（或取 argmax）得到下一个 token，拼接后继续。

整篇教程就是把这个范式"拆成零件再装回去"的过程：
配置 → 归一化 → 位置编码 → 注意力 → 前馈 → 残差块 → 主体 → 输出头 → 数据 → 训练 → 转换 → 测试。

## 任务清单（尽量手写）

1. 阅读本分支文件结构，理解 `minimind3/`（实现）、`verify/`（验收）、`tools/`（工具）、`tutorial/`（文档）四个目录的职责。
2. `tools/make_synthetic_data.py`：补全/理解预训练与 SFT 合成数据生成器（封闭小词表、随机种子可复现）。
3. `verify.py`：理解验收框架——自动发现 `verify/[0-9]*.py` 中的 `CHECKS` 列表并逐项执行。
4. 运行 `python verify.py`，确认本课 5 项检查全部 PASS。

## 验收

```bash
.venv/Scripts/python.exe verify.py
```

期望输出（节选）：

```
  [PASS] 01.1 项目骨架文件齐全
  [PASS] 01.2 合成预训练数据可生成且可复现
  ...
== minimind3 verify: 5 passed, 0 failed ==
```

## 对照标准实现（minimind）

本课没有模型代码；对照点是**数据格式**：
minimind 的 `dataset/pretrain_t2t_mini.jsonl`（`{"text": ...}`）与
`dataset/sft_t2t_mini.jsonl`（`{"conversations": [...]}`）。
我们的合成数据采用完全相同的 schema，后续课的 Dataset 类可以直接复用之。

## 常见坑

- Windows 下请用 `.venv/Scripts/python.exe`（或 venv 激活脚本）运行；
- 合成数据必须"封闭词表 + 可复现种子"，否则后续"loss 必须下降"类验收会不稳定；
- 保持 `verify/` 文件名以两位数序号开头，保证执行顺序确定。