# 第 01 课 · 骨架与验收框架

> [目录](../../README.md) → [下一课 02-config](../02-config/README.md)
>
> 难度：★☆☆☆☆ ｜ 预计用时 30~60 分钟 ｜ 前置：无（会装 Python 环境即可）

## 0. 本课目标

- [ ] 理解"语言模型 = 下个 token 预测器"的最小闭环；
- [ ] 学会用 `verify.py NN` 按课验收（本教程的评分标准）；
- [ ] 手写包入口 `minimind3/__init__.py`；
- [ ] 验收 6 项（01.1~01.5 + 01.x，累计 6 项全过）。

## 1. 理论速览

- **语言模型任务**：给定前文 $x_1..x_t$，预测下一个 token 的概率 $P(x_{t+1})$；整句概率 = 链式乘积 $\prod P(x_t|x_{<t})$。
- **token 与词表**：词表把文本切成 token（本教程用 minimind 的 6400 词 BPE 词表，第 10 课接入）；模型在 6400 维概率上做分类。
- **训练**（第 11~12 课）：拿大量文本，用交叉熵 $-\log P(正确token)$ 做梯度下降；**生成**（第 09 课）：采样，把新 token 接回输入再预测。
- **本课不碰模型**：先把"写代码-验收"的回路跑通。验收框架 = 程序化的评分标准（行为断言，非风格检查）。

## 2. 任务要求（精确规格）

> 本课只需**包入口**这一个手写文件；其余（README/requirements/.gitignore/verify/ 等）分支已自带。

### 2.1 新建 `minimind3/__init__.py`（包入口）

| 要求项 | 具体约束 |
|---|---|
| 文件路径 | 仓库根 `minimind3/__init__.py`（目录需自建） |
| 文件编码 | UTF-8 |
| 模块 docstring | 首行为 `"""minimind3 —— 手写实现的 MiniMind 同架构因果语言模型（教程产物）。"""`（可稍改措辞） |
| 模块级变量 `__version__` | 类型 `str`，值 `"0.1.0"` |
| 其它内容 | 不要 import torch / 不要定义类（第 02 课起才有） |

### 2.2 验收框架用法（必会）

| 命令 | 含义 |
|---|---|
| `.venv/Scripts/python.exe verify.py 01` | 只验收第 01 课（本课 6 项） |
| `.venv/Scripts/python.exe verify.py 01 --fast` | 带 `--fast` 跳过耗时检查（本课无耗时项，二选一） |
| `.venv/Scripts/python.exe verify.py` | 全部 14 课（毕业验收） |

### 2.3 按课留白约束（验收强制执行）

- `minimind3/` 里**只允许出现"截止当前课"的产物文件**：第 01 课只有 `__init__.py`，第 02 课才允许再出现 `config.py`，以此类推（制度目标：防跳课、防把答案回拷——第 N 课验收会拒绝任何第 N+1 课才该有的文件）；
- 仓库根 `data/` 在第 10 课之前必须不存在（第 10 课起由你生成）；
- `answers/minimind3/`、`answers/data/` 必须存在（参考答案已随分支分发）。

## 3. 手写步骤

```bash
mkdir minimind3                      # 创建你的工作区
# 写 minimind3/__init__.py（见 §2.1 规格）
.venv/Scripts/python.exe verify.py 01
```

先跑一次 `verify.py 01` 看"缺什么"（第一条会报 `缺少文件: minimind3/__init__.py`），补齐后再跑。

## 4. 验收解读（verify/01_skeleton.py）

| 检查 | 验什么 |
|---|---|
| 01.1 骨架文件齐全 | 分支自带文件 + 参考答案（answers/）+ 你的 `minimind3/__init__.py` 都在 |
| 01.2 合成预训练数据可生成且可复现 | `tools/make_synthetic_data.py` 的 `make_pretrain_corpus(num, rng)`：同种子输出逐条相等 |
| 01.3 合成 SFT 数据 schema 正确 | 每条 `{"conversations": [{"role","content"}]}`，role ∈ {user, assistant} |
| 01.4 生成脚本命令行可运行且体积小 | `--out --num --seed` 三个参数可用，产物 < 100KB |
| 01.5 验收框架可发现并运行检查 | `verify.01_skeleton` 模块注册了 ≥4 项检查 |
| 01.x 按课留白 | `minimind3/` 只含 ≤ 当前课的产物文件；`data/` 未到第 10 课不存在（防跳课/防回拷答案） |

## 5. 参考答案

完成验收后对照：`answers/minimind3/__init__.py`（3 行）。**先写后对，禁止整文件复制。**

## 6. 常见坑

- **用系统 python 跑**：系统 Python 没装 torch，验收统一用 `.venv/Scripts/python.exe`；
- **在子目录运行**：`verify.py` 必须在仓库根运行（它用相对仓库根解析路径）；
- **想偷懒复制 answers**：01.x 会拒绝任何 `minimind3/` 下的多余文件。

## 7. 对照标准实现

| 你手写 | minimind 标准 |
|---|---|
| `minimind3/__init__.py` | —（minimind 无对应包入口） |

数据 schema（`{"text"}` / `{"conversations"}`）与 minimind 仓库 `dataset/*.jsonl` 完全一致——第 10~12 课换数据时无需改代码。

## 8. 小结

✅ 手写首个文件（包入口）；✅ 掌握"跑验收看差什么"的学习回路；✅ 数据生成器就绪。
**下一课**：写 `MiniMindConfig`——全模型超参数的唯一蓝图。