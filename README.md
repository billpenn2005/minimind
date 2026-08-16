# MiniMind3 · 手写大语言模型教程（单分支一站式版）

> 从**空白的 `minimind3/` 文件夹**开始，跟着 14 章教程逐课手写一个与 minimind 等价的语言模型；
> 每章都有**精确到函数签名/成员类型/返回形状**的任务要求，写完用
> `verify.py 章节号` 程序化验收。全程 CPU、零下载。
>
> 📖 总设计与维护手册（含全部设计问答）：**[AGENTS.md](AGENTS.md)**
> 📦 参考答案（先写后对，禁止回拷）：**[answers/](answers/README.md)**

---

## 0. 目录结构

| 路径 | 角色 | 说明 |
|---|---|---|
| `minimind3/` | **你的工作区（从零开始）** | 本分支**不含**任何实现；每课动手创建一个文件 |
| `answers/minimind3/` | 参考答案（完整实现） | 14 门课的最终成品；写不出来再看 |
| `answers/data/` | 规范合成数据 | 第 10 课可用生成器复现出完全相同的内容 |
| `data/` | 你的数据目录 | 第 10 课起由你生成（`/data/` 已 gitignore） |
| `verify.py` + `verify/` | 验收框架 | 每课一个 `verify/NN_xxx.py` 检查模块 |
| `tools/` | 工具 | 数据生成器、标准实现对照导出 |
| `tutorial/NN-xxx/` | 每章教程 | 本章入口是各目录的 `README.md` |
| `AGENTS.md` | 总设计 + 维护手册 | pi 自动加载；也供人类维护参考 |
| 其余（`model/ dataset/ trainer/ scripts/` 等） | minimind **标准实现** | 从 master 分支检出的参考代码（教程分支的孤儿历史里没有，仅在你本地克隆里可见）；用它对比"你与 minimind 的差距" |

---

## 1. 章节导航（学习路线图）

> 每章的验收命令形如 `verify.py 05`（验收第 01~05 课）；`--fast` 跳过耗时训练检查。

| 章 | 目录 | 核心概念 | 本课验收 | 累计检查 |
|---|---|---|---|---|
| 01 | [tutorial/01-skeleton](tutorial/01-skeleton/README.md) | 骨架、验收框架、合成数据生成器 | `verify.py 01` | 6 |
| 02 | [tutorial/02-config](tutorial/02-config/README.md) | MiniMindConfig 超参数蓝图 | `verify.py 02` | 11 |
| 03 | [tutorial/03-rmsnorm](tutorial/03-rmsnorm/README.md) | RMSNorm 归一化 | `verify.py 03` | 16 |
| 04 | [tutorial/04-rope](tutorial/04-rope/README.md) | RoPE 旋转位置编码 | `verify.py 04` | 22 |
| 05 | [tutorial/05-attention](tutorial/05-attention/README.md) | GQA 注意力 + KV-Cache | `verify.py 05` | 28 |
| 06 | [tutorial/06-feedforward](tutorial/06-feedforward/README.md) | SwiGLU 前馈 | `verify.py 06` | 33 |
| 07 | [tutorial/07-block](tutorial/07-block/README.md) | Pre-Norm 残差块 | `verify.py 07` | 38 |
| 08 | [tutorial/08-model](tutorial/08-model/README.md) | 模型主体 + KV 编排 | `verify.py 08` | 43 |
| 09 | [tutorial/09-causal-lm](tutorial/09-causal-lm/README.md) | 因果头 + 损失 + generate | `verify.py 09` | 51 |
| 10 | [tutorial/10-data](tutorial/10-data/README.md) | 分词器 + ChatML + 掩码标签 | `verify.py 10` | 56 |
| 11 | [tutorial/11-pretrain](tutorial/11-pretrain/README.md) | 预训练循环 | `verify.py 11`（12 章起含 CPU 实训） | 60 |
| 12 | [tutorial/12-sft](tutorial/12-sft/README.md) | 指令微调（掩码损失） | `verify.py 12` | 63 |
| 13 | [tutorial/13-convert](tutorial/13-convert/README.md) | pth ⇄ HF transformers 转换 | `verify.py 13` | 68 |
| 14 | [tutorial/14-final](tutorial/14-final/README.md) | 端到端流水线 + 内容级验收 | `verify.py 14` | 71 |

> 各章累计检查数含通用护栏（01.x 起点留白、14.x 结构自洽），比旧多分支版的 5/10/…/70 各多 1 项。

---

## 2. 快速开始

### 2.1 环境（一次性）

```bash
# 1) 克隆（含教程单分支 tutorial-main 与 minimind 标准实现 master）
git clone https://github.com/billpenn2005/minimind.git
cd minimind
git checkout tutorial-main

# 2) 创建虚拟环境并装依赖（全部 CPU 即可）
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Windows
# 或 Linux/macOS: .venv/bin/python -m pip install -r requirements.txt

# 3) 自检：验收框架可用（此刻应看到 01.1 缺 minimind3/__init__.py 的理性失败）
.venv/Scripts/python.exe verify.py 01 --fast
```

### 2.2 每章循环

```bash
.venv/Scripts/python.exe verify.py 01      # ① 先跑验收：看还差什么
# ② 读 tutorial/01-skeleton/README.md 的任务要求，手写代码
.venv/Scripts/python.exe verify.py 01      # ③ 再验收：全绿进入下一章
```

---

## 3. 验收机制

- `verify.py NN`：只验收第 01~NN 课的检查（逐课累积；每课一个 `verify/NN_xxx.py` 模块）；
- `verify.py`（无参数）：全部 14 课，毕业验收；
- `--fast`：跳过训练/生成类耗时检查（此模式数据生成器仍在运行时验证）；
- 任一检查失败 → 退出码 1 且打印 `[FAIL] 检查名: 错误详情`；
- 检查是"**行为指纹**"而非实现绑定：断言形状/数值/行为（如 RoPE 保范、KV 增量 == 全量、CK gap、1e-5 转换往返），不检查你的措辞。

---

## 4. FAQ

- **参考答案在哪？** `answers/minimind3/`。先写后对；`verify/01_skeleton.py` 的"起点留白"检查会拦截把答案拷回工作区。
- **数据从哪来？** 第 10 课用 `python -m tools.make_synthetic_data --out data --num 300` 生成（确定性、可复现，与 `answers/data/` 内容一致）；教程严禁联网下载数据/权重。
- **卡课超过 2 小时？** 允许看答案：读 `answers/minimind3/<对应文件>` 并**默写**后再回来；看懂 ≠ 会写，动手才算。
- **控制台中文乱码？** Windows GBK 终端显示中文乱码是正常现象；数据本身是正确的（encode/decode 往返正确即真值）。
- **`ModuleNotFoundError`？** 检查是否在仓库根目录运行、依赖是否装进 `.venv`；训练类检查必须用 `.venv/Scripts/python.exe`。
- **为什么单分支？** 旧 14 分支链（tutorial/01-skeleton…14-final）已归档；本分支所有学习与验收在同一条时间线上，`git log` 即你的学习进程。
- **想对比 minimind 标准实现？** `bash tools/fetch_reference.sh` 导出到 `reference/`；或直接读工作区里的 `model/`、`trainer/` 等目录。

## 5. 你会在 14 章后拥有

```
minimind3/          ← 你手写的完整语言模型
├── config.py         模型蓝图
├── rms_norm.py / rope.py / attention.py / feed_forward.py / block.py / model_body.py
├── causal_lm.py      lm_head + 损失 + generate（温度/top-k/top-p/重复惩罚/KV 增量）
├── tokenizer_utils.py + datasets.py   分词与 ChatML 数据集
├── train_pretrain.py / train_full_sft.py  预训练与 SFT 循环
├── convert.py        pth ⇄ HF 转换（自包含 remote-code）
└── e2e.py            端到端流水线
```
毕业验收 `verify.py` 全绿后，模型能背出合成数据里的事实（"猫是一种会抓老鼠的动物。"），
并可作为标准 HF 目录分发。