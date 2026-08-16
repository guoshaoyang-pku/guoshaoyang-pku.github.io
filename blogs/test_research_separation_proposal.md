# Test / Research 分离架构方案（Proposal）

> **状态：✅ 已撰写，尚未执行（设计参考）**
> 本文件是一个**架构方案设计**，当前线上跑的三个 autoresearch 系统（interactive / codex / dsh）**均未实现**本方案。
> 当前实际现状仍是「**模型自律汇报中间进展 + 最后 test 收尾**」：模型自己跑训练、自己看 loss、自己在最终结束时才调用官方 `test()` 评测一次取最终分。
> 本方案探讨的是：当我们要严肃扩大预算、追求可审计的规模化运行时，test 与 research 应如何分离。写下来供讨论，**没有部署**。

---

## 1. 背景：当前 test() 机制（已查清的现状）

MLS-Bench 的 `test()` 是模型可调用的一个工具（`tools.py` 中注册），其行为分为两类：

| 任务类型 | test() 行为 | 示例 |
|---|---|---|
| **算法型** | 运行 FIXED 评测脚本（如 `run_eval.py`），纯评测、**无训练**，秒级~分钟级返回标准指标 | `causal-observational-*`、`causal-discovery-*`（SF100 等 hidden test_cmd 会被剥离反馈但仍计入） |
| **训练型** | test_cmd 直接跑 agent 可编辑的 `custom_*.py`，**每次 test 强制从零重训** | `ml-calibration`、`ml-subgroup-calibration-shift`、`llm-pretrain-*` |

### 1.1 训练型任务每次 test 从零重训（已源码级确认）

- `llm-pretrain-linear-attention` / `llm-pretrain-bitlinear` 的 test_cmd：
  - `gpt-345m | scripts/gpt_345m.sh | compute 4.0 | time 12:00:00` —— **4 GPU × 12 小时完整预训练**
  - `lm-eval-345m | scripts/lm_eval_345m.sh | compute 1.0 | time 1:00:00`
- 全库搜索 `ckpt` / `checkpoint` 均无结果：**没有 ckpt 复用机制**。模型每次调 `test()` 都从随机初始化重训一遍。
- 后果：LLM 类任务里 agent 无法「先训一个 ckpt，改点小东西再测」，任何微调都要付 12 小时全量重训的代价；test 次数预算（官方 max_tests=3，我们自研放宽到 10+）直接决定了能做的迭代深度。

### 1.2 训练型任务的防作弊漏洞（已确认）

- `ml-calibration` / `ml-subgroup-calibration-shift` 的**数据划分**（`train_test_split` 60/20/20）和**打分逻辑都写在 agent 可编辑的 `custom_calibration.py` 里**。
- 即：agent 既是运动员又是裁判——理论上它能改 split 让 test 集变小/变简单，或直接自己算分。
- 目前靠「agent 自律 + 最终 leaderboard 抽查」兜底，没有机制性隔离。

### 1.3 调度器无 GPU 时长记账（已确认）

- ophis 的 `scheduler.py` 是自定义轻量 GPU-aware 调度器，只做 GPU 分配（`_get_free_gpus`），**没有 GPU 使用时长记账**。
- 我们在对比三系统时用的是外部 `compare_systems.py monitor`（采样 util/显存估算活跃时长），不是调度器原生能力。

---

## 2. 方案：Test / Research 分离架构

### 2.1 核心思想

> **Research 与 Test 并存且解耦**：
> - **Research 阶段**：模型在自己可编辑的 workspace 里自由探索、写代码、自己训练出 **ckpt**（存在约定路径），自己看中间量（loss、指标、日志）——这部分是 OPHIS 闭环的主战场，中间过程**主要是给人看过程**。
> - **Test 阶段**：模型**有限次数**调用 `test()`，test 使用**额外分配的 GPU**（不抢占 research 的卡），对**当前约定的 ckpt** 跑**标准评测**（host 侧固定脚本、固定数据划分、固定打分），返回标准结果。
> - **test 默认不产生 ckpt**：test 只评测、不训练。训练只发生在 research 阶段。

### 2.2 组件设计

```
┌────────────────────────── Research 侧 ──────────────────────────┐
│  agent workspace（可编辑）                                        │
│   ├─ custom_*.py        （agent 写的模型/方法）                    │
│   ├─ train.sh           （agent 写的训练入口，输出 ckpt）           │
│   └─ ckpt/              （约定 ckpt 路径，见 2.3）                 │
│   agent 自由调 GPU：自己跑训练，自己看中间量（loss/grad/激活）        │
└──────────────────────────────────────────────────────────────────┘
                          │  ckpt 落盘（约定路径）
                          ▼
┌────────────────────────── Test 侧（host 托管）──────────────────┐
│  test() 调用（有限次数预算）                                       │
│   ├─ 固定评测脚本（host 侧，agent 不可改）                         │
│   ├─ 固定数据划分（host 侧，与 agent 编辑区隔离）                   │
│   ├─ 固定打分（host 侧实现）                                       │
│   └─ 使用额外 GPU（调度器单独队列），返回标准指标                    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.3 ckpt 约定

- **约定路径**：`<workspace>/ckpt/latest.pt`（或按 step 命名的 `ckpt/step-<n>.pt`，便于回滚）。
- test() 只读该路径；**test 本身不写 ckpt**（不训练）。
- 若 ckpt 缺失 → test() 返回 `NO_CKPT`，不扣 test 次数（或扣 0.5 次，设计细节可议）。
- agent 可在 research 阶段任意覆盖 ckpt；test 只对「调用时刻」的 ckpt 快照评测。

### 2.4 host 侧固定评测（堵住运动员兼裁判漏洞）

- 数据划分、评测脚本、打分器全部放到 **host-only 目录**（如 `holdout/eval_<task>.py` + 固定 split seed），agent 无写权限。
- 评测逻辑与当前 `run_eval.py` 的 FIXED 部分保持一致，但把 `train_test_split` 和打分从 `custom_*.py` 里**抽出来**。
- agent 只提交「方法定义」（模型结构/训练超参），不接触评测器。
- 防越界检查：test 时用 `git diff` 对比 agent 对评测相关文件的改动，超界即拒绝。

### 2.5 GPU 分配与记账

- **Research GPU**：调度器给 agent 一个 research 池（如 2 卡），agent 自由用、无时长限制。
- **Test GPU**：调度器单独一个 test 池（如 1 卡），test() 提交任务到该池排队执行。
- **记账**：调度器原生记录每个 research/test 任务的 GPU 活跃秒数（`util>0 或显存>100MB` 的采样×间隔，对齐现有 `compare_systems.py` 口径），写入 `gpu_usage.csv`（task / system / wall_sec / gpu_sec / peak_mem）。
- 这样三系统对比的「GPU 时长」就从外部采样变成调度器的一等公民。

### 2.6 收益

| 现状痛点 | 方案后 |
|---|---|
| 训练型任务每次 test 全量重训（LLM 4卡×12h） | 只在 research 阶段训练，test 只评测 → 迭代成本骤降 |
| 中间过程只靠模型自律汇报 | 中间过程落在 ckpt 序列 + 日志，可回放、可审计 |
| agent 既是运动员又是裁判 | 评测器 host 托管，agent 不可改 |
| GPU 时长靠外部采样估算 | 调度器原生记账 |
| 最终分数与过程解耦 | 最终分=最后一个 test()，过程=research 的 ckpt/日志，二者都保留 |

### 2.7 风险与权衡（未解决项）

- **ckpt 语义不统一**：不同任务 ckpt 格式差异大（torch save / 自定义），需要 per-task 的 host 侧加载器；这是主要实现成本。
- **LLM 类任务**：research 阶段自己训到一半的 ckpt 未必与 test 的评测脚本（`lm_eval_345m.sh` 期望的格式）兼容，需约定导出格式。
- **公平性**：如果 test 只评测不训练，「每次 test 从零重训」的官方口径就被打破了——不能再与官方 leaderboard 直接比，只能与「同一套自研三系统」横向比。需要用户确认这是可接受的（我们本就扩大预算、不追官方可比性）。
- **防作弊强度**：host 侧固定评测能防「改 split / 自己算分」，但防不了「agent 过拟合 test 集」（多次 test 后记住答案）。缓解：hidden test_cmd 机制保留（如 non-gaussian 的 SF100）。

---

## 3. 当前现状（未按本方案执行）

线上实际跑的仍是：

1. **research 与 test 不分离**：agent 在 workspace 里写代码 → 自己跑训练（自律汇报中间进展）→ 最后调 `test()` 取最终分。
2. **训练型任务每次 test 从零重训**：`custom_*.py` 全量跑（LLM 类 4 GPU×12h）。
3. **中间过程主要给人看过程**：session trace（`session.jsonl.zstd`）+ 日志保留，供人回溯 agent 的观察→诊断→干预链。
4. **调度器只分卡不记账**：GPU 时长由外部 `compare_systems.py monitor` 采样估算。

本方案仅为设计参考，**未被任何系统采用**。若要落地，建议从 `ml-calibration`（单卡、纯 sklearn、有明确 ECE 指标）先做 PoC，验证 ckpt 约定 + host 固定评测的最小闭环。

---

## 4. 来源与依据

- `llm-pretrain-*` test_cmd（4 GPU × 12h）：`vendor/tasks/llm-pretrain-*/config.json`
- `custom_calibration.py` 内含 `train_test_split` + 打分：`vendor/workspace/ml-calibration/*/scikit-learn/custom_calibration.py`
- scheduler 只做 `_get_free_gpus` 无记账：`src/mlsbench/scheduler.py`
- GPU 时长外部采样：`codex_harness/compare_systems.py monitor/report`
- hidden test_cmd 剥离反馈机制：`tools.py` `_run_all_cmds`
