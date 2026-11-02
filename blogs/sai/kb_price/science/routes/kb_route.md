# KB 路线：知识库里关于优化器子系统的全部主张

2026-10-31。只读 ArchitectureIQ 的 KB 文件，没有调用模型，没有训练。
复现：`python3 kb_price/science/scripts/sci_kb_route.py`，写出 `data/cands_kb.json` 和本文件。

子系统：两个训练配置的数据集、模型、损失、批大小和预算 T 都相同，只差优化器设置（类型、学习率、动量、权重衰减、betas）。问题是哪一个的最终测试指标（10 个种子的均值）更低。

## 1. 结论

- KB 里关于这个子系统的主张共 252 条（读过 270 条，18 条已被同一 KB 的后继主张合并）。
- 按 KB 自己的依据定层次：K 213 条，L 39 条，S 0 条。
- 写出了推导的有 40 条，逐条按 (a)(b)(c) 检查，没有一条是 S。
- 推导大多是对的，但只推到“位移 Δ 的换算”。要预测谁胜，每一条都还要经过“Δ 大者胜”这座桥。KB 只用胜负计数检验这座桥，而胜负正是要预测的关系，所以不满足 (b)。
- K1001（Δ 规则本身）的机制字段写着 no theory is established：KB 自己承认核心规则只是启发式。
- 42 条写成了可执行规则。基准和实验室成绩都是样本内的：v1.5 的 500 题有 498 题出现在某个 KB 的语料或解题记录里，实验室行就是这些循环自己跑的实验。

## 2. 搜了哪些来源

| 来源 | 主张记录 | 提到优化器 | 子系统内 | 其中可执行 |
|---|---:|---:|---:|---:|
| evolve:kb_evolve_archonly | 370 | 295 | 106 | 3 |
| evolve:kb_evolve_puct20_arch | 331 | 274 | 105 | 3 |
| evolve:kb_evolve_puct40_arch | 331 | 273 | 105 | 3 |
| loop:luna_main | 320 | 266 | 113 | 3 |
| evolve:kb_evolve_puct20_mixed | 316 | 261 | 113 | 3 |
| evolve:kb_evolve_run3 | 316 | 257 | 113 | 3 |
| loop:qwen_6ep_40eval | 315 | 260 | 113 | 6 |
| evolve:kb_evolve_run1 | 306 | 253 | 108 | 3 |
| loop:smoke_qwen | 299 | 243 | 107 | 3 |
| loop:luna_b15 | 296 | 241 | 105 | 3 |
| evolve:kb_evolve_smoke | 295 | 242 | 107 | 3 |
| loop:luna_gate | 294 | 239 | 105 | 3 |
| loop:opus_mini | 294 | 239 | 105 | 3 |
| loop:smoke | 294 | 239 | 105 | 3 |
| loop:soa_async_v1 | 161 | 161 | 53 | 6 |
| loop:soa_v4 | 48 | 48 | 17 | 12 |
| loop:soa_v1 | 45 | 45 | 19 | 16 |
| loop:soa_v3 | 43 | 43 | 19 | 12 |
| loop:_async_transport_real_20261004 | 35 | 35 | 13 | 10 |
| loop:_async_transport_real_v3_20261004 | 35 | 35 | 13 | 10 |
| loop:soa_v2 | 22 | 22 | 8 | 7 |
| loop:seed20_sol_opus | 20 | 20 | 7 | 5 |
| loop:sol_tool | 20 | 20 | 7 | 6 |
| loop:_smoke_soa | 19 | 19 | 7 | 6 |
| loop:_codex_smoke | 16 | 16 | 6 | 5 |
| loop:sol_codex | 16 | 16 | 6 | 5 |
| seed:seed_distill_v1 | 16 | 16 | 6 | 5 |
| loop:sol_empty | 12 | 12 | 4 | 4 |

表中“子系统内”按记录计：同一条主张若出现在几个 KB 里（例如 luna、smoke、opus 都从同一个 qwen KB 起步），每个 KB 各算一次。

同时查过、但没有 KB 主张的地方：

- `kb_lab`：实验室数据库（逐配置测量），没有主张
- `kb_lab_pool_v2`：实验室数据库（逐配置测量），没有主张
- `kb_lab_pool_v2b`：实验室数据库（逐配置测量），没有主张
- `kb_lab_pool_v3`：实验室数据库（逐配置测量），没有主张
- `swarm50/`：RL 训练与清单文件，没有 KB 主张
- `kb_science_loop/*/reports, lab_experiments.jsonl`：主张的证据（报告与实验行），不是主张本身
- `AIQ_KB_DATA/runs/d2_scaling_20261009`：定向研究报告：固定特征回归上的风险回升时间 U_rise≈0.28·n/σ²（U=ηt）。研究对象是固定特征线性回归，不是本子系统的训练流程，报告自己写明不覆盖学习特征与分类，记为子系统外
- `runs/d2_micromanage_20261009`：研究记录、发布或迁移文件，没有 KB 主张
- `runs/d2_scaling_20261009_gate_v1`：研究记录、发布或迁移文件，没有 KB 主张
- `runs/d2_scaling_20261009_invalid_smoke_20261009`：研究记录、发布或迁移文件，没有 KB 主张
- `runs/prototype_site_migration_20261010`：研究记录、发布或迁移文件，没有 KB 主张
- `runs/science_of_ai_handoff_20261009`：研究记录、发布或迁移文件，没有 KB 主张

## 3. 去重与谱系

- 4885 条主张记录（science loop 取 kb_live 或最后一个快照，含 retired；qwen kb_evolve 合并全部快照，同一 id 取最后版本；加上 seed KB）。
- 文本归一化后 904 种；按词集 Jaccard ≥ 0.6 合并近似文本后 876 条。
- qwen 的 kb_evolve 各次运行从头编号，有 40 个 id 在不同运行里指不同的主张，所以主张键用“运行:id”，不用裸 id。
- 谱系：retired 主张若写明 merged_into 且后继在同一 KB 里，算作被后继取代，不单列（18 条）；后继记录里保留它们的可执行规则。没有后继的 retired（rolled back、refuted）是它那条谱系的最后版本，照常列出。
- 没有沿 merged_from 把主张连起来：它列的是被合并的不同主张，经跨运行的文本相似传递后会把几条独立的现行主张并成一条。

## 4. 怎样判定“说的是这个子系统”

主张要预测同一数据、同一模型、同一预算下两组优化器设置谁胜（或打平）。规则对同模型对适用就算，即使它同时谈到架构不同的对，例如“Δ 差距压过架构差别”。不算的有三类：

- 固定优化器比较架构（提到优化器的 778 条里大多数是这类）；
- 架构与优化器同时不同的比较（22 条）；
- 只讲机制、不给比较的陈述。

做法：关键词预筛（提到优化器、学习率、动量、衰减、betas、位移）→ 按“比较的是什么”自动分组 → 人工逐条读。人工读了自动标为优化器比较的 182 条、标为可能混合的 30 条，以及其余带优化器线索的 305 条。读出子系统内 270 条。判定写在 `sci_kb_route.py` 的 SIGMA / MIXED 列表里，可以逐条复查。

## 5. 层次怎么定

只看 KB 自己记录的依据，不替 KB 补推导（推导路线另做，两条路线要相互独立，才能做捕获—再捕获估计）。

- **K**：只在点名的少数数据集、题目或实验上观察到；或是求解器从解题经验归纳的一般陈述，没有分轴记录支持与失败。
- **L**：按任务、学习率或预算分开记录了支持与失败的计数，边界来自观察到的反转。同一次比较里的种子计数不算扩展。
- **S**：写出推导的主张逐条检查三件事。(a) 前提写明；(b) 每个前提能用所预测关系以外的关系检验，例如只变 T 的对、lr×T 相同的对、只变衰减的对、重复实验、训练曲线；用所预测的关系在别的条件或特殊点上检验不算，把拟合出的关系换个说法当前提也不算；(c) 结论由前提推出。

推导用 PyTorch 的更新规则重算核对（`derivation_numbers`）：

- RMSprop（α=0.99、零初始化、无偏差校正）每步平均放大 F(T)：F(256)=1.469，F(1024)=1.121，F(2048)=1.06，与 KB 的 1.47 / 1.12 / 1.06 一致；前 20 步平均 3.87 倍。
- Adagrad：Σ1/√t 与 2√T 之比在 T=256 时为 0.955，2·lr·√T 成立。
- 动量 0.9 的速度从 0 起步：少走 9/T 的位移，T=256 时 3.5%，与 KB 写的 3.5% 一致。

## 6. S 候选逐条

失败原因分布：(b) 前提不独立（Δ 规则这座桥，或拟合常数） 23 条；(c) 推导有误或没有推导 10 条；(b)(c) 未检验且推导不全 6 条；(a) 前提没写出 1 条

| 主张 | 前提 | (a) | (b) | (c) | 结论 |
|---|---|---|---|---|---|
| _smoke_soa:K1001 | 每个任务都欠训练：Δ 大者胜 | 部分 | 否：前提就是所预测的排序 | 无推导 | 不是 S |
| kb_evolve_archonly:K0001 | Adagrad 每参数近似尺度不变，位移 ≈2·lr·√T 与梯度尺度无关；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| kb_evolve_archonly:K0112 |  | 否：稳态条件（梯度在 1/(1−m) 步内变化小）没有写出 | — | — | 不是 S：前提没有写出 |
| kb_evolve_archonly:K0157 | RMSprop 零初始化、无偏差校正：第 t 步步长≈lr/√(1−α^t)（梯度量级平稳）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| kb_evolve_archonly:K0207 | 偏差校正后每步位移 ≈lr，与 β₂ 无关（符号一致的梯度）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| kb_evolve_archonly:K0247 | RMSprop 累计位移 ≈ lr·T/√(1−α) | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：1/√(1−α) 只在第 1 步成立，全程平均是 F(T)（T=256 时 1.47，不是 10） | 不是 S：推导本身有误 |
| kb_evolve_archonly:K0364 | Adagrad 位移修正式 2·lr·√T/(1+0.005·T) | 是 | 否：拟合出的修正被当作前提 | 无推导 | 不是 S |
| kb_evolve_puct20_arch:K0210 | 各优化器的输出偏置位移：动量 SGD (lr/(1−m))·T·‖g‖，自适应 lr·T，SGD lr·T·‖g‖；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 部分：‖g‖ 未给定 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| kb_evolve_puct20_mixed:K0298 | β₂=0.999 时二阶矩前 200 步低估约 5 倍，步长更大 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：Adam 有偏差校正，零初始化不会让 v̂ 系统性低估 | 不是 S：推导本身有误 |
| kb_evolve_puct20_mixed:K0307 | Adagrad 有效位移 ≈ lr·T/√(T+T_w)，是 2·lr·√T 的 3–5 倍 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：给出的式子约等于 lr·√T，小于 2·lr·√T，却说是它的 3–5 倍，前后矛盾 | 不是 S：推导本身有误 |
| kb_evolve_puct20_mixed:K0310 | Adam 初期每步位移 lr/max(√v̂, ε)，放大 10–100 倍 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：Adam 的偏差校正更新是 lr·m̂/(√v̂+ε)，每坐标步长 ≲lr；文中漏掉 m̂ | 不是 S：推导本身有误 |
| kb_evolve_puct40_arch:K0012 | 偏置对齐时间 ≈1/(2·lr/(1−m)) | 部分 | 未检验 | 部分：缺 ‖h‖² 等因子 | 不是 S |
| kb_evolve_puct40_arch:K0027 | RMSprop 零初始化、无偏差校正：第 t 步步长≈lr/√(1−α^t)（梯度量级平稳）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| kb_evolve_puct40_arch:K0069 | 动量 SGD 的有效学习率与 Adagrad 衰减后的步长直接比较 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：把 SGD 的学习率直接和归一化步长比较，缺了 SGD 位移里的 \|g\| | 不是 S：推导本身有误 |
| kb_evolve_puct40_arch:K0289 | RMSprop 零初始化、无偏差校正：第 t 步步长≈lr/√(1−α^t)（梯度量级平稳）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 部分：忽略了初期放大（T=256 时 F=1.47） | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| kb_evolve_run3:K0297 | RMSprop 前约 10 步放大 1/(1−β₂)≈10 倍 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：α=0.99 时 1/(1−α)=100；按更新规则是 1/√(1−α)=10，且只在第 1 步 | 不是 S：推导本身有误 |
| kb_evolve_run3:K0305 | 曲率不连续处动量有效学习率公式高估位移 ≥2 倍 | 部分 | 未检验 | 否：断言，没有推导 | 不是 S |
| kb_evolve_smoke:K0297 | RMSprop 前 20 步平均放大只有 2–3 倍 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：按更新规则前 20 步平均放大是 3.9 倍 | 不是 S：推导本身有误 |
| luna_main:K0156 | Adagrad 累加器按 t·g² 增长（梯度量级平稳），第 t 步步长≈lr/√t；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| qwen_6ep_40eval:K0012 | 偏置对齐时间 ≈1/(2·lr/(1−m)) | 部分 | 未检验 | 部分：缺 ‖h‖² 等因子 | 不是 S |
| qwen_6ep_40eval:K0019 | Δ 相同则进展相同、Δ 大者胜（Δ 规则）；三处门限：过拟合（epochs≥16 且 params/N≥15）、MSE 大梯度、大偏移 | 是 | 否：Δ 规则就是所预测的排序 | 无推导 | 不是 S：是定律 |
| qwen_6ep_40eval:K0024 | 稳态有效学习率 lr/(1−m)；重球加速与噪声平滑使动量更好 | 部分 | 未检验 | 否：“动量在等有效学习率时更好”没有由前提推出 | 不是 S |
| qwen_6ep_40eval:K0307 | 插值门限：epochs≥16 且 params/N≥15 时测试损失随 D 呈 U 形；最优 D≈(2–5)·d*，d*=σ_y/(0.5√W_last) | 是 | 门限只能用所预测的胜负来定 | 部分：d* 有论证，门限是拟合 | 不是 S |
| qwen_6ep_40eval:K0309 | 插值门限：epochs≥16 且 params/N≥15 时测试损失随 D 呈 U 形；最优 D≈(2–5)·d*，d*=σ_y/(0.5√W_last) | 是 | 门限只能用所预测的胜负来定 | 部分：d* 有论证，门限是拟合 | 不是 S |
| qwen_6ep_40eval:K0310 | Adagrad 位移饱和：D≈2·lr/\|g_rms\|，与 T 无关 | 部分 | 未检验；需要梯度统计或参数位移的测量，现有数据只有终点 | 否：与更新规则的尺度不变性矛盾：梯度整体乘 c 时 Adagrad 的步长不变，D 不可能反比于 \|g\|；同一批 KB 的 K0001 正写… | 不是 S：推导本身有误 |
| qwen_6ep_40eval:K0311 | RMSprop 零初始化、无偏差校正：第 t 步步长≈lr/√(1−α^t)（梯度量级平稳）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：F(T) 由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| qwen_6ep_40eval:K0314 | 深层近似冻结（核区）；读出层按 df≈−2·lr_eff·‖h‖²(f−y) 收缩 | 是 | 可独立检验，但现有数据测不到 | 部分：比较结论没有推出 | 不是 S：比较这一步没有推导 |
| qwen_6ep_40eval:K0315 | 插值门限：epochs≥16 且 params/N≥15 时测试损失随 D 呈 U 形；最优 D≈(2–5)·d*，d*=σ_y/(0.5√W_last) | 是 | 门限只能用所预测的胜负来定 | 部分：d* 有论证，门限是拟合 | 不是 S |
| smoke_qwen:K0157 | RMSprop 零初始化、无偏差校正：第 t 步步长≈lr/√(1−α^t)（梯度量级平稳）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| soa_async_v1:K1001 | 同模型、饱和以下：名义累计位移 Δ 越大，进展越多 | 部分：Δ 公式写明，“饱和”没有判据 | 否：唯一的前提就是所预测的排序 | 无推导 | 不是 S：KB 自己写明这只是进展启发式 |
| soa_async_v1:K1002 | Adagrad 累加器按 t·g² 增长（梯度量级平稳），第 t 步步长≈lr/√t；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：Δ_Ada 由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| soa_async_v1:K1019 | SGD 的 Δ=lr·T·\|g\|/(1−m)；\|g\| 按任务取 0.01 / 0.03 / 0.003；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是 | 否：拟合出的常数被当作前提 | 部分：形式是推导，常数不是 | 不是 S：把拟合出的关系换个说法当作前提 |
| soa_v1:K1001 | 每个任务都欠训练：Δ 大者胜 | 部分 | 否：前提就是所预测的排序 | 无推导 | 不是 S |
| soa_v1:K1002 | Adagrad 累加器按 t·g² 增长（梯度量级平稳），第 t 步步长≈lr/√t；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：Δ_Ada 由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| soa_v1:K1019 | SGD 的 Δ=lr·T·\|g\|/(1−m)；\|g\| 按任务取 0.01 / 0.03 / 0.003；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是 | 否：拟合出的常数被当作前提 | 部分：形式是推导，常数不是 | 不是 S：把拟合出的关系换个说法当作前提 |
| soa_v1:K1026 | 动量速度从 0 开始（PyTorch 定义），梯度近似不变；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：位移亏欠由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则；这是 KB 里最接近 S 的一条：推导正确，还推出差距随 T 按… |
| soa_v1:K1035 | RMSprop 零初始化、无偏差校正：第 t 步步长≈lr/√(1−α^t)（梯度量级平稳）；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：F(T) 由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| soa_v2:K1002 | Adagrad 累加器按 t·g² 增长（梯度量级平稳），第 t 步步长≈lr/√t；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：Δ_Ada 由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| soa_v3:K1042 | Adagrad 累加器按 t·g² 增长（梯度量级平稳），第 t 步步长≈lr/√t；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是：更新规则层面的前提写明了；Δ 规则作为桥被引用 | 否：桥（Δ 规则）只能用所预测的排序检验 | 是：Δ_Ada 由更新规则推出 | 不是 S：换算是推导，排序要经过未独立检验的 Δ 规则 |
| soa_v3:K1049 | SGD 的 Δ=lr·T·\|g\|/(1−m)；\|g\| 按任务取 0.01 / 0.03 / 0.003；Δ 相同则进展相同、Δ 大者胜（Δ 规则） | 是 | 否：拟合出的常数被当作前提 | 部分：形式是推导，常数不是 | 不是 S：把拟合出的关系换个说法当作前提 |

## 7. 意外发现

1. **KB 不认自己的核心规则是理论。** K1001 的机制字段写着 nominal displacement 只是进展启发式，no theory is established；它的 phase 记录了同一配方换目标就反转胜负。
2. **推导停在换算，排序靠一座没有独立检验的桥。** Adagrad 的 2·lr·√T、RMSprop 的 F(T)、动量的 lr/(1−m) 与起步亏欠，都能由更新规则推出，数值核对一致。KB 甚至由此推出一个新预言：10 倍学习率的 Adagrad 对 Adam 在 T=256 胜、T=512 负，phase 记录证实了它。但从“位移更大”到“损失更低”这一步，KB 只用胜负计数支持。
3. **SGD 的换算常数是拟合的。** K1019 / K1049 的 |g|=0.01 / 0.03 / 0.003 由 368 个 SGD 对自适应优化器的比较拟合得到，是把拟合出的关系换个说法当前提。
4. **有 8 条推导本身是错的，都来自 qwen 起步的 KB**（kb_evolve_archonly:K0247, kb_evolve_puct20_mixed:K0298, kb_evolve_puct20_mixed:K0307, kb_evolve_puct20_mixed:K0310, kb_evolve_puct40_arch:K0069, kb_evolve_run3:K0297, kb_evolve_smoke:K0297, qwen_6ep_40eval:K0310）。例如 Adagrad 位移“反比于 |g| 并饱和”（与更新规则的尺度不变性矛盾，同一批 KB 的另一条正写着与梯度尺度无关），Adam“初期放大 10–100 倍”（漏了 m̂），β₂=0.999 的二阶矩“低估 5 倍”（忽略了偏差校正），RMSprop 放大写成 1/(1−β₂)。
5. **同一组配对，三条 KB 主张给出三种预言。** 动量 SGD 与 lr/(1−m) 相同的普通 SGD：luna_main:K0024 预言动量胜，实验室 123 对里判错 91%；luna_main:K0112 预言打平，实验室 123 对判平，98% 确实不显著；soa_v1:K1026 预言普通 SGD 胜，实验室 132 对里判错 8%。数据支持“普通 SGD 略好、差距在噪声内”。
6. **没有留出集可用。** v1.5 有 498 题出现在某个 KB 的语料（seed 可见集 490 题、corpus_v2 469 题、v15_minus_eval100 475 题）或 qwen 的解题记录里；剩下 2 题里没有同模型对。实验室行就是 science loop 自己跑的实验。所以下表的误差只说明规则对来源数据的拟合程度。

## 8. 全部子系统主张

误差 = 给出胜负的对里判错的比例；判平只对“二阶旋钮是噪声”这类规则出现。“—”表示规则在该数据上不作答或不可执行。

| 主张 | 状态 | 层次 | 预测的关系 | 基准 n / 误差 | 实验室 n / 误差 | 层次理由 | 文本 |
|---|---|---|---|---|---|---|---|
| soa_async_v1:K1070 | live（1 个 KB） | L | 权重衰减 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Classification MLPs,SAME net/lr,T<=1024: RMSprop is an underfitting-regime prefe… |
| soa_async_v1:K1101 | live（1 个 KB） | L | 权重衰减 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Within tested default-init nonresidual width32/depth3 SiLU MLPs with LN[T,T,T], … |
| soa_async_v1:K1142 | live（1 个 KB） | L | 权重衰减 | 2 / 0% | 8 / 0% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Tabular CE,T256,matched network: Adagrad at10x Adam/AdamW lr is competitive,not … |
| soa_v1:K1040 | live（5 个 KB） | L | 权重衰减 | — | 87 / 5% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Spiral coupled weight decay (two-spiral CE, 1-1.5 turns, Adam-family lr 1e-3-3e-… |
| soa_v3:K1046 | retired / dropped：rolled back | L | 权重衰减 | 判平 14（100% 不显著） | 87 / 5% · 判平 566（81% 不显著） | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Second-order optimizer knobs are noise at these budgets: weight decay 0-1e-3 cha… |
| _smoke_soa:K1001 | live（1 个 KB） | L（S 候选） | 按 Δ 排序 | 492 / 10% | 6719 / 25% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | CORE RULE (all families, T=256-2048; every task is under-trained): rank choices … |
| qwen_6ep_40eval:K0019 | live（1 个 KB） | L（S 候选） | 按 Δ 排序 | — | — | 把 Δ 规则写成定律，并写出三处观察到的反转门限（过拟合、MSE 大梯度、大偏移） | Identical-architecture optimizer rankings follow cumulative per-parameter displa… |
| soa_async_v1:K1001 | live（1 个 KB） | L（S 候选） | 按 Δ 排序 | 492 / 10% | 6719 / 25% | 跨任务族回测（支持 123.75、失败 45.25），phase 记录了反转；但边界“饱和以下”没有… | Optimizer comparison,SAME model/data/budget: use Delta as a progress proxy ONLY … |
| soa_v1:K1037 | retired / dropped：rolled back | L | 按 Δ 排序 | 32 / 0% | 553 / 13% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Spiral OPTIMIZER-ONLY questions (same MLP, optimizer/lr differ): the core Δ rule… |
| soa_v1:K1043 | retired / dropped：rolled back | L | 按 Δ 排序 | 27 / 0% | 406 / 10% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Spiral Δ dominance (two-spiral classification, 1-3 turns, T=256-1024, Adam/AdamW… |
| sol_empty:K0011 | live（1 个 KB） | L | 按 Δ 排序 | 369 / 9% | 4558 / 26% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Optimizer-varying questions on MLP families (spiral, XOR, tabular, uni/multivari… |
| soa_async_v1:K1002 | live（1 个 KB） | L（S 候选） | 类型换算 | 109 / 5% | 542 / 20% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Adagrad vs Adam/AdamW on the SAME architecture/budget: Adagrad Δ≈2·lr·√T, so com… |
| soa_async_v1:K1019 | live（1 个 KB） | L（S 候选） | 类型换算 | 211 / 8% | 1495 / 11% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | SGD vs ADAPTIVE optimizers: family-calibrated Delta is a proxy,NOT a plain-vs-mo… |
| soa_v1:K1002 | live（1 个 KB） | L（S 候选） | 类型换算 | 147 / 7% | 716 / 26% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Adagrad vs Adam/AdamW/RMSprop (any family): compare Δ_Ada=2·lr·√T with Δ_Adam=lr… |
| soa_v1:K1019 | live（8 个 KB） | L（S 候选） | 类型换算 | 211 / 8% | 1495 / 11% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | SGD gradient-scale calibration for the Δ rule (K1001 assumes raw \|g\|≈0.01), fit … |
| soa_v1:K1035 | retired / dropped：rolled back | L（S 候选） | 类型换算 | 6 / 0% | 226 / 37% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | RMSprop vs Adam/AdamW at the SAME lr and T (same arch): PyTorch RMSprop (alpha=0… |
| soa_v3:K1042 | retired / dropped：rolled back | L（S 候选） | 类型换算 | 376 / 7% | 2772 / 18% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Optimizer conversion for the Δ rule (K1001). Adagrad: steps decay ~lr/√t, so it … |
| soa_v3:K1049 | retired / dropped：rolled back | L（S 候选） | 类型换算 | 211 / 8% | 1495 / 11% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | SGD conversion for the Δ rule (K1001; fit on 368 SGD-vs-adaptive pairs): Δ_SGD =… |
| soa_v3:K1054 | retired / dropped：rolled back | L | 类型换算 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Adagrad vs Adam/AdamW lr=3e-3 on MSE REGRESSION (same architecture, 256 rows, bs… |
| soa_v4:K1057 | live（3 个 KB） | L | 类型换算 | 13 / 8% | 264 / 31% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | UNIVARIATE regression, same MLP, Adam/AdamW vs RMSprop: Adam-family wins at almo… |
| soa_v4:K1063 | live（3 个 KB） | L | 类型换算 | 4 / 75% | 71 / 39% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | UNIVARIATE regression, RMSprop vs RMSprop (same MLP, different lr): the LOWER RM… |
| soa_v4:K1070 | live（3 个 KB） | L | 类型换算 | — | 61 / 18% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | CLASSIFICATION MLPs (spiral/XOR/tabular), RMSprop vs Adam/AdamW at the SAME lr, … |
| sol_empty:K0005 | live（1 个 KB） | L | 按学习率排序 | 23 / 9% | 1121 / 20% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Fixed-law bigram LM, identical architecture for all choices, cross-entropy, 256 … |
| soa_v1:K1003 | live（12 个 KB） | L | 动量 | 194 / 8% | 1366 / 17% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | SGD (all families): plain SGD (momentum 0) at lr≤3e-3 is nearly frozen (wins ~8%… |
| soa_async_v1:K1079 | live（1 个 KB） | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Gaussian XOR CE,1024 rows,T256: RMSprop .003 vs Adagrad .003 favors RMSprop with… |
| soa_v1:K1004 | live（12 个 KB） | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | LR-ceiling exceptions to the Δ rule: (a) RMSprop lr≥1e-3 on MSE regression (no m… |
| soa_v1:K1021 | live（3 个 KB） | L | Δ 窗口与过冲 | 94 / 4% | 492 / 37% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Synthetic TABULAR classification Δ overshoot (lab, 10-seed, within-set, mostly T… |
| soa_v1:K1028 | retired / dropped：rolled back | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Bigram LM d_model=128 transformer trap: a post-LN transformer with d_model 128 t… |
| soa_v1:K1030 | retired / dropped：rolled back | L | Δ 窗口与过冲 | 5 / 40% | 88 / 36% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | XOR classification Δ OVERSHOOT (Adam/AdamW/RMSprop, T=512-1024, input_dim 2-16):… |
| soa_v1:K1032 | live（2 个 KB） | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Tabular LONG-budget overfit (synthetic tabular classification, 1024 train rows, … |
| soa_v1:K1034 | retired / dropped：rolled back | L | Δ 窗口与过冲 | 1 / 100% | 75 / 28% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | XOR LONG-budget Δ overshoot (xor_classification, 1024 train rows, total_samples_… |
| soa_v1:K1044 | retired / dropped：rolled back | L | Δ 窗口与过冲 | 8 / 12% | 243 / 26% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | XOR Δ overshoot (Gaussian XOR with 2-4 input features, same model, Adam/AdamW/RM… |
| soa_v3:K1041 | retired / dropped：rolled back | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Two-spiral / XOR classification (chance CE=ln2): when optimizer, lr or T differ,… |
| soa_v3:K1045 | retired / dropped：rolled back | L | Δ 窗口与过冲 | 5 / 60% | 74 / 19% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Tabular LONG budget (synthetic tabular classification, 1024 rows, 16 features, t… |
| soa_v3:K1048 | retired / dropped：rolled back | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Tabular LONG budget (synthetic tabular classification, 1024 rows, 16 features, t… |
| soa_v3:K1053 | retired / dropped：rolled back | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | Tabular LONG budget (synthetic tabular classification, 1024 train rows, total_sa… |
| soa_v3:K1055 | live（4 个 KB） | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | UNIVARIATE regression, WIDE multi-layer MLP (width≥128, depth≥2), choices differ… |
| soa_v4:K1071 | live（3 个 KB） | L | Δ 窗口与过冲 | 58 / 9% | 1512 / 8% | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | BIGRAM LM learning-rate sweet spot (same model; choices differ in optimizer/lr/T… |
| soa_v4:K1073 | live（3 个 KB） | L | Δ 窗口与过冲 | — | — | 按任务、学习率或预算分开记录了支持与失败的计数（lab / corpus 成对比较），边界来自观察到… | TABULAR classification Δ sweet spot vs EPOCHS (synthetic, 1024 rows; E=total_sam… |
| kb_evolve_archonly:K0165 | live（14 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short optimization budgets (T <= ~300 steps) with Adam-family optimizers, … |
| kb_evolve_archonly:K0207 | live（14 个 KB） | K（S 候选） | β2 | — | 46 / 13% | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | In sign-normalized adaptive optimizers (Adam or AdamW), reducing second-moment d… |
| kb_evolve_archonly:K0241 | live（14 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On complex non-linear multivariate regression under moderate step budgets (T ~ 5… |
| kb_evolve_archonly:K0247 | live（14 个 KB） | K（S 候选） | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On short step budgets (T <= ~256 steps) fitting complex non-linear decision boun… |
| kb_evolve_archonly:K0250 | live（14 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On complex multivariate regression tasks with non-linear multiplicative and osci… |
| kb_evolve_puct20_mixed:K0015 | live（13 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under tight step budgets (T <= ~500 steps) at small Adam/AdamW learning rates (l… |
| kb_evolve_puct20_mixed:K0198 | live（14 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When fitting complex non-linear decision boundaries under short step budgets (T … |
| kb_evolve_puct20_mixed:K0298 | live（1 个 KB） | K（S 候选） | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On zero-linear-signal tasks (XOR), Adam β₂=0.999 yields lower loss than β₂=0.95 … |
| kb_evolve_run1:K0298 | live（1 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | At AdamW lr≤1e-4 with T≤512 on tasks requiring symmetry breaking (e.g., XOR), be… |
| kb_evolve_run3:K0297 | live（1 个 KB） | K（S 候选） | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Adam with bias-corrected second moments (β₂=0.999) achieves ≥2× faster MSE reduc… |
| kb_evolve_smoke:K0296 | live（1 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On tasks requiring formation of multiplicative feature interactions (e.g., XOR) … |
| kb_evolve_smoke:K0299 | live（1 个 KB） | K | β2 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under T≤512-step budgets on low-complexity tasks (bigram LM, 2-feature XOR), the… |
| soa_async_v1:K1096 | live（1 个 KB） | K | β2 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Spiral CE,d4/w24 residual default LeakyReLU,LN[1,1,0,0],lr.003,T256,bs16: Adam(.… |
| soa_async_v1:K1111 | live（1 个 KB） | K | β2 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Two-input nested-oscillatory mvar_0eb811/133302,plain d3/w256 Leaky(.01),LN[T,F,… |
| soa_async_v1:K1147 | live（1 个 KB） | K | β2 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Dim16 piecewise CE,1024/2048 rows,default plain d2w192 Leaky(.01),LN10,T2048/b16… |
| kb_evolve_archonly:K0229 | live（14 个 KB） | K | 权重衰减 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On continuous MSE regression tasks with large constant target offsets under mode… |
| kb_evolve_puct20_arch:K0022 | live（14 个 KB） | K | 权重衰减 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) with identical architectures and loss… |
| kb_evolve_puct40_arch:K0011 | live（14 个 KB） | K | 权重衰减 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) with undersized base learning rates, … |
| qwen_6ep_40eval:K0173 | retired / dropped：Merged into … | K | 权重衰减 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Two ways extra movement hurts classification CE at T<=512. (1) Init logit varian… |
| soa_async_v1:K1005 | live（1 个 KB） | K | 权重衰减 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Matched-network/data optimizer recipes: pooled decay/beta counts do NOT supply a… |
| soa_async_v1:K1094 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | For dimension16 tabular CE, default-init nonresidual depth4/width256 GELU LN[T,F… |
| soa_async_v1:K1095 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | For dimension16 piecewise stabcls_06550f CE, default-init depth4/width256 nonres… |
| soa_async_v1:K1102 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | ONLY V24/L16,post-LN d128x4/heads4/FF64,T256/b32: Adagrad.0001/wd0 beats RMSprop… |
| soa_async_v1:K1127 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | One-turn allowed05d021/06bcc2,train1024/test2048,T2048,bs16,default init:d3w48 p… |
| soa_async_v1:K1128 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Three-turn allowed0717d7/0cfa0b,d3w48 plainSiLU/all3LN,Adam(.9,.95),lr.001,bs16,… |
| soa_async_v1:K1130 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | 3-turn spiral CE,1024train/2048test,default plain d3w48 SiLU/all3LN,Adam(.9,.95)… |
| soa_async_v1:K1146 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Dim16 sparse CE,plain d2w192 Leaky(.01)/LN10,T2048/b16: coupled Adam.003,NOT Ada… |
| soa_async_v1:K1165 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Bigram V24/L16,default post-LN GELU d128x4/heads4/FF64,RMSprop wd1e-5,T256/b32: … |
| soa_async_v1:K1172 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | XOR16 CE,plain d3w256 SiLU/all3LN,T512/b64,defaultinit: Adam(.9,.999)lr.0001/wd.… |
| soa_async_v1:K1173 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | 4-input symbolic MSE,default plain ReLU d4w192/LN0100,T512/b16: Adagrad .003/wd0… |
| soa_async_v1:K1175 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | V24/L16 bigram,alpha.8,default postLN GELU d128x4/heads4,T256/b32: FF64->256 rev… |
| soa_async_v1:K1179 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Univariate y=cos(2*pi*x)*(cos(2*pi*x)-3),plain LeakyReLU(.01)d1w256/oneLN,defaul… |
| soa_async_v1:K1187 | live（1 个 KB） | K | 权重衰减 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Matched width16/depth1 residual GELU/no-LN/default-init/plain-MSE,T256/bs16,inpu… |
| soa_async_v1:K1197 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Smooth-additive16 CE,default residual d2w192 ReLU/zeroLN,T1024/b32: exact Adagra… |
| soa_async_v1:K1198 | live（1 个 KB） | K | 权重衰减 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Gaussian XOR8,default plain d2w48 GELU/LN10,Adagrad1e-4,T512/b32: wd0 narrowly b… |
| soa_v1:K1005 | live（12 个 KB） | K | 权重衰减 | 判平 14（100% 不显著） | 87 / 5% · 判平 566（81% 不显著） | 只在点名的少数数据集、题目或实验上观察到 | Second-order optimizer knobs are noise at these budgets: weight decay 0-1e-3 cha… |
| soa_v1:K1045 | live（1 个 KB） | K | 权重衰减 | — | 27 / 15% | 只在点名的少数数据集、题目或实验上观察到 | beta2 (β2) 0.95 vs 0.999 on two-spiral classification (Adam/AdamW twins: same mo… |
| kb_evolve_archonly:K0098 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For first-order bigram next-token prediction under short step budgets (T <= ~512… |
| kb_evolve_archonly:K0105 | live（13 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For first-order bigram next-token prediction with small vocabularies (V ~ 32) un… |
| kb_evolve_archonly:K0115 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For first-order bigram next-token prediction under moderate step budgets (T ~ 10… |
| kb_evolve_archonly:K0128 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On small transformer language models (d_model <= 128, depth <= 4) trained under … |
| kb_evolve_archonly:K0129 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under small adaptive learning rates (Adam/AdamW lr <= ~3e-5) and short-to-modera… |
| kb_evolve_archonly:K0172 | live（13 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When cumulative parameter displacement under sign-normalized adaptive optimizers… |
| kb_evolve_archonly:K0285 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On multiplicative interaction classification tasks (e.g., s(x) = a·x_i·x_j + b·x… |
| kb_evolve_puct20_mixed:K0310 | live（1 个 KB） | K（S 候选） | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For AdamW and Adam optimizers, effective per-step parameter displacement in earl… |
| kb_evolve_puct20_mixed:K0311 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For bigram LM with vocab V ≤ 64 under T ≤ 1024 steps, cumulative parameter displ… |
| kb_evolve_puct40_arch:K0043 | live（13 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under tight step budgets (T <= ~256 steps), structural architectural features (s… |
| kb_evolve_puct40_arch:K0200 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Fitting target functions containing high-frequency components (such as 2*pi sinu… |
| kb_evolve_run1:K0295 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On XOR classification (zero linear signal) with T=1024 steps, AdamW lr=3e-3 (per… |
| kb_evolve_run3:K0295 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For XOR and degree-2 interaction tasks, cumulative parameter displacement lr×T a… |
| kb_evolve_run3:K0302 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For first-order lookup tasks (e.g., bigram next-token prediction), when one opti… |
| kb_evolve_run3:K0312 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For 2-layer GRU architectures on bigram LM (V=32, T=1024), RMSprop at lr=3e-4 (c… |
| luna_main:K0019 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | With identical architectures and short budgets (T≈500–2000), cumulative paramete… |
| luna_main:K0034 | live（14 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short optimization budgets (T <= ~256 steps) on small language modeling ta… |
| qwen_6ep_40eval:K0295 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Spiral/XOR CE (T<=2048): capacity never binds (2-4 layers, width>=128). Ask WHIC… |
| qwen_6ep_40eval:K0312 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | First-order bigram/Markov LM (V<=48, ~800 rows, T<=1024). D=lr*T (Adam), lr*T*F(… |
| soa_async_v1:K1012 | live（1 个 KB） | K | 按 Δ 排序 | 366 / 8% | 3948 / 20% | 只在点名的少数数据集、题目或实验上观察到 | Mixed-optimizer/architecture rankings: close Delta estimates are NOT winner evid… |
| soa_async_v1:K1071 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Bigram LM,SAME model/data only: neither lower RMSprop rate nor Delta .1-.3 is a … |
| soa_async_v1:K1073 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Tabular CE,1024rows,SAME network/data: Delta optima are recipe/batch-specific,no… |
| soa_async_v1:K1103 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | On two allowed five-input nested-oscillatory datasets mvar_05883d/06497c, q_74d1… |
| soa_async_v1:K1129 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | V32/L24/alpha1.4 bigram plain CE,T256/b32,default GRU64x2 ONLY: RMSprop .001/wd0… |
| soa_async_v1:K1138 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Default-init bigram V48/L24,train800/test200,GRU64x1,plainCE,b16: B=SGD.001,m.9,… |
| soa_async_v1:K1176 | live（1 个 KB） | K | 按 Δ 排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | V24/L16 bigram,default postLN GELU d128x4/heads4/FF64,Adagrad/wd0,T256/b32: on a… |
| soa_v1:K1001 | live（11 个 KB） | K（S 候选） | 按 Δ 排序 | 492 / 10% | 6719 / 25% | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | CORE RULE (all families, T=256-2048 steps; every task here is under-trained): ra… |
| sol_empty:K0006 | retired / dropped：rolled back | K | 按 Δ 排序 | 369 / 9% | 4558 / 26% | 只在点名的少数数据集、题目或实验上观察到 | Non-bigram MLP tasks (uni/multivariate regression; tabular, XOR, spiral classifi… |
| kb_evolve_archonly:K0001 | live（13 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Adagrad initialized with zero accumulators is approximately scale-invariant per … |
| kb_evolve_archonly:K0019 | live（12 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under fixed short step budgets (T ~ 500-2000 steps) with identical architectures… |
| kb_evolve_archonly:K0044 | live（13 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with zero linear baseline signal (e.g., XOR-like problem… |
| kb_evolve_archonly:K0079 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~512 steps) on discrete bigram lookup tasks with … |
| kb_evolve_archonly:K0085 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For continuous MSE regression on smooth symbolic targets under moderate step bud… |
| kb_evolve_archonly:K0086 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under moderate step budgets (T ~ 2000 steps) with severely undersized base learn… |
| kb_evolve_archonly:K0097 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On high-curvature non-linear classification tasks (such as multi-turn spirals) u… |
| kb_evolve_archonly:K0100 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For first-order bigram next-token prediction under short step budgets (T <= ~256… |
| kb_evolve_archonly:K0116 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For complex non-linear decision boundaries (such as multi-turn spirals) under sh… |
| kb_evolve_archonly:K0117 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For non-linear feature interaction classification tasks (such as XOR) under mode… |
| kb_evolve_archonly:K0120 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On continuous multivariate regression tasks with non-linear target interactions … |
| kb_evolve_archonly:K0121 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On non-linear classification tasks (such as XOR) under short step budgets (T <= … |
| kb_evolve_archonly:K0132 | live（13 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For non-linear classification tasks with high-curvature decision boundaries (suc… |
| kb_evolve_archonly:K0157 | live（11 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | In PyTorch RMSprop with zero-initialized average squared gradient accumulators (… |
| kb_evolve_archonly:K0189 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On balanced binary classification tasks with cross-entropy loss, initial output-… |
| kb_evolve_archonly:K0201 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under tight step budgets (T <= ~256 steps) on continuous MSE regression tasks wi… |
| kb_evolve_archonly:K0211 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For bigram next-token prediction under moderate step budgets (T ~ 2000 steps), A… |
| kb_evolve_archonly:K0228 | live（13 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with zero linear baseline signal (such as XOR) under sho… |
| kb_evolve_archonly:K0235 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On bigram language modeling tasks under moderate step budgets (T ~ 512 steps), l… |
| kb_evolve_archonly:K0260 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On non-linear classification tasks requiring multiplicative feature interactions… |
| kb_evolve_archonly:K0263 | live（13 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with zero linear baseline signal (such as XOR) under sho… |
| kb_evolve_archonly:K0271 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On non-linear classification tasks with high-curvature decision boundaries (such… |
| kb_evolve_archonly:K0277 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On highly non-linear binary classification tasks (such as two-spirals) under sho… |
| kb_evolve_archonly:K0279 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) with identical model architectures, c… |
| kb_evolve_archonly:K0281 | live（13 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~512 steps) on under-trained non-linear classific… |
| kb_evolve_archonly:K0284 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For 1-D regression targets with amplitude \|y\| ~ 2 under Adagrad at lr ~ 3e-3 and… |
| kb_evolve_archonly:K0286 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T ~ 256 steps) on complex multivariate non-linear regr… |
| kb_evolve_archonly:K0364 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under Adagrad with lr≤1e-3 and T≥512, the naive displacement formula D_eff=2×lr×… |
| kb_evolve_puct20_arch:K0138 | live（13 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps), Adagrad initialized with zero accumu… |
| kb_evolve_puct20_arch:K0156 | live（12 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When comparing Adagrad (learning rate lr_g, zero-initialized accumulator) agains… |
| kb_evolve_puct20_arch:K0262 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On continuous multivariate regression tasks combining periodic and linear target… |
| kb_evolve_puct20_mixed:K0070 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When fitting continuous regression targets with large mean offsets (\|E[y]\| >> 1)… |
| kb_evolve_puct20_mixed:K0297 | live（1 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | RMSprop at lr=3e-3 with zero-initialized accumulator produces effective step siz… |
| kb_evolve_puct20_mixed:K0303 | live（1 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On univariate smooth regression with absolute mean offset \|μ\|>2 and T≤256 steps,… |
| kb_evolve_puct20_mixed:K0307 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | The standard Adagrad displacement estimate 2·lr·√T underestimates effective disp… |
| kb_evolve_puct40_arch:K0027 | live（12 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | In adaptive optimizers lacking squared-gradient bias correction (such as standar… |
| kb_evolve_puct40_arch:K0135 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On token-level sequence or bigram language modeling tasks under short step budge… |
| kb_evolve_puct40_arch:K0166 | live（12 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On bigram next-token prediction tasks using transformer language models under sh… |
| kb_evolve_puct40_arch:K0289 | live（12 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | RMSprop (zero-initialized squared-gradient accumulator, alpha=0.99) behaves like… |
| kb_evolve_smoke:K0297 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | RMSprop's zero-initialized accumulator amplification provides an effective step-… |
| luna_main:K0152 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On continuous MSE regression targets with small batch sizes (batch size <= 16) u… |
| luna_main:K0156 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For identical architectures, zero-accumulator Adagrad (lr_g) versus sign-normali… |
| luna_main:K0239 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On complex multivariate regression targets with multiplicative and non-linear in… |
| luna_main:K0248 | live（14 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On zero-linear-signal classification tasks (such as XOR) under short step budget… |
| luna_main:K0281 | live（1 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Within Adagrad on short-budget under-trained nonlinear classification, learning-… |
| luna_main:K0301 | live（1 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | In standard PyTorch RMSprop (alpha≈0.99, eps≈1e-8, zero-initialized squared-grad… |
| luna_main:K0306 | live（1 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For identical unnormalized MLPs on smooth low-dimensional MSE regression with ba… |
| luna_main:K0320 | live（1 个 KB） | K | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For balanced smooth-additive tabular classification with identical residual MLPs… |
| qwen_6ep_40eval:K0012 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | MSE regression with a non-zero mean: initial output ~0, so early MSE reduction i… |
| qwen_6ep_40eval:K0124 | live（1 个 KB） | K | 类型换算 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Zero-linear-signal classification (XOR, multi-turn spiral, unnormalized 2-D): ra… |
| qwen_6ep_40eval:K0296 | live（1 个 KB） | K | 类型换算 | 3 / 0% | 110 / 3% | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Continuous MSE regression, T<=500 AND (epochs<=12 or rho<=15): Adam/AdamW lr~1e-… |
| qwen_6ep_40eval:K0310 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Adagrad (zero accumulator) displacement SATURATES: D~2*lr/\|g_rms\| with \|g_rms\| m… |
| qwen_6ep_40eval:K0311 | live（1 个 KB） | K（S 候选） | 类型换算 | 67 / 21% | 741 / 34% | 只在点名的少数数据集、题目或实验上观察到 | PyTorch RMSprop (zero-initialized square_avg, alpha=0.99, no bias correction) ha… |
| smoke_qwen:K0157 | live（1 个 KB） | K（S 候选） | 类型换算 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | PyTorch RMSprop with a zero-initialized squared-gradient accumulator (alpha~0.99… |
| soa_async_v1:K1003 | live（1 个 KB） | K | 类型换算 | 134 / 5% | 711 / 3% | 只在点名的少数数据集、题目或实验上观察到 | Normalized MLPs or bigram LMs,SGD vs adaptive BELOW saturation: plain SGD is oft… |
| soa_async_v1:K1134 | live（1 个 KB） | K | 类型换算 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | q_0fe519 exact recipe transfer to allowed XOR16 xorcls_0dd7d2 (active2/7, point … |
| soa_v2:K1002 | live（11 个 KB） | K（S 候选） | 类型换算 | 109 / 5% | 542 / 20% | 有少量比较计数，没有沿条件轴系统扩展 | Adagrad conversion (any family): because Adagrad's step decays as lr/√t, it matc… |
| kb_evolve_archonly:K0270 | live（13 个 KB） | K | 按学习率排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On sequence modeling tasks involving simple transition dynamics (such as learnin… |
| kb_evolve_puct20_arch:K0067 | live（14 个 KB） | K | 按学习率排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For complex non-linear decision boundaries (such as multi-turn spirals) under sh… |
| kb_evolve_puct40_arch:K0016 | live（13 个 KB） | K | 按学习率排序 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When fitting continuous MSE regression targets under short step budgets (T <= ~5… |
| soa_async_v1:K1057 | live（1 个 KB） | K | 按学习率排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Univariate MSE,default residual d4w192 GELU/LN0101,b32,wd0,Adam(.9,.95): at SAME… |
| soa_async_v1:K1098 | live（1 个 KB） | K | 按学习率排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Spiral CE, allowed splits05d021/06bcc2 (1turn),016ff9/01793b (2turn), T256 bs16:… |
| soa_async_v1:K1149 | live（1 个 KB） | K | 按学习率排序 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Bigram V32/L12/alpha.8,train800/test200,default post-LN GELU d128/layers4/heads2… |
| soa_async_v1:K1150 | live（1 个 KB） | K | 按学习率排序 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Bigram V32/L12/alpha1,train800/test200,default post-LN GELU d128/layers4/heads2/… |
| soa_async_v1:K1152 | live（1 个 KB） | K | 按学习率排序 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | GRU64x2,RMSprop .001 vs AdamW .0003 ONLY: budget reversal does NOT favor RMSprop… |
| sol_empty:K0003 | live（1 个 KB） | K | 按学习率排序 | 4 / 0% | 9 / 11% | 有少量比较计数，没有沿条件轴系统扩展 | Two-spiral classification, identical MLP architecture and cross-entropy, matched… |
| kb_evolve_archonly:K0014 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On multi-layer LayerNorm transformers trained with cross-entropy loss over vocab… |
| kb_evolve_archonly:K0020 | live（13 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under tight step budgets (T <= ~256 steps) on language modeling tasks over vocab… |
| kb_evolve_archonly:K0112 | live（14 个 KB） | K（S 候选） | 动量 | 判平 1（100% 不显著） | 判平 123（98% 不显著） | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | SGD with momentum m at learning rate lr achieves an effective learning rate of l… |
| kb_evolve_archonly:K0213 | live（12 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) on continuous MSE regression tasks, s… |
| kb_evolve_archonly:K0242 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On complex non-linear decision boundaries under moderate step budgets (T ~ 1000 … |
| kb_evolve_archonly:K0243 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For stationary bigram next-token prediction under moderate step budgets (T ~ 100… |
| kb_evolve_puct20_arch:K0210 | live（14 个 KB） | K（S 候选） | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) on MSE regression tasks with a non-ze… |
| kb_evolve_puct20_mixed:K0039 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On small-batch (batch size <= 32) tabular classification tasks trained over shor… |
| kb_evolve_puct20_mixed:K0141 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When fitting continuous regression targets with large constant offsets (\|E[y]\| >… |
| kb_evolve_puct40_arch:K0012 | live（13 个 KB） | K（S 候选） | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When fitting continuous targets with a large constant offset \|E[y]\| >> output_in… |
| kb_evolve_puct40_arch:K0069 | live（12 个 KB） | K（S 候选） | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~1000 steps) on continuous MSE regression targets… |
| kb_evolve_run3:K0300 | live（1 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When momentum SGD (m=0.9) and an adaptive optimizer (Adagrad or RMSprop) yield e… |
| kb_evolve_run3:K0308 | live（1 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | SGD with momentum m=0.9 breaks symmetry on zero-linear-baseline multiplicative-i… |
| luna_main:K0066 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On non-linear feature interaction tasks (such as XOR) under short step budgets (… |
| luna_main:K0158 | live（14 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When training Transformer architectures from scratch under short step budgets (T… |
| luna_main:K0309 | live（1 个 KB） | K | 动量 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On standard-normal XOR classification with identical shallow residual MLPs (T≤25… |
| qwen_6ep_40eval:K0024 | live（13 个 KB） | K（S 候选） | 动量 | 1 / 100% | 123 / 91% | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For SGD with momentum m, the steady-state effective learning rate scales to lr_e… |
| soa_async_v1:K1119 | live（1 个 KB） | K | 动量 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | 16-feature XOR,xorcls_0dd7d2,default plain d3/w24 GELU,LN[T,F,T],bs16,lr.001,wd0… |
| soa_async_v1:K1133 | live（1 个 KB） | K | 动量 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Matched normalized d3w32 SiLU/LN[T,T,T], wd0, bs32 controls on allowed mvar_0588… |
| soa_async_v1:K1155 | live（1 个 KB） | K | 动量 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Symbolic uniform[0,1] plain-MSE, residual d2w64/LN[FT], default init,T256/bs32/w… |
| soa_v1:K1026 | retired / dropped：rolled back | K（S 候选） | 动量 | 2 / 0% | 132 / 8% | 受控对比 7/7，但只有 3 个实验集 | Momentum warm-up tie-breaker (any family, T=256-1024, same model/loss): plain SG… |
| kb_evolve_archonly:K0010 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Learning non-linear feature interactions (such as XOR) requires moving weights s… |
| kb_evolve_archonly:K0023 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under severely constrained sample or step budgets (total samples < ~10,000, T <=… |
| kb_evolve_archonly:K0102 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On continuous multivariate regression tasks with complex non-linear target inter… |
| kb_evolve_archonly:K0106 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On multi-turn spiral classification tasks under moderate step budgets (T ~ 2048 … |
| kb_evolve_archonly:K0119 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under tight step budgets (T <= ~256 steps) on XOR-like classification tasks requ… |
| kb_evolve_archonly:K0124 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with zero linear baseline signal (such as XOR), raw outp… |
| kb_evolve_archonly:K0144 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For O(1)-magnitude regression targets under short step budgets (T <= ~256 steps)… |
| kb_evolve_archonly:K0202 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When training sequence models with softmax cross-entropy over moderate vocabular… |
| kb_evolve_archonly:K0209 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | RMSprop at elevated learning rates (lr ~ 3e-3) on a single-layer post-LayerNorm … |
| kb_evolve_archonly:K0217 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Small post-LayerNorm transformers (d_model <= 32, depth <= 4) remain numerically… |
| kb_evolve_archonly:K0218 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under unaccelerated SGD at small learning rates (lr <= ~3e-4) and short step bud… |
| kb_evolve_archonly:K0231 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On complex non-linear classification tasks (such as multi-turn spirals) under mo… |
| kb_evolve_archonly:K0265 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with zero linear baseline signal and high-curvature deci… |
| kb_evolve_archonly:K0269 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For causal language modeling of fixed Markov transition dynamics under short ste… |
| kb_evolve_archonly:K0272 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On zero-linear-signal classification tasks requiring multiplicative feature inte… |
| kb_evolve_puct20_arch:K0009 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under tight step budgets (T <= ~500 steps) where candidates share identical arch… |
| kb_evolve_puct20_arch:K0031 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short optimization budgets (T <= ~256 steps), the choice of optimizer and … |
| kb_evolve_puct20_arch:K0101 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On complex multivariate regression tasks with non-linear multiplicative and osci… |
| kb_evolve_puct20_arch:K0139 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On small (width <= 32) residual MLPs equipped with LayerNorm trained under short… |
| kb_evolve_puct20_arch:K0178 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When training Transformer language models with small batch sizes (batch size <= … |
| kb_evolve_puct20_arch:K0187 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) on residual MLPs with LayerNorm where… |
| kb_evolve_puct20_mixed:K0174 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~500 steps) on token-embedding models, unaccelera… |
| kb_evolve_puct20_mixed:K0184 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For highly curved decision boundaries (such as 2-D spiral classification) with u… |
| kb_evolve_puct20_mixed:K0212 | live（13 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with zero linear baseline signal (such as XOR), Adam or … |
| kb_evolve_puct20_mixed:K0296 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For tasks with effective output dimensionality ≤ 500 parameters (bigram LM with … |
| kb_evolve_puct20_mixed:K0301 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On smooth additive regression or classification targets with T≤256 steps, there … |
| kb_evolve_puct40_arch:K0073 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) on continuous multivariate regression… |
| kb_evolve_puct40_arch:K0227 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short step budgets (T <= ~256 steps) on complex non-linear classification … |
| kb_evolve_run1:K0297 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For token-embedding language models trained with AdamW at T≤512, there exists a … |
| kb_evolve_run3:K0032 | live（11 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For 2-D two-spiral classification tasks (~2 turns), standard MLPs with 2-4 hidde… |
| kb_evolve_run3:K0191 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | When training on unnormalized inputs of large scale (radius up to ~10) without i… |
| kb_evolve_run3:K0305 | live（1 个 KB） | K（S 候选） | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On loss landscapes with curvature discontinuities (piecewise-linear boundaries, … |
| kb_evolve_run3:K0316 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For multivariate regression on targets with \|constant offset\|≥2 and mixed oscill… |
| luna_main:K0020 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On fixed-architecture bigram language models under T≤~512, Adam/AdamW around lr=… |
| luna_main:K0089 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under moderate step budgets (T ~ 2000 steps) on non-linear classification tasks … |
| luna_main:K0155 | live（14 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On classification tasks with unnormalized input features spanning wide spatial r… |
| luna_main:K0298 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For small-batch continuous MSE under T≤~512, sign-normalized Adam/AdamW updates … |
| luna_main:K0311 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | On unnormalized two-spiral classification with radii up to ~20 and T≈256, high-r… |
| luna_main:K0314 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For smooth, bounded 1-D MSE regression on [0,1] with width≈256 MLPs, batch≈32, a… |
| luna_main:K0316 | retired / dropped：The claim ha… | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | For normalized or similarly conditioned MLPs on moderate-radius 1–3-turn spirals… |
| qwen_6ep_40eval:K0031 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under short budgets (T<=256) on zero/near-zero-linear-signal classification, opt… |
| qwen_6ep_40eval:K0044 | retired / dropped：merged into … | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Zero-linear-signal classification (XOR, multi-turn spiral): test CE sits at chan… |
| qwen_6ep_40eval:K0307 | retired / dropped：Superseded b… | K（S 候选） | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Under-training gate: a run is displacement-limited iff cumulative per-parameter … |
| qwen_6ep_40eval:K0309 | live（1 个 KB） | K（S 候选） | Δ 窗口与过冲 | 123 / 18% | 2389 / 25% | 只在点名的少数数据集、题目或实验上观察到 | MSE regression GATE (epochs=T*batch/N_train, rho=params/N_train FIRST). epochs<=… |
| qwen_6ep_40eval:K0313 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Token LMs with small signal headroom S=ln V-H(Y\|X) (simulate the stated table: S… |
| qwen_6ep_40eval:K0314 | live（1 个 KB） | K（S 候选） | Δ 窗口与过冲 | — | — | 只在点名的少数数据集、题目或实验上观察到 | MSE regression at epochs>=12 with rho=params/N>=15: plain SGD lr 1e-3-3e-3 can B… |
| qwen_6ep_40eval:K0315 | live（1 个 KB） | K（S 候选） | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Zero-linear-signal CE (spiral/XOR) obeys the same interpolation gate as MSE: com… |
| seed20_sol_opus:K1018 | retired / dropped：rolled back | K | Δ 窗口与过冲 | — | — | 只在点名的少数数据集、题目或实验上观察到 | XOR overconfidence regime (xor rule with 8-16 Gaussian inputs, only 2 active, 10… |
| smoke_qwen:K0295 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | CE STABILITY CEILING for token LMs. On softmax-CE token tasks (bigram LM, V~20-5… |
| smoke_qwen:K0297 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | PROGRESS = min(D,D_stable)*S when ARCHITECTURE varies too (T<=1024, loss still n… |
| soa_async_v1:K1063 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Univariate MSE,RMSprop rate-only controls: lower lr is NOT a universal short-bud… |
| soa_async_v1:K1082 | retired / dropped：merged into … | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | On sym_0c2620, at 2048 steps/batch16, fixed width128 depth2 nonresidual LeakyReL… |
| soa_async_v1:K1083 | retired / dropped：merged into … | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | On smooth_additive stabcls_021b60 (8 features, 1024 train rows), fixed width256 … |
| soa_async_v1:K1086 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | On 1.5-turn spiralcls_0516f4, fixed width256 depth2 residual SiLU, two LN, Adam(… |
| soa_async_v1:K1104 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Same default w256/d5 residual LeakyReLU(.01), LN[T,T,F,T,F], MSE, bs32, lr.001/w… |
| soa_async_v1:K1114 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Two-input polynomial MSE,default plain d3w256 Leaky(.01),LN101,AdamW(.9,.999),wd… |
| soa_async_v1:K1117 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | 3-input oscillatory MSE,mvar_000e80,default residual d5/w32 SiLU,LN[F,F,T,F,T],A… |
| soa_async_v1:K1118 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 只在点名的少数数据集、题目或实验上观察到 | Bigram CE+L1(.01),V32/L16/alpha1.4,bg_013a4e,default post-LN TF d32/layers4/head… |
| soa_async_v1:K1174 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Nested-sine MSE sin(2*pi*sin(2*pi*x)),default plain d2w128 GELU/LN01,T512/b32: p… |
| soa_async_v1:K1178 | live（1 个 KB） | K | Δ 窗口与过冲 | — | — | 一个点名配方（固定架构与超参）上的受控对比 | Univariate sine+linear MSE,default residual d4w192 GELU/LN0101,wd0,b32,Adam(.9,.… |
| soa_v1:K1013 | live（9 个 KB） | K | Δ 窗口与过冲 | 110 / 10% | 2270 / 16% | 求解器从解题经验归纳的一般陈述，附机制叙述，没有记录分轴的支持与失败 | Bigram LM optimizer sweet spot: labels are sampled from P(y\|x), so gradients car… |

被后继合并、不单列的主张：luna_main:K0300 → luna_main:K0301；qwen_6ep_40eval:K0009 → qwen_6ep_40eval:K0309；qwen_6ep_40eval:K0069 → qwen_6ep_40eval:K0310；qwen_6ep_40eval:K0297 → qwen_6ep_40eval:K0310；qwen_6ep_40eval:K0302 → qwen_6ep_40eval:K0309；soa_async_v1:K1081 → soa_async_v1:K1114；soa_async_v1:K1084 → soa_async_v1:K1117；soa_async_v1:K1085 → soa_async_v1:K1118；soa_async_v1:K1087 → soa_async_v1:K1119；soa_async_v1:K1091 → soa_async_v1:K1118；soa_async_v1:K1092 → soa_async_v1:K1119；soa_async_v1:K1190 → soa_async_v1:K1197；soa_async_v1:K1191 → soa_async_v1:K1198；soa_async_v1:K1192 → soa_async_v1:K1197；soa_async_v1:K1193 → soa_async_v1:K1198；soa_v4:K1062 → soa_v4:K1071；soa_v4:K1064 → soa_v3:K1055；soa_v4:K1067 → soa_v4:K1071
