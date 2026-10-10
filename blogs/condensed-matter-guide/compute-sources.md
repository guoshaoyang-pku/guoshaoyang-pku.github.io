# 凝聚态计算的算力：可核查的实例与展示边界

这份笔记为科普子站的算力示意图提供证据。核查日期 2026-10-10。GPU 时 = GPU 数量 × 墙钟小时，CPU 核时 = CPU 核数 × 墙钟小时。不同代 GPU、CPU 核时不能直接折算。下表是具体算例，不是全领域平均值，也不是 BLG 精度报价。

## 论文实测点

| 任务 | 计算对象 | 论文报告的硬件与时间 | 换算后的总量 | 核查位置与限定 |
|---|---|---|---|---|
| 常规电子结构与声子响应 | bcc W 原胞，45³ k 网格，单个 q 点声子及前置电子态计算 | 32 CPU 核，8040 s；优化后 8 张 Ampere GPU，2153 s | 71.5 CPU 核时；4.8 GPU 时 | Gong & Dal Corso，CPC 308, 109439 (2025)，§6.1、Tables 4/6。这里不只是一次基态自洽计算。标准 GPU 路线反而更慢，算法与工作负载影响很大。[1] |
| 数千电子 DFT 基态 | 含单空位的 Mo 超胞，431 原子、6034 电子，Γ 点 | 24 V100 GPU，0.372 Summit 节点时、4 节点 | 335 s，2.23 GPU 时 | Das 等 DFT-FE 1.0，CPC 280, 108473 (2022)，Table 4。包含初始化、自洽和离子力。有限元、赝势特定基准。[2] |
| 大规模 DFT 基态 | 含单空位的 Mo 超胞，8191 原子、114674 电子，Γ 点 | 3600 V100 GPU，3250.1 s | 约 3250 GPU 时 | 同上 Table 7。约 54 分钟墙钟不等于工作站 54 分钟。它展示大规模数值求解，不证明交换相关近似准确到所需物理尺度。[2] |
| 直接多电子神经波函数 | 石墨烯 2×2 超胞，8 C、48 全电子 | 32 A100 GPU，3 天训练 | 约 2304 GPU 时 | Li、Li、Chen，Nat. Commun. 13, 7895 (2022)，补充 Table 2、Note 3。表给单次训练配置，多个 twist、尺寸和结构仍要额外计算。[3] |
| 更大直接多电子神经波函数 | LiH 3×3×3 超胞，108 全电子 | 128 A100 GPU，4 周训练 | 约 86016 GPU 时 | 同上补充 Table 2、Note 4.5。单次训练并不等于热力学极限和全部误差已经验证。[3] |
| 超算级 GW 自能 | 998 原子 Si 基准、特定非对角自能任务 | 9408 Frontier 节点，604.96 s，含 I/O | 约 1581 节点时 | Zhang 等 SC 2025，Table 5。仅此自能任务，不能当作完整 DFT→介电矩阵→GW→BSE 的总成本。峰值 1.069 EFLOP/s 是内核性能。[4] |

Frontier 每节点有 4 块 MI250X 物理加速板，每板含 2 个计算芯粒。论文把 9408 节点写成 75264 GPU devices。网页建议保留节点计数，避免与 A100 的物理板计数混淆。[4]

## 算力图可以怎么写

用“工作站 → 小集群 → 大型超算”表示资源跨度，用上表实例锚定实际数量。把“单个参数点”和“扫描很多材料、温度、密度、几何”区分开。一个小矩阵参数点可以很便宜，大扫描仍会消耗大量总算力。不要把不同任务连成一条“算力越多、精度越高”的普适曲线。

以下属于规划经验，不是统计意义的行业平均：小型 tight binding／平均场常在工作站上完成；常规单位胞 DFT 常用一个或数个 CPU 节点／GPU；复杂 GW/BSE、二维强关联和系统收敛研究可以从小集群扩展到大型超算。具体硬件和时间要由活性空间、k 网格、温度、对称性、目标误差和软件实测决定。

## 共同对象与研究方向

| 研究问题 | 共同计算对象 | 常见输出 |
|---|---|---|
| 哪个结构、成分更稳定 | 总能量 E、自由能 F、原子受力 | 晶体结构、缺陷能、相稳定性 |
| 电子如何运动 | 能带、单电子 Green 函数（加入或移去电子的传播规律） | 光电子谱、有效质量、准粒子寿命 |
| 相互作用让电子形成什么状态 | 多电子波函数／密度矩阵、关联函数（两处物理量怎样一起变化） | 磁序、关联绝缘体、超导、拓扑相 |
| 外场如何产生信号 | 响应函数（输入外场到输出信号的对应关系） | 光学吸收、电导、磁化率、介电常数 |
| 原子如何振动与传热 | 声子、原子轨迹、散射率 | 振动谱、热导、结构相变 |
| 如何找到新材料与快算法 | 上述量的数据库、近似模型、代理模型 | 材料筛选、机器学习势、数值方法 |

这些方向互相重叠。DFT 是一类计算框架，GW/BSE 与 QMC/DMRG 是不同用途的方法，材料筛选是工作方式，不能作为同一层的互斥“学科分类”。

## 三种不同的 scale

**固定近似下的数值收敛。** 更密网格、更多空带、更大基组可以消除对应数值误差，但不会自动修复所选交换相关泛函和响应公式的偏差。Bosoni 等的 960 种晶体验证项目明确把代码间 numerical precision 与对实验的 accuracy 分开。[5]

**算法升级。** Gerard 等 2024 的可迁移神经波函数在 LiH 的 32→108 电子算例上，用约原 DeepSolid 1/50 的成本得到更低变分能。它说明算法收益可能很大，不构成所有材料/响应都有 50 倍加速的结论。[6]

**多体空间增长。** 半填充、总自旋 z 分量为零的 L 格点 spinful Hubbard 模型，在不进一步使用空间对称性前，基底维数为 C(L,L/2)²。L=16、20、24 时约为 1.66×10⁸、3.41×10¹⁰、7.31×10¹²。仅保存一个 complex128 向量就约需 2.65 GB、546 GB、117 TB。这是说明精确对角化如何迅速变难的算术实例，实际 Krylov 求解需更多向量，可进一步利用对称性。不是这些格点数的统一算力报价。

QMC 的统计误差一般随有效独立样本平方根下降。存在 sign problem 时，采样代价还包含约 1/〈sign〉² 的惩罚。张量网络在固定纠缠复杂度时可以很好地 scale，二维宽度增加时所需纠缠容量可以迅速增长。这两类求解器不能用“多加 1000 倍算力就精确”概括。

## 来源

1. Xuejun Gong & Andrea Dal Corso, *An alternative GPU acceleration for a pseudopotential plane-waves density functional theory code with applications to metallic systems*, Computer Physics Communications 308, 109439 (2025). [DOI](https://doi.org/10.1016/j.cpc.2024.109439)，[全文 v2](https://arxiv.org/html/2412.01695v2)。
2. Sambit Das 等, *DFT-FE 1.0: A massively parallel hybrid CPU-GPU density functional theory code using finite-element discretization*, Computer Physics Communications 280, 108473 (2022). [DOI](https://doi.org/10.1016/j.cpc.2022.108473)，[全文 v2](https://arxiv.org/html/2203.07820v2)。
3. Xiang Li、Zhe Li、Ji Chen, *Ab initio calculation of real solids via neural network ansatz*, Nature Communications 13, 7895 (2022). [DOI](https://doi.org/10.1038/s41467-022-35627-1)，[全文 v2](https://arxiv.org/html/2203.15472v2)，[期刊补充 PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-022-35627-1/MediaObjects/41467_2022_35627_MOESM1_ESM.pdf)。
4. Benran Zhang 等, *Advancing Quantum Many-Body GW Calculations on Exascale Supercomputing Platforms*, SC 2025, pp. 48–59. [DOI](https://doi.org/10.1145/3712285.3772093)，[全文 v1](https://arxiv.org/html/2509.23018v1)。
5. Bosoni 等, *How to verify the precision of density-functional-theory implementations via reproducible and universal workflows*, [全文 v1](https://arxiv.org/html/2305.17274v1)。本笔记只引用其正文明确说明的 verification/validation 区别。
6. Leon Gerard 等, *Transferable Neural Wavefunctions for Solids*, [预印本全文 v1](https://arxiv.org/html/2405.07599v1)。成本比较见摘要、§2.3，A100 timing 设置见 Supplement §S2。
