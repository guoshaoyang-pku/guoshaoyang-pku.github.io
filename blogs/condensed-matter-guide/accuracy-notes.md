# AB 双层石墨烯：第一性原理精度与算力的文献证据

核查日期：2026-10-10。对象为普通 AB/Bernal BLG，优先考虑高质量、栅控、hBN 封装器件。扭转 BLG 和 AA 堆垛的结果仅作为方法参照。以下是文献核查，未启动新计算。

## 结论

目前核查到的文献没有证明：现实高质量 AB 器件的完整第一性原理计算，已经在总误差上超越优秀实验。这个结论不等于原则上不可能。对能带、层外屏蔽、激子和部分磁响应，已有定量成功；对关联相界、超导和器件输运，没有普适的精度保证。

三类误差没有固定排序。扩大同一 DFT/HF 计算只能降低数值误差；改变电子相关近似、扩大降阶空间、匹配响应算符和升级多体求解器，才能减少对应的物理近似误差。真实器件的结构和散射输入另需确定。

## 1. 微观电子结构：DFT 之后加什么、改变多少

| 证据 | 增加的物理 | 量化结果 | 精度边界 |
|---|---|---|---|
| Liang & Yang 2012 [1]，加偏置 AB BLG | LDA → G0W0 自能 → BSE 电子空穴相互作用 | 采样场强下，GW 准粒子 gap 比 DFT 增大 56%、67%、78%、81%；激子结合能 35、54、76、80 meV | 这些是方法间修正，不是已知总误差；当时不能分辨 <10 meV 的能量差，且实际介电环境与孤立模型不同 |
| Slizovskiy 等 2021 [2] | DFT 层外极化率与自洽静电 | PBE 11.0 Å³、LDA 10.8 Å³；得到 BLG εz≈2.6 | Bernal 光学 gap 的更好匹配需要 εz≈2.35、γ1≈0.35 eV，或处理未完全抵消的自能/激子作用。泛函差约 2% 不能当作总误差 2% |
| Konschuh 等 2012 [3] | 增加名义未占据的碳 d 及更高轨道 | 本征 SOC 劈裂约 24 μeV；删去这些轨道后只剩 <1 μeV | 小绝对能标可以计算；这是劈裂和基组完整性的证据，不是现实器件的 μeV 总误差认证 |

网格、空带和基组可以用算力检验收敛。固定交换相关泛函的偏差不会随网格消失；固定 G0W0/BSE 仍有自能与相互作用顶角近似。增加自洽程度也不是保证单调接近精确解的误差控制。

## 2. 降阶：能转，但不能只转能带

Gava 等 2009 [4] 对比 DFT 与当时简单 tight binding：低掺杂时 DFT 的 K 点 gap 约为简单模型的一半。差异来自模型只保留层间电荷转移、遗漏层内极化。论文给出了修正描述。这是可修复的模型误差，不能说现代模型必然错一倍。

对磁、电和光响应，低能 Hamiltonian 与响应算符必须一起降阶。高能带虽然不占据，仍通过虚跃迁贡献磁矩、屏蔽和顶角；拟合能带好，不保证响应好。Sethi 等的补充 Fig. S4 [5] 中，低能态几乎完全来自四个 C p_z 轨道，但只用四带求磁矩仍显著低估完整远带求和。

精确消去高能自由度的表达式可以写出，例如 Feshbach 投影 H_eff(E)=PHP+PHQ(E−QHQ)⁻¹QHP；也可用路径积分定义低能有效作用量。困难是计算并保留能量/频率依赖、非局域作用、多体项与响应算符。静态少带模型只是进一步截断。扩大活性空间和保留这些项有实际路线，但没有适用于所有 BLG 响应的统一总误差界。

### 最直接的响应改进：2026 年激子磁矩预印本

Sethi 等 [5] 用第一性原理 LDA/GW/BSE 描述 hBN 封装、偏置 AB BLG，并补齐 Berry 相位、电子空穴相对运动和质心量子几何项。主文 Table 1，PDF 物理第 12 页：

| 处理 | s 激子谷 g 因子 | p 激子谷 g 因子 |
|---|---:|---:|
| 仅带边电子/空穴磁矩 | 15.06 | 15.06 |
| 由激子包络波函数加权 | 18.45 | 13.03 |
| 完整响应表达式 + GW/BSE | 20.32 | 1.79 |
| 实验 Ju 等 2017 [6] | 19.8 ± 0.1 | 1.4 ± 0.8 |

p 响应从约一个数量级的偏差变为进入实验误差范围；s 的残差 0.52，即约 2.6%，大于实验报告的拟合误差 ±0.1，即约 0.5%。实验拟合误差不是全部系统误差，不能把它直接解释为严格的 5.2σ 物理矛盾。理论也没有完整总误差条，因而不能认证比实验更准。

这项巨大改进属于补齐响应公式，不能全归因于 DFT，也不能严格全叫降阶误差。它说明正确的响应算符与相关电子空穴态不可缺少。

补充材料 PDF 物理第 15–16 页：BSE 激子能量数值收敛至 5 meV，未给出 g 因子同等完整的收敛误差；采用共格 hBN 单层，忽略 graphene/hBN 不共格结构；hBN 封装时的介电矩阵近似沿用 pristine BLG。两种 hBN 堆垛给 g_s=13.48 与 20.32，后者更接近实验，不能当作已知真实器件结构下的盲预测。主表 g_p=1.79；补充比较段写 1.76，存在小的稿件内差异，此处采用主表。

## 3. 多体求解：数值可以很精，模型仍可能有偏差

Assouline 等 2024 [7] 在高质量 AB BLG 分数量子 Hall 器件中，用四带波函数、栅极屏蔽、高 Landau 能级 RPA 屏蔽与 DMRG 计算激活隙：12 T 下理论 5.6 K，实验 5.1 ± 0.2 K，差约 10%。这是有效模型的成功，没有构成端到端第一性原理精度证明。

补充材料检查圆柱周长 12–24 ℓB，χ≥1600 时对 bond dimension 不敏感；半填充仍需周长与缺陷区域外推。遗漏可能的自旋纹理、高能级屏蔽的 RPA 近似，以及器件无序，不会因增加 DMRG 迭代自动消失。热力学 gap 与输运激活 gap 是不同量，不能混作同一精度比较。

Leaw 等 2019 [8] 的 Bernal 晶格 QMC 达到 900 格点，但真实 t⊥/t≈0.10–0.15 的低能动量尺度仍超出可达尺寸；为揭示机制使用 t⊥/t=1–10。这是有限尺度瓶颈的直接例子，不是现实器件的精密模拟。

Sun 等 2014 [9] 的偏压 Bernal Hubbard DQMC 遇到 sign problem，并用平均场补充相界。示例参数下平均符号约 0.1，但该参数并非真实器件，不能据此报我们所需算力。固定统计精度的采样代价含 1/〈s〉²：若符号从 0.1 降到 0.001，需要再增加约 10⁴ 倍工作；若符号为 10⁻¹⁰，则相对无符号问题的惩罚约 10²⁰。一般符号问题的复杂性结果 [10] 不证明特定 AB BLG 目标一定指数困难。

## 4. 真正超越 DFT 的波函数计算

Mostaani 等 2015 [11] 用 diffusion Monte Carlo 算 AB 结合能 17.7±0.9 meV/atom；呼吸模约 83±7 cm⁻¹，对照实验 80±2 cm⁻¹。论文明确保留未知 fixed-node 偏差、赝势与有限尺寸误差。Monte Carlo 统计误差不等于总误差。

Li、Qian、Chen 2024 [12] 的 DeepSolid 多体神经波函数直接计算极化，不依赖近似 DFT 交换相关泛函：双层极化率 11.6(1) Bohr³，统计约 0.9%。但算例是 AA 堆垛，2×2 超胞、16 个 C 和 64 价电子；不是 AB 器件的精度认证，也未把变分波函数和热力学极限误差全部认证到 0.9%。

Saritas 等 2026 预印本 [13] 的 DMC 已做到最多 256 原子、1024 电子；AB 结合能 15.4(4) meV/atom，呼吸模 73(1) cm⁻¹，对照实验 80(2) cm⁻¹。它展示可扩展性，也展示统计误差变小仍不能认证准确度；fixed-node 等偏差仍在。不同年份结果不是受控的同一方法/条件算力曲线。

## 5. 算力投资能兑换什么

| 增加的工作 | 可减少的误差 | 尚不能据此保证的事 |
|---|---|---|
| 更密网格、更多空带/轨道、有限尺寸外推 | 积分、基组、截断、尺寸误差 | 所选电子相关近似的总准确度 |
| 更多独立 MC 样本 | 固定条件下统计误差约随样本数平方根下降；缩小 10 倍通常需约 100 倍样本 | fixed-node、波函数/模型和器件误差 |
| 更大活性空间，动态相互作用，匹配响应算符 | 降阶遗漏与部分屏蔽误差 | 没有验证就给全流程误差界 |
| QMC/DMRG/变分波函数与 HF 交叉求解 | 可识别和降低部分求解器近似 | 所有二维低温、掺杂问题都有可负担的收敛路径 |
| 真实结构、应变、hBN/栅极与散射建模 | 理想模型到特定器件的偏差 | 未知输入能靠数值收敛自动辨识 |

因此不能把“充足算力”换成一个通用倍数。可 scale 的数值部分、需升级方法的近似部分、可能指数困难的多体部分同时存在；表达式本身可以写出。

## 6. 对当前 blg-autolab 的含义

当前 README 记录的是四带 Hartree/部分低能 Fock、单粒子/Berry/interband 响应与常数 τ Boltzmann 输运。未建立完整的第一性原理参数—相互作用—响应算符链条。扩大当前扫描能改进有限网格结果，不能自动去掉动态关联、高能与谷间交换遗漏，也不能由常数 τ 得到特定真实器件的散射与 Rxx。

合理的第一阶段是选定一个已知器件结构下的正常态响应，例如层外极化或磁光激子响应，建立 ab initio → 降阶 Hamiltonian、作用与算符 → 多体响应的链条。分别记录数值误差、方法间变化和结构不确定度，再用未参与校准的实验条件检验。光学和分数量子 Hall 的成功不能迁移成零场器件 Rxx、超导或完整相图的准确度认证。

## 参考文献与核查位置

1. Liang & Yang, “Enhanced Many-Electron Effects on Excited States of Gated Bilayer Graphene,” PRB 86, 205423 (2012). [DOI](https://doi.org/10.1103/PhysRevB.86.205423)；[arXiv v1 全文](https://arxiv.org/html/1210.6018v1)，GW gap、Fig. 3–4、能量分辨率讨论。
2. Slizovskiy 等, “Out-of-plane dielectric susceptibility of graphene in twistronic and Bernal bilayers,” Nano Lett. 21, 6678 (2021). [DOI](https://doi.org/10.1021/acs.nanolett.1c02211)；[arXiv v3 全文](https://arxiv.org/html/1912.10067v3)，单层极化率及 Bernal 光学 gap 比较。
3. Konschuh 等, PRB 85, 115423 (2012). [DOI](https://doi.org/10.1103/PhysRevB.85.115423)；[arXiv v2 全文](https://arxiv.org/html/1111.7223v2)，SOC 与高轨道讨论。
4. Gava 等, “Ab initio study of gap opening and screening effects in gated bilayer graphene,” PRB 79, 165431 (2009). [DOI](https://doi.org/10.1103/PhysRevB.79.165431)；[arXiv v2 全文](https://arxiv.org/html/0902.4615v2)，DFT/TB 比较与层内屏蔽。
5. Sethi 等, “Quantum Theory of Exciton Magnetic Moment: Interaction and Topological Effects,” [arXiv:2509.07284v3](https://arxiv.org/abs/2509.07284v3)，2026-06-22 修订，预印本。主表 PDF 物理第 12 页；方法第 15 页；堆垛第 16 页；远带效应 Fig. S4 第 27 页。
6. Ju 等, “Tunable excitons in bilayer graphene,” Science 358, 907–910 (2017). [DOI](https://doi.org/10.1126/science.aam9175)。磁响应实验数值及拟合误差亦由 [5] 主表和正文复核。
7. Assouline 等, “Energy gap of the even-denominator fractional quantum Hall state in bilayer graphene,” PRL 132, 046603 (2024). [DOI](https://doi.org/10.1103/PhysRevLett.132.046603)；[arXiv 全文及补充](https://arxiv.org/html/2308.05729)，Fig. 2c 及 DMRG 方法。
8. Leaw 等, PRB 100, 125116 (2019). [全文](https://arxiv.org/html/1903.06177)，有限格点和层间耦合范围。
9. Sun 等, “Electrically controllable magnetic order in the bilayer Hubbard model on honeycomb lattice — a determinant quantum Monte Carlo study,” PRB 90, 125429 (2014). [全文](https://arxiv.org/html/1405.1186)，平均符号与平均场补充。
10. Troyer & Wiese, “Computational Complexity and Fundamental Limitations to Fermionic Quantum Monte Carlo Simulations,” PRL 94, 170201 (2005). [arXiv](https://arxiv.org/abs/cond-mat/0408370)。一般复杂性结果，非 BLG 具体下界。
11. Mostaani、Drummond、Fal’ko, “Quantum Monte Carlo Calculation of the Binding Energy of Bilayer Graphene,” PRL 115, 115501 (2015). [DOI](https://doi.org/10.1103/PhysRevLett.115.115501)；[arXiv v1](https://arxiv.org/abs/1506.08920v1)，主文结合能、呼吸模与误差讨论。
12. Li、Qian、Chen, “Electric Polarization from Many-Body Neural Network Ansatz,” PRL 132, 176401 (2024). [DOI](https://doi.org/10.1103/PhysRevLett.132.176401)；[arXiv v2](https://arxiv.org/abs/2307.02212v2)，AA 超胞、极化率与方法。
13. Saritas 等, [arXiv:2609.05845v1](https://arxiv.org/abs/2609.05845v1)，2026-09 预印本；[全文](https://arxiv.org/html/2609.05845v1)，BLG 结构/呼吸模表与 fixed-node 讨论。

## 7. 低密度与小能标的具体含义

“低密度”没有对所有位移场都适用的固定阈值。为说明本次讨论：在理想、无隙、四重自旋/谷简并的抛物线 AB BLG 中，γ1=0.39 eV、v=10⁶ m/s 给出 m*=0.0343 me，EF=πℏ²|n|/(2m*)。|n|=10¹⁰、10¹¹ cm⁻² 对应约 0.35、3.5 meV。这里的 EF 只是参考尺度；有位移场、三角翘曲或关联破缺时需重新计算。

McCann & Koshino 综述 [14] Eq. 95 给三角翘曲的 Lifshitz 能量 εL=γ1(v3/v)²/4≈1 meV；用上述抛物线关系映射是约 3×10¹⁰ cm⁻² 的密度数量级，不能当成精确翘曲带的转变密度。此处“细小”指相对于 eV 级微观 hopping 的绝对能量小，却可能与 EF、kBT、序参量 gap 或两候选相的自由能差相当。

例如 1 K 的 kBT=0.086 meV；0.1 meV 的误差在 eV 尺度上很小，但相对于 0.35 meV 的参考 EF 已约 29%，也超过 1 K 热能。它可能改变费米面口袋、gap 的符号、相排序与激活输运。究竟会不会翻转相排序，需同候选相自由能差及其误差比较，不能从这个参考 EF 单独推断。小绝对能标本身不是不可算的证明，[3] 的 24 μeV SOC 劈裂就是反例。

14. McCann & Koshino, “The electronic properties of bilayer graphene,” Rep. Prog. Phys. 76, 056503 (2013). [DOI](https://doi.org/10.1088/0034-4885/76/5/056503)；[arXiv v2 全文](https://arxiv.org/html/1205.6953v2)，§II.3–II.5、Eq. 95。上述 EF 与参考密度由所列公式和参数计算，未作为实验值引用。
