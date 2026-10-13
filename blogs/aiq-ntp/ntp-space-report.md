# 少采teacher和数据，优先扩大模型设计空间

下一轮保留teacher与数据的全部自由度，默认只采12个代表条件，固定V64、前缀32、续写32、4096条和AdamW700更新。模型搜索扩到8类序列算子、7个明确混合宏块和5种外层连接。有限文法有1,970种架构、2,006个含初始化的模型setting，其中现有代码可生成的下一步子集为179架构／199模型。新Mamba、KDA、连接和混合路线尚未实现，文法数量不代表可训练或稳定。当前默认仍为20个Transformer；最新实测1,200次训练确认26题，累计70题待人工review，窗口匹配捷径仍在。

## Teacher和数据先取12组，完整自由度继续保留

默认3种初始化（random、距离priorα1、α2）×4个代表机制，固定一层d32、4heads、FF比例2、GELU。四机制为全历史、W4、W16（均无位置、恒等路径0），以及全历史＋sinusoidal＋恒等路径.25。V64、前缀32／续写32、train4096、评测同长。AdamW3e-4／wd.01／700steps是唯一默认训练计划，其他长度、容量、词表、优化器和预算轴保留为后续诊断。

机制配对必须共享确切teacher权重和输出校准。初始化对照须共享底层权重抽样，并只改声明的prior。Canonical全局index本身不保证这一干预关系，pilot materialization前需冻结实际谱系。seed重复另计。默认12组是小pilot建议，尚未派发。

![当前空间](ntp-current-space.svg)

![扩大模型空间，保留teacher数据轴](ntp-next-space.svg)

## 模型以8类算子、5种连接和7个混合宏块组合

序列算子（mixer）为Transformer attention、GRU、RNN、对角SSM、Mamba-1、Mamba-2、Gated DeltaNet、Kimi Delta Attention（KDA）。对角SSM是固定线性递推，Mamba有输入相关选择机制。KDA的遗忘gate细化到每个key维，Gated DeltaNet是每head标量gate。

Block可选Mixer→FF串联或Mixer∥FF并联，Dense或门控FF，LayerNorm或RMSNorm。Transformer并联仍区分pre／post-norm，避免保留无效轴。每个模型只选一种可配置activation；门控FF也使用该函数。GRU原生门、Mamba核心SiLU、GDN／KDA短卷积SiLU保持原定义。纯Mamba使用原生block，不追加FF。

| 外层连接 | 定义 |
|---|---|
| 相加 | x＋F(x) |
| 缩放分支 | x＋0.5F(x)，identity系数固定1 |
| 可学习gate | x＋sigmoid(g)F(x)，每block一个标量g，初始0 |
| 长skip | 保留局部残差，额外跨2个block加0.5倍旧隐藏状态；单block排除 |
| 无残差 | F(x)，仅作为正常配置的受控消融；不关闭算子固有循环门和内部skip |

混合宏块固定为Attention→Mamba1、Attention→Mamba2、Attention→GDN、Attention→KDA、Attention∥Mamba2、Attention∥KDA、KDA→KDA→KDA→Attention。宏块重复1／2／3次，内部算子默认参数冻结，不能用逐层自由选择绕过两轴限制。3:1宏块是benchmark设计，未接入完整Kimi Linear的MLA。

## 文法约有1,970种架构，只有子集已有代码支持

| 家族／固定宏块 | 架构模板 | 含初始化的模型setting |
|---|---:|---:|
|Transformer|238|238|
|GRU|55|55|
|普通RNN|67|67|
|对角SSM|141|177|
|Mamba-1|108|108|
|Mamba-2|158|158|
|Gated DeltaNet|214|214|
|KDA|214|214|
|attention_mamba1|111|111|
|attention_mamba2|111|111|
|attention_gated_deltanet|111|111|
|attention_kda|111|111|
|parallel_attention_mamba2|110|110|
|parallel_attention_kda|110|110|
|kda3_attention1|111|111|

每个prototype以0／1／2轴邻域计数：1＋Σaᵢ＋Σᵢ＜ⱼaᵢaⱼ，再去除明确几何冲突和有效spec重合。aᵢ为该轴非起点值数。SSM初始timescale从架构数移除，全局activation、FF比例和外层连接各算一轴，FF宽度随width×ratio派生。prototype不是已晋升anchor，新算子先校验和pilot。

当前代码可生成的下一步子集为TF114、GRU9、RNN14、SSM42，共179架构；含20个SSM初始化对照为199模型。TF新增15个有来源的无残差单轴／双轴消融。完整文法包含这些基线，不能与179／199相加。

| 口径 | 当前默认 | 下一步建议 |
|---|---:|---:|
| 默认teacher／数据条件 |640个索引条件|12个代表条件|
| 默认训练计划／条件 |6种可选，700步基线|1种，AdamW3e-4／700步|
| 已支持候选子集 |20架构／20模型|179架构／199模型|
| 完整候选文法 |—|1,970架构／2,006模型|
| 每固定数据的完整候选计划容量 |120|子集4776；大文法48,144（未实现）|

保留的teacher／数据全轴目录，在199模型的可生成子集参数门槛下仍为12,096个teacher、979,776个数据条件、23,514,624个题目setting。这是备用容量，不作为默认采样目标。扩展文法尚未通过参数cap、teacher小于所有candidate、kernel、因果、梯度和学习检查，因此不能将大文法乘到该目录上认定合法总量。

## 先验证算子和连接，再在少量固定数据上取局部对照

先接入标准算子，核验生成代码、原生数值对齐、因果与有限梯度。再增加外层连接与明确的串并联宏块。每个局部组保留起点、单轴和组合cell，记录实际初始化谱系、batch流和预算，正常改动变弱仍保留。跨算子比较披露参数与计算量差异。

默认700steps和约10秒开发校准继续保留。时间／FLOPs题需各选项预算对齐，不在空间计数中混乘。新隐藏测试只确认或拒绝，不换成员／答案。失败与未解决组保留在分母中。

来源：[Mamba官方实现](https://github.com/state-spaces/mamba)、[Kimi Linear官方说明](https://github.com/MoonshotAI/Kimi-Linear)、[FLA KDA实现](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py)、[Gated DeltaNet论文](https://arxiv.org/abs/2412.06464)。本地：[计数](ntp-space-counts.json)、[spec](ntp-next-space-spec.json)、[proposal](ntp-proposal.html)。
