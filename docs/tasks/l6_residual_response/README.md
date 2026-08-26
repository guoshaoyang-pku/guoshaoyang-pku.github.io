# L6 · 残差—响应精确模型

## 一句话

`−1` 不是 loss gap 的普遍指数，方差也不是普遍的唯一控制量。单数据集的精确对象是
**训练采样残差与模型所学 logits 的内积**；跨训练集取期望后是 covariance，
且只有响应近似线性时，方差才是首项。

## 精确式

令 `δ = p̂ − p`，模型在训练集 `D` 上学到 logits `z_D`。同一 context 的 gap 为

```text
G_D = CE(p, q_D) − CE(p̂, q_D) = δᵀ z_D,
E_D[G_D] = Σ_i Cov_D(p̂_i, z_{D,i}).
```

其中 `E[δ]=0`、`Cov(δ)=[diag(p)−ppᵀ]/f`。写成 `z_D=z*+u_D` 后，单次数据集有
`G_D=δᵀz*+δᵀu_D`。所以把 `∂L/∂p` 当成固定常数时，第一项的 signed 期望为 0；
其 RMS 为 `√(Var_{Y~p}[z*(Y)]/f)`，典型绝对大小是 `f⁻¹ᐟ²`。二项分布有
`E|p̂−p|≈√(2/π)√(p(1−p)/f)`；对应到 loss 还要乘
`|logit(p)|`。因此 `p=1/2` 时虽然比例残差仍有 `f⁻¹ᐟ²` 涨落，loss 的一阶项
却恰好为零。

二元平滑响应 `q_D≈p+ρδ` 给出最小展开
`G_D≈logit(p)δ + ρδ²/[p(1−p)]`：第一项是均值 0、绝对尺度 `f⁻¹ᐟ²`
的单数据集涨落；第二项的期望是 `ρ/f`。因此二者不是互斥理论，而是同一个展开的
一阶随机项与二阶系统性 bias。

若 `z_D = z₀ + Aδ + 1/2 B[δ,δ] + ...`，才有

```text
E[G] = tr(A Cov(δ)) + 1/2 B:E[δ⊗δ⊗δ] + ...
```

因此二阶矩只是局部线性响应下的第一项；非线性响应会选择三阶、四阶或其他矩。
更一般地，若 learned response 满足 `u(λδ)≈λ^a u(δ)`，则
`E[δu(δ)]∝f^{−(a+1)/2}`：饱和/sign 响应给 `−1/2`，线性响应给 `−1`，
三次响应给 `−2`。

还要区分“分布无偏”和“loss 无偏”：train 与独立 val 的经验分布都对 `p` 无偏，
但 `q_D` 使用同一个 train `p̂` 训练，所以 train loss 有 optimistic bias；独立 val loss 对固定 `q_D` 无偏估计
population CE，但相对 `H(p)` 有 estimation excess。resolved count table 中两者分别为
`∓(K−1)/(2f)`（平滑、全支持、大样本渐近）。低维支撑只让有效维数固定为
`K−1`，不会让经验分布本身变成有偏估计。

## 两个实验及数据

| run_id | 具体数据 | 回答什么 |
|---|---|---|
| `l6_counttable_freq_exact_v1` | 二元 `p={.50,.20,.05}`，`f=4..4096`，Jeffreys smoothing `α=.5`，逐个二项计数精确求和 | resolved count table 何时才接近 `f⁻¹`；方差项何时足够 |
| `l6_response_moments_exact_v1` | `δ` 为 f 个 Rademacher 样本均值，精确枚举；响应 `u(δ)=δ, sign(δ), δ³` | 同样的采样残差，响应不同可给 `f⁻¹, f⁻¹ᐟ², f⁻²` |

产物在 `results/<run_id>/{config.json,metrics.csv,summary.json}`；两臂均无随机抽样，
所以 `seed = null`。图由 `docs/plot_scripts/plot_l6_residual_response.py` 生成。

## 边界

这两个 run 是数学模型的精确 positive/counterexample controls，不是 nanoGPT 结果。
自然语料的一次固定 corpus 还混合了 context 分布、长尾、碰撞、跨 context transfer
和优化动力学，不能拿这里的 `−1` 当作它必须通过的验收线。
