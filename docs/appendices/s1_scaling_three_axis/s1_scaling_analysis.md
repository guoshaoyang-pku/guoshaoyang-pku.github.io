# V5 scaling statistics

- Source revision for the training batch: `7583ae3222ffb4bbfb13262295a6a828e1f08d3f`.
- Every row below is a single seed-42 run; these are descriptive fits, not uncertainty intervals.
- Gap is fixed validation loss minus the same-step current-batch online train loss.

## Table-size axis
- bigram: after subtracting the no-gram floor `0.02`, clean-window log-log slope `0.5761`, R² `0.9969`, fit n=`12` over `R=2e3–2e5`; raw-gap sensitivity slope `0.5010`. The 31 raw points retain the small-R collapse regime and the large-R saturation regime, which are excluded from this local fit. K=`3541098`; K/R is a load ratio only. Collision rate was not measured.
- trigram: after subtracting the no-gram floor `0.02`, clean-window log-log slope `0.6648`, R² `0.9997`, fit n=`8` over `R=1e5–9.3e5`; raw-gap sensitivity slope `0.6526`. The 31 raw points retain the small-R collapse regime and the large-R saturation regime, which are excluded from this local fit. K=`19027841`; K/R is a load ratio only. Collision rate was not measured.

## Epoch-length axis
- **Erratum 2026-08-30**: points above 1×L4 share the shard-1 frequency index and are wrap-around replays of shard 1, i.e. pass-count points (2×L4 3-epoch endpoint `5.582` ≈ long-replay 6-pass `5.609`). The earlier quadratic fit in ln(L4 multiplier) (vertex `0.411502×L4`, R² `0.774155`) mixed two variables on one axis and is retired. The true epoch-length segment (≤1×L4, nested prefixes, 3 passes) declines gently from `3.552` at 0.125× to `2.469` at 1×L4.
- Long replay: `s1v5_128_ep_tri_1xL4_10ep` reaches gap `8.6752` at epoch 10 / step 3370; the epoch-boundary increments peak early and decline toward about `0.7–0.8` gap per epoch. This supports gradual concavity / slowing, not a demonstrated plateau. The matched no-gram control reaches `0.4802`.

## Dose axis
- **Erratum 2026-08-30**: the batch behind the numbers below (`nglab*_input_v5_freq10`) has `table_lr_scale=2.0` in its configs (2x-era). The 128x authoritative batch is `nglab*_input_v5_128x_freq10`: gap `10.895` at 0.25× down to `0.355` at 5×, `-0.087`/`-0.055` at 6×/8×; positive-gap slope (≤5×, n=10) `-1.176`, R² `0.899`; recorded in `s1_dose_points_128x.csv`.
- 2x batch, positive-gap points through 5× only: log-log slope `-1.726983`, R² `0.887225`, rank correlation `-1.000000`; the sign change is bracketed between 5× and 6×, so no global power law is reported for either batch. On a completed-passes axis the dose points collapse onto the long-replay curve (main report fig 7c).

## Exact-frequency axis
- bigram: token-mass-weighted geometric-bin log-log slope `-0.252746`, R² `0.997165`, rank correlation `-1.000000`, n=`7`; positive-gap bins only, with exact-f points retained in the CSV.
- trigram: token-mass-weighted geometric-bin log-log slope `-0.318121`, R² `0.995548`, rank correlation `-1.000000`, n=`7`; positive-gap bins only, with exact-f points retained in the CSV.

## Files
- `s1_table_size_points.csv`: one row per formal bigram-R/trigram-R run.
- `s1_epoch_length_points.csv`: one row per formal 3-epoch point.
- `s1_epoch_long_replay_points.csv`: epoch-boundary values for both/no-gram 10-epoch replay.
- `s1_dose_points.csv`: final dose endpoints (2x batch, see erratum).
- `s1_dose_points_128x.csv`: final dose endpoints for the 128x authoritative batch.
- `s1_dose_frequency_gap.csv`: raw final frequency-bin rows, including token fractions.
- `s1_frequency_exact_points.csv`: shared-context exact-f rows from the formal frequency run.
- `s1_scaling_fits.csv`: descriptive fit coefficients and rank correlations.
- `v5_optimizer_points.csv`: final metrics for the clean 11-arm optimizer refresh.
