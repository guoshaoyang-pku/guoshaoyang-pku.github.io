# V5 scaling statistics

- Source revision for the training batch: `7583ae3222ffb4bbfb13262295a6a828e1f08d3f`.
- Every row below is a single seed-42 run; these are descriptive fits, not uncertainty intervals.
- Gap is fixed validation loss minus the same-step current-batch online train loss.

## Table-size axis
- bigram: after subtracting the no-gram floor `0.02`, clean-window log-log slope `0.5761`, R² `0.9969`, fit n=`12` over `R=2e3–2e5`; raw-gap sensitivity slope `0.5010`. The 31 raw points retain the small-R collapse regime and the large-R saturation regime, which are excluded from this local fit. K=`3541098`; K/R is a load ratio only. Collision rate was not measured.
- trigram: after subtracting the no-gram floor `0.02`, clean-window log-log slope `0.6648`, R² `0.9997`, fit n=`8` over `R=1e5–9.3e5`; raw-gap sensitivity slope `0.6526`. The 31 raw points retain the small-R collapse regime and the large-R saturation regime, which are excluded from this local fit. K=`19027841`; K/R is a load ratio only. Collision rate was not measured.

## Epoch-length axis
- Quadratic descriptive fit in ln(L4 multiplier): vertex `0.411502×L4`, predicted gap `2.648528`, R² `0.774155`; this summarizes the observed U-shape and is not a mechanistic law.
- Long replay: `s1v5_128_ep_tri_1xL4_10ep` reaches gap `8.6752` at epoch 10 / step 3370; the epoch-boundary increments peak early and decline toward about `0.7–0.8` gap per epoch. This supports gradual concavity / slowing, not a demonstrated plateau. The matched no-gram control reaches `0.4802`.

## Dose axis
- Positive-gap points through 5× only: log-log slope `-1.726983`, R² `0.887225`, rank correlation `-1.000000`; the sign change is bracketed between 5× and 6×, so no global power law is reported.

## Exact-frequency axis
- bigram: token-mass-weighted geometric-bin log-log slope `-0.252746`, R² `0.997165`, rank correlation `-1.000000`, n=`7`; positive-gap bins only, with exact-f points retained in the CSV.
- trigram: token-mass-weighted geometric-bin log-log slope `-0.318121`, R² `0.995548`, rank correlation `-1.000000`, n=`7`; positive-gap bins only, with exact-f points retained in the CSV.

## Files
- `s1_table_size_points.csv`: one row per formal bigram-R/trigram-R run.
- `s1_epoch_length_points.csv`: one row per formal 3-epoch point.
- `s1_epoch_long_replay_points.csv`: epoch-boundary values for both/no-gram 10-epoch replay.
- `s1_dose_points.csv`: final dose endpoints.
- `s1_dose_frequency_gap.csv`: raw final frequency-bin rows, including token fractions.
- `s1_frequency_exact_points.csv`: shared-context exact-f rows from the formal frequency run.
- `s1_scaling_fits.csv`: descriptive fit coefficients and rank correlations.
- `v5_optimizer_points.csv`: final metrics for the clean 11-arm optimizer refresh.
