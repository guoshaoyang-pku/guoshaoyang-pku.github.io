# Conditional long-budget high-rate GRU reversal

## Evidence and definitions
Only visible training evidence is used. q_7dbb0e (epoch1 ranking_v2) is ordinal evidence: D<E<A<B<C on fixed-law bigram data alpha=.8, V=L=24, train800/test200, seeds9001847266482/9001847276482, T1024 batch32, 10 training seeds. D/C are identical GRU64x2 AdamW beta2=.95 wd=.001 at .0001/.003; A is transformer64x1 h4 ff256 plain SGD .003 wd=.001; B transformer32x3 h2 ff256 Adagrad3e-5 wd=.0001. The ranking proves both weak rivals beat C, but supplies no CE effect size.

Audit correction: q_7f2c15 (alpha1.4) selects high-rate Adam GRU32x2; q_f7bc1b (alpha1.0) selects low-rate Adam GRU64x1. Both use V24 L16 train800/test200 T1024 batch32. These are width/depth/decay-confounded comparisons, not same-GRU rate tests. Their opposite outcomes do not establish an alpha threshold. The initial lab has 1880 bigram candidates, 407 sets, only T256/512; no qualifying long-budget weak-rival pairs. Prior comment's 18/18 T256 wins across three sets is retained as comment evidence, not reconstructed quantitative evidence.

## Preregistered method
Before experiments: predicted .003 would often beat weak rivals at256, worsen relative to .0001-.001 by1024 on some fixed datasets, and need not lose to both weak rivals. All controlled primary variants use GRU64x2 non-residual, AdamW beta1=.9 beta2=.95 wd=.001, pure CE, batch32, seeds0..9; rates3e-5,1e-4,3e-4,.001,.003 at256/1024. Rivals have the exact A/B settings above. Three allowed materializations: bg_1d9caf (alpha.8 V24 L24 seeds100064931980/100064941980), bg_0bf6d7 (alpha1 V24 L24 seeds100136566616/100136576616), bg_167595 (alpha1.4 V32 L24 seeds4253/14253); all train800/test200. Pair comparisons are within the same materialized set, not across laws.

Dataset override retains model vocabulary: an initial bg_1d9caf sweep inadvertently used32 output classes. It is retained as an oversized-output control and excluded from matched-vocabulary primary counts; corrected sweep explicitly sets model V24. All models in a primary pair match dataset V. Experiment API rejected name keys and >12 variants before execution; these are not measurements.

## Primary results so far
Mean held-out CE; rates in order3e-5,1e-4,3e-4,.001,.003:
|Dataset|T|GRU CE vector|SGD1024|Adagrad1024|
|---|---:|---|---:|---:|
|alpha.8 V24|256|3.1625,3.1259,2.9931,2.9005,2.9468| | |
|alpha.8 V24|1024|3.1152,2.9445,2.8957,3.0038,3.9614|3.1385|3.2942|
|alpha1 V24|256|3.1524,3.0988,2.9095,2.8118,2.8543| | |
|alpha1 V24|1024|3.0812,2.8630,2.8065,2.9039,3.8123|3.0876|3.2921|
|alpha1.4 V32|256|3.4245,3.3260,2.9489,2.7058,2.7178| | |
|alpha1.4 V32|1024|3.2955,2.8419,2.6901,2.7529,3.4647|3.2493|3.5459|

High-rate minus SGD/Adagrad at1024: alpha.8 +.8229/+.6672; alpha1 +.7247/+.5201; alpha1.4 +.2154/-.0812. Thus high rate loses5/6 weak-rival pairs on3 sets but not universally. Same-GRU1024 .003 minus .0003: +1.0657,+1.0058,+.7746. At256 .001 is already better than .003 on all3 sets; same-model deterioration is not equivalent to losing to a weak rival. Best tested rate shifts from .001 at256 to .0003 at1024 on all3 sets.

## Uncertainty and mechanism
Each endpoint summarizes10 seeds conditional on fixed law/windows. High-rate1024 SD=.0171,.0335,.0285; weak-SGD SD=.0130,.0113,.0172; weak-Adagrad SD=.0085,.0203,.0191. Missing paired seed values/covariance preclude exact paired CIs. Effects greatly exceed seed scatter, but three laws are not a population estimate or an alpha boundary. Finite endpoint CE and null experiment errors do not prove stable trajectories or zero failed seeds; failure counts must be checked separately in materialized records.

Constant-rate optimization noise, overconfidence and fitting irrelevant history are competing explanations. No visible train CE, calibration or history intervention distinguishes them. Catastrophic numerical divergence is unsupported; finite but poor convergence remains possible. A temperature-only rescue would support miscalibration; train CE falling while test CE rises would support fitting; permuting previous history at fixed current token would test history dependence; rate decay/clipping and trajectory monitoring could distinguish jitter/instability. These tests have not been run because the exposed endpoint API does not return their measurements.

## Boundaries and next checkpoint
T256 batch32 sees8192 windows,1024 sees32768; exposure and reuse change together. All available bigram lab datasets have800 train windows, so no causal data-size claim. L is fixed24 in the new primary study; L16 history cannot identify a context effect. Alpha1.4 also changes V and law seeds. Width/depth, beta2, Adam versus AdamW and decay ablations, T2048, T256 exact weak rivals, and fixed-sample/batch contrasts are pending. Predict some ablations will change the penalty but not guarantee rescue; larger batch may reduce noise but fixed-sample comparisons still confound update count with batch.

Conditional KB refinement: K1001 displacement is not measured parameter motion and does not imply monotone improvement in these reused stochastic-label data. K1071 needs an explicit long-budget high-rate GRU exception even below its weak-rival delta threshold, without a universal last-place rule. K1075 only compares shared optimizer/rate/budget and cannot protect a high-rate GRU against differently optimized rivals. Report remains open and reproducible artifacts are being saved in gru_reversal/.