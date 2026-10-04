# Falsification audit of q_b204a3 (preregistered, ongoing)

## Question and sources
The hypothesis proposes an unspecified interaction score for XOR2 at T256, momentum-free SGD lr .0003: strong rectifier advantage, helpful skips, selective LN rescue with extra LN insufficient for deeper plain smooth stacks, deep smooth zeroLN last. It was formulated after revealing a five-recipe ordinal result; it supplies neither coefficients nor a unique rule for unseen pairs. We do not fit that rank or emit evaluation answers.
Read exact q_b204a3 and q_f74790, frozen K1006/K1018/K1039/K1065/K1133 and reports Refa052dfdc/Reb8454abdf. K1039 is MSE-only and provides no XOR CE evidence. K1018/K1065 chiefly scope higher-dimensional adaptive regimes, not this XOR2 SGD recipe. K1133 optimizer-parity controls and the two prior reports are different recipes, not replications.

## Definitions and controlled test
Use allowed XOR2 splits xorcls_12d3d2 and xorcls_13a317, selected lexicographically after excluding 076df5 already used by K1133. Fresh means new recipe measurements, not a claim these points have never appeared in the lab. Standard-normal two inputs, sign-disagreement labels, 1024 train/2048 test. Fixed width192, default PyTorch Linear initialization, 2-logit linear head without activation, CE. Depth2/4 excludes input projection. Block=activation(Linear(LN(x)) + x if residual); no post-head LN. OneLN means terminal block input LN. SiLU/GELU/LeakyReLU(.01) crossed with plain/residual and zero/terminal/allLN. SGD lr .0003,m0,wd.001,T256,b16, ten seeds0..9; exact focal optimizer/budget. Full 36-cell factorial per split, no hidden target reconstruction.

## Predictions fixed before measurements
P1 Leaky has lower mean CE than each smooth activation at every matched depth/residual/mask on each split (24 contrasts per split). Strong advantage must survive this direct interpretation.
P2 residual lowers CE at each matched depth/activation/mask (18 per split).
P3 terminalLN lowers CE relative to zeroLN in smooth stacks (8 per split).
P4 plain depth4 smooth allLN does not beat matched depth2 terminalLN (2 per split), operationalizing the extra-LN/depth assertion.
P5 depth4 smooth zeroLN is worst among the same activation's depth/residual/mask cells: every competitor has lower CE (22 per split).
These directional operationalizations are stronger than an undefined score and will be reported separately, including counterexamples and tiny margins. No pooled scalar from ranks.
Backtest same predictions on frozen nonexcluded zero-failure within-set lab pairs excluding focal record; separate exact full optimizer/budget matches from type/lr-only and bundled architectures. Historical ordinal comparisons have no numerical margins and will not be pooled with CE. Preserve actual dependency counts.

## Uncertainty and boundaries
Paired seed differences and unadjusted t9 95% intervals conditional on fixed data; split counts, unique configurations, shared anchors, failed seeds and cached reuse must be explicit. Initialization distributions are exact, but equal seed need not mean common minibatches across parameter counts. Inspect measurement curves for baseline; no step0 inference from update1. Mechanism remains report-only without gradient/scale diagnostic intervention.
If primary counterexamples require a boundary check, preregister lr .003 retaining all else; no XOR8 unless dimensionality ambiguity remains. No universal LN/residual/activation law is proposed. Frozen inputs saved compressed; initial uncompressed snapshot exceeded per-file limit and was removed. Results pending.

## Primary checkpoint and boundary preregistration
The 72 unique primary cells completed 720 finite seed runs, zero failures. P1 succeeds24/24 per split; P2 18/18; P3 8/8; P4 fails2/2 per split; P5 succeeds35/44 per split and fails9/44. Preregistration P5 denominator was misstated22: each smooth activation has two possible deep-zero anchors, making44 comparisons, with duplicate reciprocal anchor comparisons; retain all dependencies. Deep plain allLN minus shallow plain terminalLN margins for SiLU/GELU: -.092638/-.148068 on12d3d2; -.098824/-.154059 on13a317, all paired95% intervals below zero and10/10 seed wins.
Boundary prediction before running: at lr .003, depth4 plain smooth allLN still beats depth2 plain smooth terminalLN; depth4 residual smooth zeroLN still beats depth4 plain smooth zeroLN. Test only these ten anchor cells per split (two activations x five recipes), no activation-law boundary claim. Width, budget, decay, initialization/head and data unchanged. XOR8 unnecessary for falsifying a XOR2 assertion.
Refresh pins Je8247e6fbbc4_research_e0010_v000001_90e2647e9082. Current evidence still requires historical eligibility audit and full phase table. API rejected >12 variants and an unsupported name override before training; six accepted batches supplied the factorial. Re-fetch of the first twelve cells was cached receipt recovery, not fresh replication.
