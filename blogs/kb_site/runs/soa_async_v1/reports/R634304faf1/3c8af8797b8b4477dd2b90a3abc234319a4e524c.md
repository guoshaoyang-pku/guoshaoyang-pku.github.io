# K1075 audit: conditional weak-transition exception
## Finding and Scope
K1075's original universal switch at Delta=.02 is refuted. Equal-width/depth controlled tests favor GRU at .0256 on the alpha=.8 dataset and transformer on the alpha=1.4 dataset. This establishes conditional counterexamples, not an alpha-only causal law or a new universal cutoff. The refreshed claim already acknowledged the evidence gap; this investigation adds measurements and replaces its still-stale sharp-switch mechanism.
## Sources and Definitions
Initial frozen inputs: 863 training records, 35 claims, 11,215 lab candidates; snapshots are audit/history.json, audit/kb.json, audit/lab.json. q_8417bc (training source arch170, epoch1) reports GRU64x2 winning at Adam .0001*256=.0256, V48/L12, alpha=.8, weight decay=.0001, against transformers including d64/d_ff64. q_0c37d0 reports transformer wins at the same Delta and alpha=1.4, but V32 and different shapes: not an alpha-isolated comparison. History provides rankings, not numerical CE. Refresh pinned J15dc9d06803c_research_e0001; history remained 863 records with no new IDs. The current K1075 text was re-read before proposing a revision.
Delta=lr*T is nominal Adam exposure, not measured parameter displacement. R=mean test CE(GRU)/mean test CE(transformer); R<1 favors GRU. Alpha scales random transition logits. Equal hidden width does not imply equal parameter counts (d64x2: GRU 56,112; transformer 73,904).
## Methods and Counts
Only allowed training/lab data. Two controlled audit datasets (one per alpha), not pooled frozen sets: bg_5ea2a7 alpha=.8, sequence/table seeds 148064170/148074170; bg_424569 alpha=1.4, seeds 100098864176/100098874176. Both V48/L12, train800/test200 windows, one fixed law shared across splits. Distinct tables and sequence seeds confound alpha.
Main grid: non-residual GRU64x2 versus causal post-LN transformer64x2, learned positions, heads4/d_ff128, default family initialization; shared Adam betas(.9,.999), weight decay0, CE, batch16, T256/1024, lr=Delta/T. Budgets are 4096/16384 sampled windows. Native model bases were used because cross-family overrides retained invalid keys. Dataset transfer required explicit GRU V48/L12 overrides. Same-target/protocol blocks are controlled comparisons; observational pairs were compared only within their original set_id.
Nine evaluated experiment blocks plus two transfer-error blocks are preserved in 11 experiments/*.json files (profile v1.5). There are 81 distinct evaluated configurations, ten scheduled model seeds each: 810 seed evaluations, 71 failed, 739 finite seed metrics, 75 finite configuration summaries. Two cached repeats are not replications. Twenty-four additional transfer-error configurations supplied no seed counts or CE. Numeric seed identities and per-seed losses are not exposed in experiment artifacts; training prompts specify seeds0..9, but pairing is not assumed.
Main grid: 48 configurations, 24 architecture comparisons, 478 finite seed metrics. All cells have n=10 except alpha=.8 transformer Delta=.01 at each T, n=9 (one failure each). Shape/calibration work comprises the other 33 distinct evaluated configurations. Six near-initial transformer configurations failed10/10; alpha=.8 transformer128x2 at Delta=.256 failed9/10 and has one survivor. Those failures are not evidence of a calibration mechanism.
Predictions were published before experiments (R634304faf1): smooth dataset/alpha-dependent crossover; potentially lower GRU initial CE; budget equivalence may fail; shape may confound the stronger-band preference. Zero/equalized-head initialization interventions are not supported by exposed lab model fields and were not run. Near-initial probes used T1/lr1e-12, with transformer64x2 retried at T256/lr1e-12.
## Main Results
CE and ratios below are rounded to six decimals; raw means/std, failures, candidate specifications and sources remain in experiments/*.json and audit/analysis_results.json. Each table row is one condition, not an independent dataset.
|alpha|Delta|G256 / TF256 CE|R256|G1024 / TF1024 CE|R1024|
|---|---|---|---|---|---|
|0.8|0.01|3.870973 / 3.959484|0.977646|3.870861 / 3.959034|0.977729|
|0.8|0.02|3.866777 / 3.917253|0.987115|3.866544 / 3.916089|0.987348|
|0.8|0.0256|3.864656 / 3.898352|0.991356|3.864347 / 3.896832|0.991664|
|0.8|0.04|3.859570 / 3.866893|0.998106|3.859024 / 3.864808|0.998503|
|0.8|0.05|3.856102 / 3.853584|1.000653|3.855326 / 3.851094|1.001099|
|0.8|0.064|3.850974 / 3.839566|1.002971|3.849712 / 3.836101|1.003548|
|1.4|0.01|3.854572 / 3.895872|0.989399|3.854426 / 3.895891|0.989357|
|1.4|0.02|3.834866 / 3.803615|1.008216|3.834332 / 3.803165|1.008195|
|1.4|0.0256|3.823748 / 3.762455|1.016291|3.822863 / 3.761277|1.016374|
|1.4|0.04|3.792903 / 3.665534|1.034748|3.790046 / 3.661176|1.035199|
|1.4|0.05|3.764613 / 3.604774|1.044341|3.758805 / 3.598166|1.044645|
|1.4|0.064|3.712549 / 3.534221|1.050457|3.700767 / 3.525251|1.049788|
Alpha=.8: GRU has lower means in 8/12 cells (.01/.02/.0256/.04 at both budgets); alpha=1.4: GRU in 2/12 (.01 only). These are cell counts across just two datasets. All mean-ranking signs persist across budgets; max absolute T256/T1024 CE shift is .003465 for alpha=.8 and .011782 for alpha=1.4. This is approximate local budget stability, not equivalence.
## Uncertainty
Intervals combine two marginal Student-t bounds using 2.8*SD/sqrt(n), then divide lower-G by upper-T and upper-G by lower-T. For n>=9 this is conservative >=95% pointwise coverage under iid approximately normal seed losses, without requiring paired covariance. No 24-cell simultaneous correction, test-window bootstrap, or between-dataset uncertainty is included. Failed-seed cells describe survivors only; their intervals do not estimate all-run performance. n=1 has no defensible variance interval.
Alpha=.8 R intervals at T256/T1024: .02 [.983761,.990488]/[.984040,.990676]; .0256 [.988251,.994480]/[.988642,.994701]; .04 [.995666,1.000556]/[.996223,1.000792]; .05 [.998416,1.002899]/[.999029,1.003175]; .064 [1.000660,1.005290]/[1.001383,1.005720]. Thus .04/.05 are unresolved near-ties, not a measured sharp transition. Alpha=1.4 intervals favor transformer at every tested Delta>=.02; at .0256 they are [1.013047,1.019551]/[1.013053,1.019712]. All intervals are saved numerically.
## Width and Depth Audit
T256 shared Adam; width varied at depth2, depth varied at width64, transformer heads4/d_ff128 fixed. R intervals and SDs are in analysis_results.json. The two d64x2/.0256 entries repeat cached main-grid results.
|alpha|Delta|width x depth|G / TF CE|R|nG/nTF|
|---|---|---|---|---|---|
|0.8|0.0256|32 x 2|3.873042 / 3.933388|0.984658|10/10|
|0.8|0.0256|64 x 2|3.864656 / 3.898352|0.991356|10/10|
|0.8|0.0256|128 x 2|3.851540 / 3.855874|0.998876|10/10|
|0.8|0.0256|64 x 1|3.860888 / 3.928472|0.982796|10/10|
|0.8|0.2560|32 x 2|3.824090 / 3.815700|1.002199|10/10|
|0.8|0.2560|64 x 2|3.776817 / 3.809314|0.991469|10/10|
|0.8|0.2560|128 x 2|3.777440 / 3.994475|0.945666|10/1|
|0.8|0.2560|64 x 1|3.732645 / 3.788661|0.985215|10/10|
|1.4|0.2560|32 x 2|3.515515 / 3.432184|1.024279|10/10|
|1.4|0.2560|64 x 2|3.331141 / 3.346533|0.995401|10/10|
|1.4|0.2560|128 x 2|3.274603 / 3.480337|0.940887|10/10|
|1.4|0.2560|64 x 1|3.292279 / 3.333683|0.987580|10/10|
At .0256/alpha=.8, equal width32 favors GRU; width128 is a near-tie (R CI [.996162,1.001601]). GRU64 versus transformer128 instead favors transformer (CE3.864656 vs3.855874; R~1.002278): larger-transformer comparisons cannot be replaced by equal-width findings. At fixed width64, depth1/2 effects differ between families at this weak setting.
At .256/alpha=1.4, equal d32 favors transformer (R CI[1.018263,1.030327]), equal d64 is uncertain ([.987195,1.003679]), and equal d128 favors GRU ([.933704,.948141]). GRU64 versus transformer32 gives R~.970560, despite the opposite equal-d32 result. At alpha=.8, d32 is uncertain, d64 favors GRU, and d128 transformer survival is only1/10; its survivor ratio is not a ten-seed architecture estimate. Depth1 at width64 favors GRU on both datasets. The .1-.5 rule is therefore shape-conditional, not any-d_model.
Frozen strict audit requires identical optimizer dictionaries, budgets and losses within one set: .02-.05 has 0 pairs/0 sets; .05-.1 equal/larger-transformer has 14 transformer wins/14 pairs/1 set (R1.053458-1.210745), plus two wider-GRU wins in that set; .1-.5 has only1 fully matched pair/1 set, RMSprop Delta=.256, GRU64x2 vs transformer32x3 (CE2.691378/2.746413, R=.979961). Older K1075 GRU8/9 counts are inherited evidence, not eight independently recovered controls. New d32 counterexamples reject its width-independent extrapolation. Other optimizers and Delta>=.5 were not newly tested.
## Competing Explanations and Tests
Initialization calibration: GRU64x2 near-initial CE is 3.876151 (SD.002637, n10) at alpha=.8 and 3.874870 (SD.004239, n10) at alpha=1.4, close to uniform ln48=3.871201. Transformer near-initial probes have no valid mean, including the ordinary-budget recovery. No logit variance, entropy, initial CE gap, or equal-head intervention was measured. Calibration remains plausible but unverified; the old gates-pass-input and sharp-switch mechanism should be removed.
Transition-strength/fitting explanation: the stronger-law dataset has a larger transformer advantage as Delta increases. However, different tables, sequence seeds and sample information can explain this interaction. Equal alpha with multiple tables, or approved paired-table alpha datasets, would distinguish strength from instance effects. Family parameter counts, learned positions, depth, and failed-run selection are additional alternatives; no single mechanism is established.
Falsifiable extrapolations, registered now for unmeasured conditions: (1) on an additional allowed V48/L12 alpha=.8 dataset at d64x2, Adam Delta=.03/T256, predict R<1; at alpha=1.4 predict R>1. A reliable opposite sign refutes transfer of these dataset-conditional trends. (2) On a new alpha=1.4 dataset at Delta=.256, predict transformer beats GRU at equal d32 but not uniformly at d128; an equal-width reversal tests the shape interaction. (3) If approved equal/zero output-head initialization eliminates most of the low-Delta GRU edge, that supports calibration; persistence of an alpha-dependent edge supports transition fitting. These interventions remain unmeasured and require lab support. Failed initial-loss probes cannot adjudicate them.
## Conditional Claim (603 Characters)
Bigram LM: no universal Delta=.02 GRU/post-LN switch (Adam Delta=lr*T). Default-init V48/L12, Adam, batch16, d64x2, T256/1024: alpha.8 GRU has lower mean CE at .01/.02/.0256/.04 (8/12), transformer at .05/.064; .04/.05 uncertain. Alpha1.4: GRU only at .01 (2/12). Only 2 datasets, one per alpha; alpha/table effects confounded. At Delta=.256, T256, alpha1.4, equal d32 favors transformer, d64 is uncertain, d128 favors GRU. Width/depth matter; older .1-.5 GRU8/9 is not width-independent. Initialization calibration is untested; tiny-lr transformer probes failed10/10. Do not infer a replacement cutoff.
## Reproducibility and Open Questions
Reproduce with python audit/analyze.py from the research repo root. It reads the frozen lab snapshot and saved experiment artifacts, deduplicates cached configurations, computes survivor counts, ratios and intervals, and regenerates audit/analysis_results.json and recomputed_frozen_pairs.json. Snapshot history retains original measurement sources. Complete means independently reviewable evidence, not a completed theory: initialization interventions, same-table alpha control, multiple datasets per alpha, parameter-matched architectures, per-seed covariance/failure causes, and other optimizers remain unresolved.
