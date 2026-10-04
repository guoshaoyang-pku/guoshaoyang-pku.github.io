# Vocabulary provenance audit and matched repair

## Finding and Provenance
exp:59726980c70d genuinely used model vocabulary 32 on data vocabulary 24; it was not an exact V24 replication. All 12 original candidates were matched to allowed-lab candidate_spec.json, executable model.py, and results/summary.json by model configuration, budget, and exact reported mean. These receipts say execution=candidate_py_files. GRUs construct Embedding(32,64) and Linear(64,32); SGD constructs Embedding(32,64), position Embedding(24,64), Linear(64,32); Adagrad constructs Embedding(32,32), position Embedding(24,32), Linear(32,32). train.py calls Model() directly and computes CE using pred.shape[-1], without slicing logits to data vocabulary. Thus the available executable provenance rules out a silent override to24.

The source-matched paths include bg_1d9caf/x_769a645f46 (original high-rate GRU), x_12699d7f22 (SGD), and x_911b833d8e (Adagrad), under the allowed lab experiments directory. All 12 mappings and sources are preserved in original_provenance.json; repair sources, receipt hashes and seed endpoints are in verified_artifacts.json and sources/. Matching is content-based, not an exposed transport transaction-ID mapping. Direct Torch instantiation was blocked by sandbox ctypes restrictions; construction is established from executable source plus execution receipts, not a newly measured tensor dump.

Frozen load_history/load_kb/load_lab inputs were read; observations were refreshed to e0002 before preregistration and again after results. K1077 was absent from both frozen and refreshed saved KB. Its reported concurrent narrowing cannot be inspected here. This report supplies the correction wording and annotates the related saved K1071 rather than inventing K1077's contents. exp:1b5b7a7fd193 is data/model V32 alpha1.4; comparing its losses with the V24 alpha.8 law is not an isolated alpha intervention.

## Contract and Preregistration
Preregistration was published as R9d96d388c9 before the new experiment. One new set, exp:2ba2012e9989, uses training-visible bigram_lm/bg_1d9caf only: alpha=.8, V24, context24, train800/test200; instance/sequence seed100064931980, table seed100064941980. No evaluation-question data were used for training.

For each explicit model vocabulary24 and32, context24: GRU64x2 no residual/dropout/attention, AdamW betas(.9,.95), wd.001, lr.0003/.003 at T256/T1024; causal transformer64x1 heads4 ff256 SGD lr.003 wd.001 momentum0 at T1024; causal transformer32x3 heads2 ff256 Adagrad lr3e-5 wd1e-4 at T1024. Batch32 throughout, nominal seeds0..9. Samples seen are8192 at T256 and32768 at T1024. Stored inherited momentum/betas irrelevant to an optimizer are not execution arguments: optimizer.py confirms the stated AdamW/SGD/Adagrad calls.

Predictions made before measurement: at T1024 high-rate GRU has worse CE than low-rate GRU and weak SGD. Adagrad reversal was not guaranteed. No direction was predicted for V24-minus-V32. There are12 candidate conditions and120 seed endpoint records, but not120 independent fresh replications: the six V32 conditions reproduce the old means/SD exactly and may reuse deterministic cached results.

## Results and Uncertainty
CE is unregularized held-out token cross-entropy in nats, lower is better. Source-derived model dimensions are Vx64 embedding and Vx64 head weight for GRU/SGD, Vx32 for Adagrad; transformer positions are24x64 or24x32. Both embeddings and heads change with V, so this is a vocabulary-dimension intervention, not a head-only intervention.

All-seed mean +/- sample SD follows. Each entry has10 finite endpoint values. Pointwise95% mean half-width is t9*SD/sqrt(10); these quantify seed variability conditional on this fixed train/test law, not new-dataset uncertainty.

| Condition | V24 CE +/- SD | 95% half-width | V32 CE +/- SD | 95% half-width |
|---|---:|---:|---:|---:|
| GRU .0003 T256 | 2.993145 +/- .005586 | .003996 | 3.034409 +/- .007765 | .005555 |
| GRU .003 T256 | 2.946757 +/- .003741 | .002676 | 2.926591 +/- .004615 | .003301 |
| GRU .0003 T1024 | 2.895730 +/- .002359 | .001688 | 2.894268 +/- .002639 | .001888 |
| GRU .003 T1024 | 3.996647 +/- .033581 | .024023 | 3.703194 +/- .039110 | .027978 |
| SGD T1024 | 3.138486 +/- .013714 | .009810 | 3.261408 +/- .022839 | .016338 |
| Adagrad T1024 | 3.294227 +/- .008989 | .006430 | 3.537140 +/- .036134 | .025848 |

Important transport/reporting correction: repaired V24 high-rate GRU candidate24117cfb is excluded=true in its underlying summary, with6/10 flagged failed (seeds0,4,5,6,7,8). Its tool-returned3.961416 +/- .017093 is the mean and population SD of ONLY four unflagged seeds, not10. The other11 candidates have0/10 flagged failures. All ten high-rate endpoints are finite (3.932565 to4.043258); six flagged values exceed4 while four unflagged values are below4, consistent with endpoint threshold failure. The actual supplied threshold is not recorded in the exposed spec. Failed curves are entirely NaN-masked; their training completion cannot be verified from curves. These flags are not evidence of numerical divergence. The table transparently uses every recorded finite endpoint, not just survivors; official selected mean is preserved separately.

Paired seed differences at V24 (high-rate T1024 minus comparator), mean with pointwise95% t9 half-width: low-rate +1.100918 +/- .022977; SGD +.858161 +/- .017680; Adagrad +.702420 +/- .023955. Each inequality holds10/10 endpoint pairs, also4/4 restricting to unflagged high-rate seeds. Thus both preregistered tests pass, and the optional Adagrad reversal occurs. At T256 high rate is better than low rate by.046388 mean CE: no universal rate penalty.

V32 high-minus-low is+.808926 +/- .027500, high-minus-SGD+.441786 +/- .028031, high-minus-Adagrad+.166054 +/- .042282; each10/10. V24-minus-V32 high-rate T1024 is+.293453 +/- .022271, whereas SGD is-.122922 +/- .012888 and Adagrad-.242913 +/- .026208. Low-rate T1024 difference+.001462 +/- .002437 is unresolved. Vocabulary mismatch affects magnitudes materially but does not cause the observed long-budget ordering reversal.

## Explanation, Scope and Tests
The V24 ordering among measured T1024 conditions is low-rate GRU < SGD < Adagrad < high-rate GRU. The endpoint evidence supports harmful high-rate long-budget performance, not optimization divergence. Competing explanations include excess fitting of fixed training transitions, overconfident predictions, optimization instability, and extra unused output-class normalization effects. Changing V also changes parameter count, initialization and the random-number stream; its effect cannot be attributed only to unused logits or extra head rows.

Discriminating future tests, not performed here: record training CE, calibration and gradient/parameter norms across checkpoints; instability predicts spikes/nonfinite optimization statistics, fitting predicts declining training CE with rising held-out CE. A shared V32 backbone with logits masked to24 versus an unmasked32 head, aligned initialization/minibatches, would separate softmax normalization from embedding/initialization effects. New allowed independent V24 laws would test generalization beyond one dataset. None of those outcomes are asserted in this report.

Correction/refinement intended for K1077: replace any description of exp:59726980c70d as exact V24 replication with dataV24/modelV32. Cite exp:2ba2012e9989 as explicit executable modelV24/context24 evidence supporting the scoped T1024 high-rate penalty and SGD reversal; disclose6/10 high-rate exclusion flags and the four-seed selected tool mean. Do not infer an alpha threshold, universal last place, or divergence. Adagrad wins on this measured law, not as a guaranteed general rule.

q_7dbb0e is historical context, not the training dataset. This repair does not measure its full five-choice order or its additional transformer rival, nor establish exact unseen-instance losses. Reproducibility: analyze_provenance.py regenerates analysis_results.json from saved receipt endpoints; sources and JSON snapshots are committed with this report. Remaining uncertainties are transport receipt-ID linkage, exact endpoint failure threshold, failed-run trajectories, fresh versus cached controls, and replication across independent allowed laws.
