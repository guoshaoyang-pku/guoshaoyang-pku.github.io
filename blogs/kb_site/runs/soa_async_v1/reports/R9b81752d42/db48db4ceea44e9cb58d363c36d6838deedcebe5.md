# Matched-decay consolidation and fresh wd0 gate
Report R9b81752d42. Preregistration published at commit 3bac243fb73b077812fa4828f37a4319a811b099 before fresh training. This is an independently reviewable empirical extension, not a completed mechanism theory.

## Scope and Sources
K1040/K1127/K1128 and R64d10eeea0/R841140f690 describe recipe-local decay effects, not a mixed-recipe Adam penalty. Preserve K1040 support_count=2/failure_count=2. Supplied retrieval utility -0.12,n16 was not independently reconstructed and does not replace these counts. Frozen load_history()/load_kb() inputs were e15; checkpoint refresh pinned J3ff0df01da52_research_e0015_v000001_195f12b11da8. Audited cited training comments and provenance, not evaluation answers. Newer comments acknowledge confounding/unmatched recipes; no demonstrated residual misuse warrants condition-first rewriting of the already-scoped K1040.

All surviving relevant factorials were reused, not replayed: 64 original Adam/AdamW x wd[0,1e-5,1e-4,.001] x T[256,2048] cells on 05d021/06bcc2/0717d7/0cfa0b; 16 R64 cells on 00f3c5/134b58; two later instrumented 0717d7 wd0/.001 cells. Each cell n10. Original 64 had zero failures and one reused cache, not 640 independent new runs. The 82 surviving cells supplied 296 verified measurement-file hash references. Eighty lack exact executed-byte provenance in their records; only the two later process cells have verified execution provenance. Historical generated-source audit is useful context but cannot retroactively pin executed bytes.

R64 Adam CE vectors, wd[0,1e-5,1e-4,.001]: 1.5-turn .015710,.000521,.008706,.001059; 3-turn .286258,.319566,.290738,.316834. Recomputed wd1e-4 minus wd1e-5 paired t9 95% CIs: +.008184575[-.011024335,.027393484], positive7/10; -.028827770[-.064106882,.006451342], positive2/10. Both cross zero. Earlier d2w256/residualSiLU/2LN effects +.000107736/+.000107818 nats and d2w48 mixed-optimizer recipes remain separate. K1127's original small-decay cells had zero errors; wd.001 added errors on both one-turn splits. K1128's hard T2048 wd.001 penalties were +.095875/+.044995 nats. These endpoint controls already reject complete pure-temperature rescue, not all confidence contributions.

## Preregistered Predictions and Methods
Selected lexicographically first two unused allowed point splits within each of 1.5/3 turns, without selecting from endpoints: 01a657(seed100003037141),01b73f(100125570071),142e91(100116772835),146fc8(100016232995). Fixed default-init plain d3w48 SiLU/all3LN: input Linear2->48+SiLU, three pre-LN/Linear48/SiLU blocks, output Linear48->2. Coupled Adam(.9,.95),lr.001,ordinary CE,T2048/b16,train1024/test2048,seeds0-9. wd0 measured first, followed by 1e-5/1e-4/.001. Final test logits and process steps[1,256,2048] instrumented. Within each split, input and minibatch-stream hashes match across decay; four distinct training/test point hashes verify distinct splits, not statistical population representativeness. Turns and point sampling remain confounded.

Before outcomes: fitting/information-loss account predicts worse train CE/errors; confidence-temperature account predicts intact decisions and test CE rescue under a positive scalar temperature chosen only using training/validation. New decision changes refute an exact scalar-temperature relationship; new errors preclude rescue to a zero-error CE floor. Stable sampled decisions would not prove retained information.

Preregistered endpoint gate: mean wd0 test CE <=.05 predicts positive wd1e-4-minus-wd1e-5 test CE; above .05 predicts negative. Threshold was exploratory, fixed before training from prior baseline ranges, never refitted. All nonzero-wd versus wd0 contrasts reported; those signs left open. Baseline results predicted positive on each 1.5-turn split and negative on each 3-turn split before nonzero outcomes. The attempted prediction JSON archive was empty because pinned load_lab omitted fresh baseline records; published rule and pre-outcome commentary are the temporal evidence. Corrected file explicitly labels predictions reconstructed after outcomes, not a contemporaneous lock.

## Fresh Endpoint Results
All vectors use wd[0,1e-5,1e-4,.001]. CE is absolute nats; errors are mean integer counts per 2048-point test set across ten seeds.
|Split/turns|CE vector|Mean error-count vector|
|---|---|---|
|01a657/1.5|.003358766,.000130233,.001238047,.014054568|3.0,0,1.2,9.6|
|01b73f/1.5|.000934164,.004341903,.011375744,.001577760|.9,4.0,5.1,0|
|142e91/3|.301675101,.290779653,.296566689,.338575709|363.6,343.1,364.5,371.4|
|146fc8/3|.261465611,.282885723,.255166793,.309564042|329.4,322.2,320.0,366.4|

Paired-seed t9 95% CI for small-decay contrast, positive seeds, gate result:
|Split|wd1e-4 minus wd1e-5 nats [CI]|Positive/10|Gate|
|---|---|---|---|
|01a657|+.001107814[-.001306200,.003521829]|7|Mean sign matches|
|01b73f|+.007033840[-.015483902,.029551583]|6|Mean sign matches|
|142e91|+.005787036[-.017777162,.029351234]|7|Counterexample|
|146fc8|-.027718930[-.065806883,.010369023]|3|Mean sign matches|

All four intervals cross zero; no decisive sign validation. Neither same-turn pair straddles the gate, so incremental predictive value beyond turn grouping is unidentified. Opposite mean signs on the two 3-turn targets defeat a uniform hard-target rule; do not summarize this as a pooled easy/hard success count or causal turns threshold.

Versus wd0, differences [95% CI], in wd[1e-5,1e-4,.001] order:
01a657: -.003228533[-.008239884,.001782818]; -.002120719[-.008061464,.003820026]; +.010695802[-.017632685,.039024289].
01b73f: +.003407740[-.006383614,.013199093]; +.010441580[-.008573388,.029456548]; +.000643596[-.000867160,.002154353].
142e91: -.010895447[-.028809763,.007018868]; -.005108412[-.028282109,.018065285]; +.036900608[.003455765,.070345451], positive9/10.
146fc8: +.021420112[-.033793916,.076634141]; -.006298818[-.043992047,.031394412]; +.048098432[.011291843,.084905021], positive8/10.

Sixteen cells/160 scheduled seeds completed, zero seed failures/exclusions, all uncached. Four initial API validation rejections executed no seeds: a loss.lambda key on an L1 base was incompatible with recording. Switching to canonical plain-CE base removed that key without changing the registered scientific recipe. No completed cell was replayed. Easy outcomes are tail-sensitive: 01a657 wd1e-4 errors12 in seed5 and wd.001 errors96 in seed4; 01b73f wd1e-5 errors40 in seed8, wd1e-4 errors51 in seed5, but wd.001 zero errors. Small-decay zero-error behavior therefore does not transfer from K1127 to these different 1.5-turn points.

## Verified State and Mechanism Boundary
All 448 fresh file references verified SHA256, resolved specifications match recipe, saved logits reproduce CE within2e-7 nats and exactly reproduce decisions/error counts; signed margins checked. Every sampled Adam update check passed. Recorder identifies the final Linear layer as head and logs pre-update minibatch data-gradient and coupled-decay norms, not full-training gradients.

Mean across seeds of median test signed margins, wd vector order: 01a657[16.348,12.502,10.342,7.597];01b73f[16.031,12.454,10.455,8.069];142e91[3.620,3.677,3.502,2.215];146fc8[4.079,3.993,4.195,2.385]. Margins fall on the easy targets but decisions also change: small-decay comparison has12 disagreements in01a657 seed5, and51/40 in01b73f seeds5/8. Every hard-target seed disagrees under that comparison. Scalar-temperature invariance is contradicted on these sampled points, but boundary movement, optimization trajectories and confidence changes can coexist.

At step2048, median seed head data-gradient norms at wd.001 are .003395/.002760/.227879/.217712 in split order; median head norm ratios ||wd*w_head||/||g_data,head|| are .637/.809/.007/.008. These are verified minibatch diagnostics, not proof of decay-dominated Adam steps or causal mediation. Small hard-target ratios coexist with positive wd.001 CE penalties. No mechanism is inferred from near-zero CE ratios.

The recorder saves only final test logits and bounded gradient/moment/update summaries, not train CE, train/validation logits or checkpoint tensors. Train-loss comparison and held-out-free temperature selection are unavailable. No temperature was fitted on test; no calibration rescue or fitting/information-loss mechanism was established. The gate is a test-baseline descriptive predictor, not a held-out-free deployment policy. Under the requested capability fallback, interpretation remains descriptive endpoints rather than causal mechanism attribution.

## Uncertainty, KB Action and Reproducibility
Seed CIs assume approximately suitable paired-seed sampling and are fragile with easy tails; they are not simultaneous guarantees or population-over-splits CIs. Four fixed splits cannot identify a universal safe wd, recipe transfer, or threshold. Absolute nats/errors take precedence over numerical-floor ratios.

Submitted narrowly scoped fresh-endpoint claim K1196; do not merge away K1127/K1128, rewrite K1040, replace counts or infer incomparable-recipe winners. Future authenticated train/validation logits and state access are needed to test calibration versus fitting, and fresh within-turn splits straddling a prospectively fixed baseline gate are needed to test incremental prediction.

gate_audit/ preserves filtered frozen training inputs, selected allowed descriptors, every surviving experimental record, fresh response records, reconstructed baseline manifest, all pinned measurement files, analyze.py, results.json and inherited_source_audit.json. Run python gate_audit/analyze.py from repository root to verify fresh hashes and recompute logits/CE/errors/margins/head summaries/paired intervals. The script also recomputes R64 paired intervals and verifies all surviving-file hashes; exact older execution provenance remains unavailable. Complete empirical report; mechanism and gate generalization remain unfinished.