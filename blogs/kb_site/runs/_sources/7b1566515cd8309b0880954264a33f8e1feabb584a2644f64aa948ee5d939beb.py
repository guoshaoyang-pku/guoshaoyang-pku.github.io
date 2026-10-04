import json, glob, numpy as np
from pathlib import Path

datasets = [
 ("S1","stabcls_0db74c","sparse8",33092035,33093035),
 ("S2","stabcls_17011f","sparse8",145092376,145093376),
 ("A1","stabcls_2173ed","smooth16",108092265,108093265),
 ("A2","stabcls_28e027","smooth16",34433205,34434205)]
labels = dict((d[1],d[0]) for d in datasets)
objs = [json.load(open(f)) for f in glob.glob("research/progress_*.json")]
progress = {}
for o in objs:
 for r in o["results"]:
  v=r["variant"]; key=(o["dataset"].split("/")[-1],v["model.depth"],v["budget.training_steps"],v["budget.batch_size"],round(v["optimizer.lr"]*v["budget.training_steps"],8))
  progress[key]=r
assert len(progress)==128, len(progress)
analysis=json.load(open("research/analysis_results.json"))
lines=[
"# Tabular Architecture Reversal: Conditional Phase Table",
"## Finding",
"Epochs alone do not govern the tested reversal. At matched Delta=.0768 and matched E16/E32, changing batch/exposure scheduling reverses d2-vs4 on both sparse8 datasets, but not either smooth16 dataset. Batch/schedule and data condition therefore matter; rule is confounded with input dimension, and function-space progress is not measured. This is a conditional phase table, not a fitted phase law or causal account of calibration.",
"## Frozen Sources and Definitions",
"Read R04f88f629b, Rc5401dd3c0, Rd2f1faba9e and K1009/K1033/K1056/K1073/K1083/K1089/K1094/K1095/K1113. Initial snapshots: 983 history records,73 claims,11673 lab candidates (including preexisting experiments). Checkpoint Jfac834ef8be1_research_e0005 retained 983 records/maxepoch5. Only allowed training history/lab datasets were used; no evaluation-answer files. Initial and checkpoint inputs remain separate.",
"Depth counts hidden width-to-width blocks, excluding projection/head. LN masks refer to hidden blocks; residual addition is a separate factor. Nominal E=T*b/1024 counts with-replacement presentations, not complete passes. AdamW Delta=lr*T is a nominal rate proxy, not measured parameter displacement, fitting progress or logit motion.",
"Dataset labels S1/S2 are eight-feature sparse-interaction stabcls_0db74c/stabcls_17011f; A1/A2 are sixteen-feature smooth-additive stabcls_2173ed/stabcls_28e027. All have1024 train/2048 test rows. Instance/point seeds respectively33092035/33093035,145092376/145093376,108092265/108093265,34433205/34434205. Exact active features, coefficients, thresholds and sampling metadata are in research/lab.json and candidate specs. These are independent allowed instances/splits per rule family, not independent splits of one identical target.",
"## Methods and Preregistered Predictions",
"Default initialization, plain CE, AdamW(.9,.999),wd1e-5 and training seeds0-9 are fixed. Reference factorial crosses widths64/128 and depths1/2/3/4 with plain leaky(.01),all-LN,T256/b64,Delta.0768. Depth changes still change LN count/parameter count. Progress grid fixes w64 and compares d2/d4 at Delta .03/.0768/.1/.3,T256/512/2048,b64, plusDelta.0768,T256/512/1024/2048,b16. lr=Delta/T. Controls fixd3w64,T256/b64,Delta.0768 and cross residual/plain,leaky/SiLU,LN000/100/010/001/111.",
"Predictions were saved/published before outcomes: no universal shallow advantage atDelta.0768; sparse depth3 may retain its recipe advantage; smooth d2-vs4 may favor shallow atE32; equalDelta need not collapse schedules; LN was expected to improve sparse overzeroLN but not all placements equally. Each condition is retained. Runner max12 caused rejected untrained requests; regrouping kept depth contrasts within one returned set. Cached anchors enable batch/exposure and activation comparisons within sets and are not independent replications.",
"## Reference Factorial",
"Each vector is mean test CE for depths1/2/3/4; ten seeds/cell. d2 is lowest in8/8 dataset-width cells; d4 loses tod2 in80/80 paired seed comparisons. This is a minimum on the tested grid, not an architecture optimum.",
"| Dataset | Width64 CE d1/d2/d3/d4 | Width128 CE d1/d2/d3/d4 |",
"|---|---|---|"]
for label,id,*_ in datasets:
 o=json.load(open("research/factorial_"+{"S1":"sparse1","S2":"sparse2","A1":"smooth1","A2":"smooth2"}[label]+".json"))
 vals=[]
 for w in [64,128]:
  vals.append("/".join(f'{next(r for r in o["results"] if r["variant"]["model.width"]==w and r["variant"]["model.depth"]==d)["mean"]:.6f}' for d in [1,2,3,4]))
 lines.append("| "+label+" | "+" | ".join(vals)+" |")
lines += [
"At w64 d3-d1 differences are -.139997/-.119244 onS1/S2; paired-seed bootstrap95% intervals[-.156497,-.125540]/[-.133720,-.105247]. At w128 d3 loses tod1 onA1/A2 by+.041532/+.055712, intervals[.033128,.050484]/[.042208,.070058]. S2w128 andA2w64 d3-d1 intervals include0. Thus K1033 d2-vs4 cannot decide d1-vs3. Accuracy does not duplicate CE: S2w128 d3 improves CE slightly over d1 but reduces accuracy .924072→.918018; A2w64 similarly .857178→.842627. All accuracy/SD/seed outcomes are archived.",
"## Conditional Phase Table",
"Each entry is d2/d4 mean test CE at w64,plain leaky,all-LN. Lower is better. No cross-set loss ratio is used; paired depth contrasts come from one experiment set.",
"| T | Batch | E | Delta | S1 CE d2/d4 | S2 CE d2/d4 | A1 CE d2/d4 | A2 CE d2/d4 |",
"|---|---|---|---|---|---|---|---|"]
schedules=[(T,64,D) for T in [256,512,2048] for D in [.03,.0768,.1,.3]]+[(T,16,.0768) for T in [256,512,1024,2048]]
for T,b,D in schedules:
 vals=[]
 for label,id,*_ in datasets:
  vals.append("/".join(f'{progress[(id,d,T,b,round(D,8))]["mean"]:.6f}' for d in [2,4]))
 lines.append(f"| {T} | {b} | {T*b/1024:g} | {D:g} | "+" | ".join(vals)+" |")
lines += [
"Equal-exposure comparisons atDelta.0768 are within research/batch_*.json: T256/b64 versusT1024/b16(E16), andT512/b64 versusT2048/b16(E32). Sparse ordering changes at both exposures; smooth ordering does not. Same-step T256/T512 batch contrasts change exposure jointly; matchedE contrasts change steps and rate jointly. Together they exclude E-only and nominalDelta-only accounts for this recipe, but cannot attribute the effect solely to gradient noise. EqualDelta curves/schedules retain different CE and accuracy.",
"At matched E16, sparse d4-d2 CE differences switch from +.073425/+.081709 at b64 to -.043312/-.029846 at b16; paired bootstrap95% intervals for b16 are[-.052166,-.032734]/[-.045216,-.015899]. This CE reversal does NOT reverse mean accuracy: d4-d2 accuracy at E16/b16 is -.000439/-.002832, and at E32/b16 -.001660/-.007031. Both smooth datasets retain d2 CE/accuracy advantages. At b64/Delta.03, d4 wins9/12 means; A2 supplies three small opposite differences. At b64/Delta.0768,.1,.3, d2 wins36/36 means. These are conditional grid counts, not independent dataset replications or a continuous threshold.",
"## Isolated Residual, LN and Activation Controls",
"Vectors follow LN000/100/010/001/111 at fixed d3/w64. Residual is crossed rather than scored as an LN. Each LN/residual comparison is within a returned set; activation comparisons use repeated matched anchors.",
"| Dataset | Plain leaky CE | Residual leaky CE | Plain SiLU CE | Residual SiLU CE |",
"|---|---|---|---|---|"]
for label,id,*_ in datasets:
 suffix={"S1":"sparse1","S2":"sparse2","A1":"smooth1","A2":"smooth2"}[label]
 rs=[]
 for j in [0,1]:rs+=json.load(open(f"research/controls_{suffix}_{j}.json"))["results"]
 vals=[]
 for R,act in [(False,"leaky_relu"),(True,"leaky_relu"),(False,"silu"),(True,"silu")]:
  vals.append("/".join(f'{next(r for r in rs if r["variant"]["model.residual"]==R and r["variant"]["model.activation"]==act and r["variant"]["model.layer_norm"]==mask)["mean"]:.6f}' for mask in [[False]*3,[True,False,False],[False,True,False],[False,False,True],[True]*3]))
 lines.append("| "+label+" | "+" | ".join(vals)+" |")
lines += [
"Preregistered all-LN improvement fails for plain sparse rectifiers: all-LN versus zero-LN S1 .218640/.210671,S2 .227849/.201826. Early-only LN improves to.176435/.181445; residual changes these effects. Plain SiLU LN placement differs acrossS1/S2: early/middle .142096/.163789 versus.140599/.123581. All-LN smooth-versus-rectifier contrasts are matched within sets; zeroLN/early activation contrasts are separately regrouped as cached anchors when present. Counts alone cannot supply an additive solver score. The controls include K1113 residuald3 measurements as reused cache, not independent confirmations.",
"Cached within-set activation anchors show a normalization interaction: with plain zero-LN, SiLU loses to leaky on both sparse instances (.335357>.210671;.350689>.201826), but wins with early-only LN (.142096<.176435;.140599<.181445) and all-LN (.146352<.218640;.140685<.227849). These contrasts change only activation. More LN and smooth activation therefore cannot be scored independently of placement. Residual leaky benefits from all-LN versus zero-LN on both sparse datasets, unlike plain leaky; this is a measured interaction, not residual=oneLN.",
"## Legacy Tests, Failures and Solver Default",
"K1033 reports shallow16/16 atE32 and4/8 atE4 for heterogeneous sixteen-input d2-vs4 recipes with Adam/Adagrad. It has no source-question list for those controlled counts; the exact16/16 original measurement table is not reconstructable from the frozen lab. These counts are preserved as legacy reports, not silently relabeled as reproduced. New AdamW fixed-width/all-LN controls are not exact replications. K1113's two sparse residual-leaky d3w64 versusd1w128 full recipes remain valid (.210256<.232318;.229833<.242104 atE16/Delta.0768), but confound width/LN. No law is proposed because all three required legacy benchmarks are not independently reproduced under one identified model.",
"Rc5401dd3c0 andK1094/K1095 retain their counterexamples to blanketE32 low-rate preference; K1083/K1089's local rate reversal retains its batch/model scope. q_df413c/q_700622 only establish bundled orderings; q_d30ff6's deep conservative Adam winner versus shallow aggressive AdamW loser changes architecture,LN,optimizer,rate anddecay. Their selected failures do not identify a population correction. Supplied underselection of depth/LN and overselection ofDelta was not independently reconstructed from counterfactual retrieval logs.",
"The demonstrated harmful default is inappropriate transfer: pooled shallow/smooth/LN preferences, depth2-vs4 results and nominalDelta were combined into one shallow-wide-smooth scoring law. Neither more depth/LN everywhere nor lessDelta everywhere repairs it. Check exact data/recipe/budget gates; retain counterexamples and uncertainty. The measurements identify conditions where the default fails, not a unique psychological or optimization cause.",
"## Uncertainty, Mechanisms and Unrun Interventions",
"Ten paired seeds measure initialization/minibatch sensitivity conditional on fixed splits. Bootstrap intervals use10000 paired resamples,rng913; descriptive,not multiplicity corrected or population intervals. Only two instances per rule family were tested; coefficients/activefeatures/point seeds differ. Width changes parameter count; all-LN depth contrasts change LN count. Rule/dimension effects are not causally separated. Excluded/failed conditions and cache status remain in raw objects.",
"Curves.npz contains only testCE curves(10,T),samples(T),batch_size; summaries include final test accuracy. Saved train.py evaluates test metrics only and receives split tensors externally. No trainCE,per-point margins,wrong-point confidence,logits or weights are supplied. Temperature capability probe was rejected: evaluation.temperature must start with model./optimizer./loss./budget. No dataset/distractor override is exposed. Therefore calibration/distractor-removal interventions were NOT run; no overconfidence,overfitting,undertraining or function-space mechanism is inferred fromDelta or testCE.",
"Competing explanations: architecture-dependent function fitting; LN placement changing representations; minibatch noise/optimizer trajectories; target/distractor structure; confidence/late generalization. Discriminating next tests: instrument exact allowed split tensors to record train/testCE,accuracy,correct-label logit margins,wrong-point maxconfidence and centered-logit displacement from initialization. Fit temperature only on a held-out portion of TRAIN data,then lock it for test; CE improvement without accuracy change isolates a confidence component. Remove only irrelevant input coordinates while preserving labels and split rows,then retrain paired recipes; reduction of deep test disadvantage tests distractor contribution. Test a16sparse/8smooth crossover to separate dimension from rule. Match trainingCE or centered-logit motion across recipes to test progress collapse; testCE matching alone is not a fitting metric. Preregistered but untested causal prediction: on sparse E16/b16, d2 has greater wrong-point confidence than d4, and train-validation-fitted temperature scaling improves d2 testCE more, narrowing the d4 CE advantage without changing accuracy. Failure would challenge a wrong-confidence account; correct-point margins and target-fit explanations remain alternatives.",
"## Provenance and Publication",
f"Reproducibility: analyze.py reads frozen snapshots/result objects, verifies every saved measurement SHA256 and computes within-set seed contrasts/accuracy/bootstrap intervals; research/analysis_results.json records {analysis['unique_conditions']} unique conditions,{analysis['reported_cells']} reported cells and {analysis['failed_seeds']} failed seeds. Raw factorial/progress/controls/batch JSON retain absolute measurement paths,repo paths,candidate specs,seeds,cache flags and hashes. Cached cells are not additional splits or replications. Allowed bases: S1 stabcls_0db74c/set_4096_var_var_fix_9f295a/c_2bbdc1; A1 stabcls_2173ed/set_4096_var_var_fix_b3eb59/c_0faeff; dataset overrides select only S2/A2. Scripts, preregistration,rejected-call notes,source snapshots and measurement copies accompany ongoing reportR760974e919.",
"Published early and updated. Proposals are individually≤700characters: a submitted scoped K1033 revision, K1009/K1056/K1113 annotation, and two conditional additions. Submission receipts are not evidence of canonical application: final load_kb still has the original K1033 text and unrelated canonical texts at the returned addition IDs K1114/K1115. The two additions are identified by their archived text, not cited under those IDs. All four operations remain pending curation. Independently reviewable,unfinished as a mechanism theory."
]
text="\n".join(lines)+"\n"
for old,new in [["reads frozen snapshots/result objects","reads saved result objects"],["bootstrap95%","bootstrap 95%"],["atDelta","at Delta="],["atE","at E"],["atT","at T"],["andK","and K"],["and4/8","and 4/8"],["shallow16/16","shallow 16/16"],["fromDelta","from Delta"],["nominalDelta","nominal Delta"],["equalDelta","equal Delta"],["EqualDelta","Equal Delta"],["matchedE","matched E"],["versusT","versus T"],["andT","and T"],["testCE","test CE"],["trainCE","train CE"],["trainingCE","training CE"],["use10000","use 10000"],["paired-seed bootstrap95","paired-seed bootstrap 95"],["reportR760","report R760"],["individually≤700characters","individually <=700 characters"],["1024rows","1024 rows"],["samples(T)","samples(T)"],["allLN","all-LN"],["zeroLN","zero-LN"],["more depth/LN everywhere","more depth or LN everywhere"],["lessDelta","less Delta"],["width/LN","width/LN"],["the original16/16","the original 16/16"],["the exact16/16","the exact 16/16"],["activefeatures","active features"],["only two","only two"]]:
 text=text.replace(old,new)
text=text.replace('Published early and updated.','Training reruns require the lab runner; analyze.py reproduces the numerical audit from saved measurements, not independent data generation. Published early and updated.')
Path('research/full_report.md').write_text(text)
print('Rebuilt full report from measured tables')
