import json,gzip
from pathlib import Path
R=Path(__file__).resolve().parent
s=json.loads((R/'statistics.json').read_text())
rows=s['rows']
def f(v):return f"{v:.6f}"
def ci(c):return '['+','.join(f(x) for x in c['ci'])+']'
text="""# q_ae3f60: Wide XOR16 Optimizer Reversal and Recipe Bounds
## Finding and Status
The original D<E ordering transfers to both primary allowed XOR16 splits and a preregistered third split: at w256/T512/b64, Adam1e-4 beats AdamW1e-3 in all30 paired seeds across3 fixed datasets. Matched Adam and AdamW rate twins each favor1e-4 there. Narrowing only width to24 reverses the preference on all3 datasets; reducing only batch to16 does NOT. This is a conditional recipe result, not a universal low-rate/XOR rule or a measured overshoot mechanism.
Report-only, pending independent review. No KB mutations were submitted. Three experiment calls used the three remaining slots, not extra jobs.
## Sources and Effective Recipe
Read frozen1193 history records/113 claims/14028 lab rows and exact q_ae3f60 training record; archives preserve measurement sources. Gold D<E is ordinal10-seed-mean evidence only: original CE margins, individual losses and train-evaluation results are unavailable. Original Gaussian XOR16 active8/10,point8001196265529/calibration8001196266529,train1024/test2048.
Plain default MLP: Linear16->256,SiLU,three SiLU(Linear(LayerNorm(x))) blocks,Linear256->2; depth3 means three hidden blocks and FIVE Linear layers. D=Adam1e-4,E=AdamW1e-3; both betas(.9,.999),wd1e-4,plainCE,T512,b64,CPU. Adam uses coupled decay; AdamW decoupled. All parameters,including biases/LN,enter the optimizer.
Inspected generated source matches this construction,torch.manual_seed per run,torch.randint replacement minibatches,and full-test CE/accuracy after every update. Linear initialization is untouched; LN affine/eps1e-5 and optimizer eps1e-8/AMSGradfalse are constructor defaults,not explicit overrides or measurements. Inherited momentum0 and leaky_slope.01 in some specs are inactive for Adam/SiLU; no loss penalty remains.
All36 receipts have source_provenance=null: current generated model/optimizer/loss/train files were inspected,archived and hashed, but exact EXECUTED bytes/environment are not pinned by the runner. Do not equate retrospective source hashes with execution attestation.
## Prior Evidence Audit
Within frozen sets,exact model/loss/budget with unequal Adam-family rates reproduces14/14 higher-rate wins on3datasets. Twelve are on xorcls_0dd7d2: exp:3bde4fc36ac5 has two LN backgrounds x three rate pairs at w24/GELU/T256/b16/wd1e-5/beta2.95; exp:d82786f1e971 has three rate pairs at each T256/T2048,w24/GELU/LN101/b16/wd0/beta2.95. Those twelve also match all nonrate optimizer fields.
The other two base pairs,xorcls_b6e156/xorcls_ed472f,w192/T256/b16,change decay.0001->.001 alongside rate.0001->.0003. Thus14 comparisons are NOT14 independent datasets,not all pure rate controls,and none tests q_ae3f60's wide recipe. They are retained rather than pooled away. K1119's narrow long AdamW/SGD reversal is a distinct optimizer comparison,not a low-rate rule.
## Frozen Design and Prediction Tests
Published preregistration Rb130ee313e at d29b644b0ee057bd0567fd9d31ad81395b0bf895 before experiments. Metadata-only selection: first three lexicographic base-lab Gaussian XOR16/train1024/test2048 IDs. No existing wide256 XOR16 experiment cells were present.
Primary0dd7d2:active2/7,point173198078;primary14bbf0:active3/10,point242198285;prospective19c4ba:active5/9,point206198177. Different active coordinates and point splits travel together; these are not the original hidden split.
Each dataset has12 cells: Adam/AdamW x lr.0001/.001 at wd.0001/betas(.9,.999),in w256/b64,w256/b16,w24/b64 backgrounds,T512/all3LN/SiLU/plain fixed. Each boundary changes one recipe setting. Batch16 also reduces exposure32768->8192; its effect is not a noise-only intervention. Width changes parameter count and initialization stream.
Before outcomes predicted D<E and low rate wins within both families at w256/b64;predicted batch16 AND width24 reverse the preference;predicted same-rate family effects smaller than rate effects. Wide and width predictions pass3/3 datasets;batch prediction fails3/3. Third-split predictions were unchanged after first-two results.
## Absolute Full-Recipe Results
CE nats;D=Adam.0001,E=AdamW.001. Intervals paired95% t9;wins count D lowerCE. Every row n=10 finite paired seeds;SDs/fullprecision and secondary accuracy are in statistics.json.
|Split|w/b|D CE|E CE|E-D|95% interval|D wins|
|---|---|---:|---:|---:|---|---:|
"""
for r in rows:
 text+=f"|{r['dataset'].split('_')[-1]}|{r['width']}/{r['batch']}|{f(r['means'][0])}|{f(r['means'][3])}|{f(r['full']['effect'])}|{ci(r['full'])}|{r['full']['a_wins']}/10|\n"
text+="""## Family-Rate Separation
High-minus-low CE at fixed optimizer family;positive favors low rate. Brackets are paired95% intervals;all contrasts n10.
|Split|w/b|Adam effect [interval]|AdamW effect [interval]|
|---|---|---|---|
"""
for r in rows:
 text+=f"|{r['dataset'].split('_')[-1]}|{r['width']}/{r['batch']}|{f(r['adam_rate']['effect'])} {ci(r['adam_rate'])}|{f(r['adamw_rate']['effect'])} {ci(r['adamw_rate'])}|\n"
text+="""Both families favor low rate in6/6 wide mean comparisons (b64 and b16 separately),and high rate in3/3 narrow comparisons;all pointwise rate intervals exclude0. Within-family low-rate wins10/10 except AdamW,b16,0dd7d2:9/10,losing seed7. Full D/E at that condition also loses seed7;all other wide full comparisons win10/10. Narrow D loses every seed.
At w256/b64,AdamW-minus-Adam at low rate is+.000686/+.000731/+.000764,each10/10 Adam wins;high-rate family effects are-.040143/+.017079/+.025508 with Adam wins7/10,8/10,8/10 and all intervals including0. Family mean effects are smaller than matched rate effects in all9 backgrounds,but family ranking is not universal. All family intervals/counterexamples are archived.
Secondary D/E accuracy at w256/b64 is .92593/.91382,.92026/.90718,.90776/.89814 (dataset order above);do not replace CE with accuracy. Narrow high rate improves accuracy markedly as well.
## Uncertainty, Explanation, and Boundaries
36 dataset/config cells,360 scheduled and verified finite seed outcomes,0 failures,0 exclusions,0 cache reuse. Paired intervals describe training-seed variation conditional on each fixed split,assuming approximately normal independent seed differences;no multiplicity correction,test-row resampling or population-dataset interval. Three datasets,not30 independent datasets;reused cells make comparison counts dependent.
The rate-by-width endpoint interaction is controlled evidence for these discrete recipes. It does not identify overfitting,confidence inflation,oscillation,feature learning or optimizer-state mechanisms. Curves evaluate held-out CE after update1 onwards,not step0/training CE;finite runs do not prove stable trajectories. Coupled versus decoupled decay remains a competing conditional influence,not an explanation of the original margin.
Discriminating future tests:freeze a fourth metadata-selected split,measure train/test CE and logit calibration together;record gradients/update norms and controlled initializations to separate fitting/generalization from instability. A decay0 rate-family factorial would test dependence on coupled decay. None was run;original train evaluation was not invented.
Unmeasured prediction:first remaining metadata-selected XOR16 split xorcls_19ee72 should retain D<E at w256/b64 AND b16,but reverse at w24/b64;matched rates should have the same width-specific signs. Opposite paired mean signs reject corresponding transfer predictions. No continuous width threshold,other activation/LN/depth,T change,or universal batch boundary is established.
## Reproducibility and Checkpoint
Run python xor_audit/analyze.py then python xor_audit/make_report.py. Inputs,selection,preregistration,old within-set pairs,three raw receipts,36 specs/seed summaries/curves and144 retrospective source files are preserved. Analyzer verifies108 measurement SHA256 hashes,seed counts and receipt means;statistics.json retains CE SDs,all paired intervals,wins,seed counterexamples and accuracy.
Experiment receipts: b8ad6960c85b,f7f40b9841d4,86fc3bde2a83. Refresh pinned J65c518c3342c_research_e0012_v000001_8a48781d17d4,1193 history/118 claims. Newly visible K1172 preserves the original ordinal result and refers to this investigation;it supplies no independent transfer measurements. Other new claims are unrelated. No duplicate measurements or proposals were added.
The empirical recipe comparison is independently inspectable subject to missing execution-byte attestation. Mechanism remains unresolved;report awaits independent review.
"""
(R/'report.md').write_text(text)
print(text)
