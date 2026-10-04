# Low-Rate Univariate Optimizer Audit

## Findings
A recipe-local exception to categorical Adam preference reproduces at **lr=3e-5, T256** on two nearby one-cycle targets: RMSprop wins all ten paired seeds on each. At T512, Adam wins all ten seeds on both targets; at 1e-4, Adam also wins both mean comparisons. RMSprop1e-4 beats3e-5 on the four-cycle analogue at T512, but equal3e-5 RMSprop superiority does not transfer reliably. Neither a universal RMSprop preference nor a universal lower-RMSprop-rate rule follows.

## Sources and Focal Errors
Read frozen history/KB first:1043 records,73 claims,13156 lab rows; then existing reports, especially R1e820ab3a7/R7dd7d943d3/R79e71854c8. Persisted Adam controls exp:34aa6df219ff/9188999e597c were not retrained.
q_5604c0,ranking_v2/epoch7:gold E<B<D<A<C versus predicted E<A<B<D<C. All share residual d4w192 GELU,LN[F,T,F,T],default init,T256/b32. A=AdamW3e-5,wd1e-5,betas(.9,.95);B/D=RMSprop3e-5,wd0/1e-4;E=Adam.0003,wd1e-5,betas(.9,.95). Target sin(2*pi*x)-x,point5000111262274.
q_b97423,dataflip500/epoch7:goldB versus predictedC. All share residual d3w192 SiLU,all3LN,default init,T512/b64. A/B=RMSprop3e-5/1e-4,wd0;C=AdamW3e-5,wd.001,betas(.9,.95). Target -(cos(2*pi*x)+x)*sin(8*pi*x),point1372980798. Gold supplies ordinal outcomes,not focal MSE,seed margins or mechanisms.

## Historical Audit and Checkpoint
Exact executable-model/loss matching reproduces the comment’s RMSprop equal3e-5 wins3/4 questions excluding q560:both T256 questions win;T512 is1/2. Including q560’s two decay variants gives5/6 pairs across5questions,T2564/4 across3questions. Decay/betas vary;shared gold is not independent replication. RMSprop rate>=Adam-family rate gives19/24 across15questions. q_7f8fb0 remains the T512 low-rate counterexample.
Exact RMSprop rate-only history favors higher rates2/2 atT512:q_b97423 and q_9d6733(.001 versus3e-5).
Base-lab same-set/plainMSE/T256/equal-rate but unmatched-model comparisons reproduce RMS wins15/25 at3e-5,4/7 at1e-4,4/23 at.001,0/14 at.003;zero reported failures. Strict full-model/loss/budget matching yields zero base Adam-family/RMSprop pairs. Older K1057 counts78/88 and equal-rate21/24 therefore remain inherited observations,not newly verified exact controls.
Refresh pinned Jed4a66d29ba2_research_e0008_v000001_57bf3f95e2ba:1073 history records/91claims,30 newly visible IDs. Matched historical counts were unchanged. K1057/K1063 were already narrowed at checkpoint;this investigation extends that repair.

## Preregistration and Methods
Published R0d5af46860 before training;published the nearby-target/decay amendment before its outcomes. Neither focal point seed is allowed,and q_b97423’s exact expression is absent.
Allowed datasets all use uniform[0,1],train256/test256: sym_bca7ef,sin(2*pi*x)-x,point114197898; sym_b41081,x**2+sin(2*pi*x)-x+.5,point62093124; sym_a052d9,sin(8*pi*x)+4*x,point77197787. Bca/b410 preserve q560 architecture/b32;a052 preserves qb architecture/b64. These are different expressions/splits,not independent same-target replications.
Main grids cross RMSprop/Adam/AdamW,lr3e-5/1e-4,T256/512,wd0,plainMSE;Adam betas(.9,.95),CPU/default initialization,seeds0–9. Generated code verifies paired initialization/batch streams and replacement sampling. AdamWwd0 duplicates Adam numerically. RMSprop uses constructor defaults;inherited beta metadata is ignored by its executable code.
Preregistered bca3e-5/T256 RMS preference and a052T512 higher-RMS-rate preference succeed. Preregistered a052T512 equal3e-5 RMS preference **fails by mean**,with uncertainty spanning0. Nearby b410 predictions succeed:RMS at3e-5/T256;Adam at3e-5/T512 and1e-4/both budgets. Decay-restoration predictions succeed;beta effects were exploratory.
45 fresh configurations/450 seed attempts,zero failure flags/exclusions. Two cache-only joint assemblies add21 reused rows,not training or replication. Seven receipts retain66 rows,45 unique specs and32 distinct curve hashes.

## Main Results
Mean held-out MSE;A=Adam(.9,.95),R=RMSprop,wd0. AdamW is identical to A here.
|Dataset/T|R3e-5|A3e-5|R1e-4|A1e-4|R/A at3e-5;R seed wins|
|---|---:|---:|---:|---:|---|
|bca7ef/256|.00632375|.00941933|.00168989|.000293793|.671359;10/10|
|bca7ef/512|.00139025|.000245782|.000928923|.000138127|5.65643;0/10|
|b41081/256|.0156579|.0263981|.00935193|.000654989|.593143;10/10|
|b41081/512|.00192477|.000940101|.00280573|.000198962|2.04741;0/10|
|a052d9/256|.456679|.461376|.454197|.458262|.989818;7/10|
|a052d9/512|.452659|.452184|.443495|.448019|1.00105;3/10|
Sources:bca exp:1fe8aa290130;b410 exp:641629738d89;a052 exp:20a420493bae. Full means,SDs,recipes,endpoints and contrasts are archived.
Paired95% R-A intervals at3e-5:bcaT256[-.004567,-.001625],T512[.000193,.002095];b410T256[-.013712,-.007768],T512[.000570,.001399]. A052T512 equal3e-5 interval[-.003019,.003968];equal1e-4[-.013686,.004637].
RMS1e-4 minus3e-5:bcaT256=-.004634,high/low ratio.267228,9/10 high-rate wins,CI[-.007162,-.002106];bcaT512=-.000461,ratio.668170,6/10,CI crosses0. A052T512=-.009164,ratio.979756,9/10,CI[-.013998,-.004330].
B410T512 reverses RMS means:high/low1.45769,although higher rate wins6/10 seeds;CI[-.002205,.003967]. High-rate seeds7/8 have MSE.014901/.006448. They are poor finite outcomes,not flagged failures,and remain in the mean.

## Restored Recipes and Uncertainty
Within cache-only joint exp:be0d2a4577e6,bcaT256:Adam.0003/wd1e-5=.000551592;RMS3e-5/wd0=.00632375;RMS3e-5/wd1e-4=.00634149;AdamW3e-5/wd1e-5=.00941933. These preserve E<B<D<A;the complete focal order is untested because SGD C was omitted. RMSwd0 beats AdamW10/10 seeds. Stronger RMS decay slightly worsens MSE by.000017739.
Within joint exp:df2582c005d3,a052T512:AdamW3e-5/wd.001=.45218369;RMS1e-4=.44349481,ratio.980785,wins9/10,CI[-.013928,-.003450]. This is four-cycle analogue transfer,not focal replay.
Exploratory beta2.999 Adam:bcaT256 MSE.0176625/.000942449 at3e-5/1e-4,retaining the low/high family reversal. A052T512 Adam.456141/.458592 changes the equal3e-5 family mean sign relative to beta2.95. Beta strata cannot be discarded.
Intervals are unadjusted paired Student-t(df9),conditional on fixed splits;normality is approximate. No multiplicity correction,test-point resampling or population success probability is claimed. All18 seedwise T256/T512 curve-prefix comparisons are identical:budgets are continuations,not independent replications.

## Fitting Versus Overshoot
Observed held-out fitting is faster for RMS at3e-5:bca step16 R/A=.1291/.5697,step128=.01755/.05870;Adam overtakes byT512. A052 step16=.8203/2.6644,step128=.4659/.4804,then near-ties atT512. This does not prove oscillatory-component acquisition or accumulator causality;step0 was unmeasured.
Bca RMS1e-4/T512 last64 mean=.005139 versus endpoint.000929;average within-seed temporalSD=.01028 versus Adam.000142. Higher-rate RMS fluctuates more yet beats lower-rate RMS by endpoint mean. A052 likewise combines larger fluctuations with improved endpoints. Fluctuation alone therefore cannot establish harmful overshoot.
Competing explanations include optimizer transient/state,late noise,initialization/LN interactions,target offset/envelope,decay/betas and generalization. Underfitting predicts worse train and test residuals;reversible overshoot predicts reduced late update variance and improved both residuals after a paired downshift;generalization predicts better train but worse test fit. Train loss,updates,outputs and checkpoint/state forks are unavailable,so these tests remain unresolved.

## Predictions, Boundaries and Publication
Unmeasured prediction:allowed sym_3bb817,tanh(2*sin(2*pi*x))-x,same d4w192 residualGELU,LN[F,T,F,T],b32,wd0,beta2.95/default init/seeds0–9:predict RMS/Adam meanMSE<1 at3e-5/T256,and>1 at3e-5/T512 and1e-4/both budgets. Opposite signs falsify each transfer prediction. This target was not measured.
No exact focal replay,same-target split replication,continuous optimum,width/depth/LN/batch threshold or universal optimizer default is established. Target and point split change together;causal frequency dependence remains unresolved.
optimizer_audit/ preserves frozen/checkpoint inputs,source reports,preregistrations,receipts,45 executable recipes,source paths/SHA256,curves and analyses. Both audit.py and analyze.py ran successfully;hashes,endpoints,budget prefixes and archives were validated.
Submitted revisions K1057/K1063 and new K1126;claim texts are625/597/541 characters. Published full report **R0d5af46860**,commit **7053d60178ae82767b7c0d9858408b61f4167cf1**. The endpoint evidence is independently reviewable;mechanism remains unfinished.