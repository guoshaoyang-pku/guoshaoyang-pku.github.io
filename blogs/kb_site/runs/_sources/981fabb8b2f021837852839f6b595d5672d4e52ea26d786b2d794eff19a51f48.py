import json,os,re
HERE=os.path.dirname(os.path.abspath(__file__))
a=json.load(open(os.path.join(HERE,'analysis.json')))
t=a['new']['table']
def row(contrast,family,depth,residual,optimizer,kind):
    return next(x for x in t if (x['contrast'],x['family'],x['depth'],x['residual'],x['optimizer'],x['kind'])==(contrast,family,depth,residual,optimizer,kind))
def pair(x):return ', '.join(f'{v:.5f}' for v in x['ratios'])
families=[('XOR','xor_classification'),('Tabular','synthetic_tabular_classification'),('Spiral','spiral_classification'),('MSE','multivariate_regression')]
d4=[]
for label,fam in families:
    for opt in ['SGD0','SGD9','Adam']:
        vals=[pair(row(c,fam,4,r,opt,k)) for c,k in [('terminal/early','terminal'),('all/terminal','all')] for r in [False,True]]
        d4.append('| '+label+' | '+opt+' | '+' | '.join(vals)+' |')
d5=[]
for label,fam in families:
    vals=[pair(row('depth5/2',fam,5,r,'SGD0',k)) for k in ['zero','all'] for r in [False,True]]
    d5.append('| '+label+' | '+' | '.join(vals)+' |')
try:publication=json.load(open(os.path.join(HERE,'publication.json')))
except FileNotFoundError:publication={'actions':'One scoped spiral claim and a K1059 annotation are proposed; surviving parent repairs are unchanged.'}
report=f'''# Initial Scale versus Normalization Geometry

**Answer**
Not yet established: direct initial-scale measurements and the nonnormalizing scalar rescue were not executable. Matched endpoint interventions reveal optimizer/family/residual interactions and LN-count failures, but do not identify activation shrinkage versus normalization geometry. No automatic frozen label is justified.

**Sources and Definitions**
Read R04f88f629b, R838be7703a and current KB; their surviving controls were not retrained. Initial archives:983 history,73 KB,11611 lab rows; first lab read was11601 before ten further rows entered the successful archive. Checkpoints: J09c7bcf6a305_research_e0004 archived953/68/12221; final e0005 archived983/72/12891. Neither refresh exposed new question IDs. No external newly added row matched the focal recipe/datasets. Deferred J290340c95a93:000005 is not visible through direct lookup; substantive evidence there cannot be independently audited. Supplied negative utilities are motivation, not reconstructed exposure statistics.
The generated plain block is SiLU(Linear(LN(x))); residual is SiLU(x+Linear(LN(x))). Depth excludes a separate projection/SiLU. TerminalLN normalizes the INPUT of the final hidden block, not head features directly. Early/terminal each have one LN; all/zero normalize every/no block input. Skips are followed by SiLU, not an unchanged identity path.
Calibrated Delta: Adam/AdamW/RMSprop lr*T; Adagrad2*lr*sqrt(T); SGD lr*T*g/(1-m),g=.03 for momentum MSE,.01 otherwise. This fitted proxy is not measured motion or a within-SGD momentum bonus.

**Old versus New**
Legacy K1059 reports terminal SGD31/31 in14sets,medianCE ratio.69; Adam70/102,.92. These original filters remain underidentified, not reproduced by the documented audit, and not directly refuted by different denominators. Nonexcluded BASE MLP/plainCE/nonresidual d>=2/within-set gives same-type SGD36/52(.981824),Adam27/70(1.051067);equal lr gives5/5(.965316),6/12(1.024390);full optimizer equality gives0 SGD pairs,Adam2/3(.763836). Width/depth/activation remain unmatched. Including archived prior experiment rows did not recover the legacy counts.
K1038 DOES reproduce: nonexcluded base/plainCE-or-MSE,SGD d>=3 SiLU/GELU zeroLN,SGD Delta>=3*adaptive,rivalDelta<.1:11/30 in11sets,medianSGD/rival1.055903. Plain3/15 in5sets,1.896676;residual8/15 in6sets,.991373. Multivariate3/11,univariate2/10,spiral4/5,XOR2/4. Normalized SGD>=1LN413/471 in171sets,.577774 uses no deep/smooth restriction. These comparisons do not match architectures or terminal placement.
One [regime table]({HERE}/unified_regime_table.csv) retains392 scoped rows,including legacy rows flagged unverified,exact reconstructed strata and split-specific new results. It cannot pool incompatible definitions or turn the first two legacy counts into a validated scale model.

**Matched Controls**
Fix width24,SiLU,default initialization,wd0,T256,bs16. SGD0 lr.003/m0;SGD9 lr.0003/m.9;Adam lr3e-5,betas(.9,.999). Saved optimizer code supplies remaining defaults,including SGD dampening0/nesterovfalse and Adam eps1e-8/amsgradfalse. CE Delta=.00768 each;MSE momentum=.02304,others=.00768:training exposure is matched,actual updates are not.
Two metadata-identical allowed point splits per family:XOR063fc1/4779bd(input4);sparse-interaction tabular168586/ffe3ff(input2);1.5-turn spiral00f3c5/01a657;MSE0e8d19/d2b5db,identical target x0**3+x0*x1+sin(2*pi*x1). CE train1024/test2048;MSE256/256. No original-question split,smooth-additive tabular or offset-MSE transfer is assumed.
The completed matrix is d2/d5:SGD0 x plain/residual x zero/early/terminal/all;d4:all three optimizers x both residual statuses x all four masks. Adaptive d2/d5 placement remains untested. Forty configurations per split,320 unique configurations,3200 new seed evaluations,zero failed seeds. Forty-one receipts contain482 rows,162 cached;eight cached-only interaction assemblies add no replication. All numerical contrasts are within a set.

**D4 Regimes**
Each cell is the mean-loss ratio on split1,split2;lower than1 favors the numerator. P/R denote plain/residual. Exact means,SDs,margins,seed counts and paired intervals are in [paired results]({HERE}/contrasts.json).
| Family | Optimizer | Terminal/Early P | Terminal/Early R | All/Terminal P | All/Terminal R |
|---|---|---|---|---|---|
{chr(10).join(d4)}

**Depth and Failures**
D5/D2 ratios below use within-set SGD0 zero/all anchors. No pooling across families or residual status.
| Family | Zero P | Zero R | All P | All R |
|---|---|---|---|---|
{chr(10).join(d5)}
Preregistered plain SGD0 CE terminal<early succeeds18/18 cells over d2/d4/d5,six point splits,not18 independent datasets. D4 SGD9 succeeds5/6,Adam6/6. SGD9 tabular ffe3ff fails:early.69552698,terminal.69601060,margin+.000483626,paired95% interval[-.00272260,.00368985],terminal wins2/10 seeds. Terminal superiority is not guaranteed.
The stronger-SGD/weaker-Adam placement prediction fails descriptively on both XOR and both tabular splits,holds on both spiral splits. Cached within-set paired log-ratio interactions support the tabular direction with unadjusted intervals below0;both XOR intervals include0. The added pre-d5 prediction that plain allLN d5 would lose to d2 fails6/6 classification comparisons;allLN d5 actually wins. Plain d4/d5 all<terminal CE prediction succeeds12/12. D5 terminal<early plainCE succeeds6/6. These are dependent recipe comparisons,not population success rates.
Residual effects remain separate:spiral d2 SGD0 early beats terminal2/2;d4 SGD0/SGD9 terminal beats early2/2 each,Adam reverses2/2. MSE residual SGD0 early beats terminal atd2/d4(4/4),but d4 SGD9/Adam terminal wins2/2 each. D5 residual zeroLN beats plain zeroLN8/8,while its depth5/depth2 zeroLN order improves only spiral,not XOR/tabular/MSE. Different definitions of a zeroLN deficit must not be conflated.
Two matched count failures:residual spiral d5 SGD0 early/zero1.009630/1.013809,margins+.00596274/+.00854062,early wins2/10 seeds each;paired95% intervals[-.00262149,.01454696]/[.00084386,.01623738]. Residual spiral d4 Adam all/terminal1.040143/1.029088,margins+.02948329/+.02139677,all wins2/10 and3/10;intervals[.00124826,.05771833]/[-.00912454,.05191808]. Both failure types replicate MEAN signs,not significance on both splits. D4 plain all/terminal improves24/24 across all four families/optimizers;residual improves22/24,with the two Adam-spiral failures retained.

**Uncertainty and Motion**
Seed SD is dispersion;paired95% t9 intervals assume approximately normal independent seed differences and have no multiplicity correction. Point instances share a rule/target,are not random population samples,and tabular here is only sparse interaction. Cached results and shared seeds/budgets are dependent. Small mean wins should not force a solver rank.
Curves start AFTER update1,not step0. D5 plain zeroLN held-out losses change:spiral first split.69728715->.69482442;MSE.84474086->.52148171 and.85284666->.54239756. This rejects completely unchanged predictions after update1,not frozen hidden features,small parameter motion or input-sensitive learning. Head/bias fitting could account for progress. Actual update norms and training loss remain unavailable.

**Mechanism Tests**
Head raw RMS sqrt(mean(h**2)),across-example centered RMS,input Jacobian RMS,per-layer weight/bias gradient RMS,actual update/displacement RMS,step0 CE/MSE and learned LN gamma/beta were NOT measured. Sandbox torch import fails at blocked ctypes.dlopen;the endpoint API exposes no diagnostics or documented tail-scalar control,and allowed pool exports no point arrays or model states. Reconstructing different NumPy seeds would not supply these official measurements. Scalar-tail rescue was not run;the main scale-mediation question remains unresolved.
Scale sufficiency predicts a fixed training-calibrated scalar s=RMS_terminal/RMS_early on the unnormalized head input substantially rescues SGD gradients/updates/loss. LN geometry predicts remaining differences after RMS matching:LN removes the feature-mean direction and approximately the radial direction;scalar multiplication preserves direction and sensitivity/RMS ratios. Bias/logit calibration,learned affine capacity,SiLU-after-skip composition,optimizer state and generalization compete. Adaptive scale cancellation requires restrictive constant-gradient-scaling/epsilon assumptions,not arbitrary CE training.
The required replay must export identical points/weights/minibatches,record these diagnostics at0/1/16/256,and freeze scalar calibration on training data only. Include both scalar-only and inverse-head-weight compensation controls:the latter keeps initial logits/loss fixed but changes SGD effective head learning rate by s**2,so measure head versus hidden updates separately. RMS restoration without optimization rescue rejects scalar sufficiency,not all scale contributions;successful rescue alone does not exclude geometry or logit calibration.
Future unmeasured prediction on metadata-selected allowed spiralcls_01b73f:same residual d4 Adam allLN>terminal,SGD0/SGD9 reverse that sign;residual d5 SGD0 earlyLN>zeroLN andterminal<zeroLN. An opposite replicated mean sign rejects the corresponding transfer prediction. No other width,activation,initialization,budget,turn-count or loss-family boundary is established.

**Solver and Publication**
Condition first:loss family,full optimizer recipe,depth/residual status,then tail placement. Plain SGD0 CE at this recipe supports terminal over early,not arbitrary cross-depth/activation ordering. Momentum tabular has a tiny counterexample;residual spiral Adam and residual MSE SGD0 require their own placement/count strata. Do not sum LN+residual units,infer a momentum bonus fromK1019,or label deep zeroLN frozen. K1006/K1038/K1022/K1061/K1101/K1108/K1109 surviving scopes remain intact;initial-scale repair already present at checkpoint was not repeated.
Run python scale_audit/analyze.py,then python scale_audit/make_report.py. Frozen/checkpoint inputs,source_reports.json.gz,old_pairs.json,41 raw receipts,320 local executable recipe/curve archives,interactions.json,curve_audit.json and unified table retain sources. Analyzer checks ten seeds,failures,curve hashes and every final seed loss. Training reruns require the allowed runner;offline analysis uses archived local files.
[Reporting corrections]({HERE}/reporting_corrections.md) record the16MiB snapshot failure,corrected missing-gzip import,publication text-parsing failure,lab archive count and interim ratio/source transcription corrections. No measurements were changed. Predictions were saved before their experiments;the checkpoint amendment's first publish failed,then unchanged text was published after the first allLN run. Initial full reports were published before training.
{publication['actions']} Published continuing report Reb8454abdf. The empirical evidence is independently reviewable;the requested scale/geometry mechanism is not completed.
'''
report=re.sub(r',(?=\S)', ', ', report)
report=re.sub(r';(?=\S)', '; ', report)
report=re.sub(r':(?=[A-Za-z0-9])', ': ', report)
report=re.sub(r'\b(in|at|over|across|with|after|before|below|above|paired|wins|succeeds|fails|improves|contains|retains|archived|input|Adam|Adagrad|Multivariate|univariate|spiral|XOR)(?=\d)', r'\1 ', report)
report=re.sub(r'(?<=\d)(?=sets\b|seeds\b|rows\b|cells\b|splits\b|pairs\b)', ' ', report)
for old,new in [('medianCE','median CE'),('plainCE','plain CE'),('atd2','at d2'),('fromK1019','from K1019'),('SGD>=1LN','SGD >=1LN'),('d>=3','d >=3'),('d>=2','d >=2')]:report=report.replace(old,new)
assert len(report.splitlines())<=68,len(report.splitlines())
with open(os.path.join(HERE,'final_report.md'),'w') as f:f.write(report)
print('report lines',len(report.splitlines()),'characters',len(report))
