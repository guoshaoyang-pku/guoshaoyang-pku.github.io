import json,math,numpy as np
r=json.load(open('slow_sgd/results.json'))
lines=[
'# Slow Plain SGD: Final Loss, Progress and Transfer',
'## Finding',
'The proposed GRU64x1/x2-over-TF32/64 rule for every 0<lr*T<=.8 is not validated: fresh equal-width cells produce two mean-sign counterexamples, both statistically unresolved. Duration transfers the signs; context/dataset transfer does not preserve all of them. Lower final CE must not be confused with greater progress.',
'## Sources and Reproduction',
'Frozen inputs: 1,283 training-history records,139 claims,14,280 allowed lab rows. No evaluation answers were accessed. Frozen training ordinals were used only for the historical audit; focal q_43b634 answer was not used. R58730d84ac is preserved in slow_sgd/prior_report.json.gz. Checkpoint J9456db3995be_research_e0015_v000001_7f3a7ec0bc75 added no question IDs; current K1124 already contained the interim counterexample.',
'The 68/68 backtest reproduces exactly:12sets/8datasets,TF-minus-GRU CE .043636751-.165139318,numerically including16 GRU2 contrasts. Raw73 becomes68 after excluding one unverified head_scale=.1 arm and collapsing four duplicate specifications. Matching precedes deduplication within sets; sets are not independent table replications. CE lambda fields are ignored by verified plain-CE source, not different loss interventions.',
'Exact lab shape scope:heads2;FF128 has GRU1/TF32x4 n30 plus all eight GRU64x1/x2-versus-TF32/64x1/x4 shape pairs n4 each (the GRU1/TF32x4 subgroup included);FF64 has GRU1/TF32x4 n10. No arbitrary FF/head/depth coverage. Frozen aggregate failure/seed fields are absent, so 68/68 cannot establish zero failures or paired uncertainty.',
'Full old36 SGD cells reproduce median GRU-TF=-.0984372735,ratio=.9699963318;old36 Adam cells reproduce -.0391818047,.9873249559. Old SGD grid includes exposure1.536 outside the proposed <=.8 band. Prior report supplies n10 and numerical-control fail0, conservative pointwise intervals and initial/progress evidence; repeated configurations are dependent.',
'History12/12 favorable ordinals reproduce across q_b9142b/q_162e81/q_a3b0c0, but only4/12 meet b16/wd0: q_b9142b,T1024,lr3e-5,heads4/FF64-256. q_162e81 is b32/wd1e-5,T512; q_a3b0c0 b32/wd.001,T256. Ordinals provide no CE margins,CIs,CE0 or failure counts. These are three question blocks, not12 independent replications.',
'## Design and Definitions',
'Preregistration Rf9c4bed639@e13f1a06595364b143c3187f4972f3624f69cab8 preceded training: all four GRU-TF final margins positive,TF progress greater,duration signs retained; Adam TF4 beats G1. Metadata selection: first lexicographic allowed V24/alpha1/train800/test200 dataset per context: bg_04502b L12,sequence/table100100120924/100100130924;bg_0bf6d7 L24,100136566616/100136576616. Existing tables,new recipes; no outcome-based selection.',
'G1/G2=nonresidual GRU64x1/x2;TF1/TF4=causal post-LN GELU TF64x1/x4,heads2/FF128,learned positions,dropout0. Default family initialization,biased linear heads,plain token-mean CE. Width is matched,parameters and family initialization are not. Executed model/optimizer/loss/train source is SHA256-verified for all60 valid configurations; no calibration intervention was executed.',
'SGD momentum0/wd0,b16,T512/2048,Delta=lr*T .2/.6/.8. Rates T512:.000390625/.001171875/.0015625;T2048 one quarter. T512/2048 see8192/32768 windows,not equal duration or token exposure. Delta is raw nominal exposure,not update norm or optimizer-equivalent motion.',
'M=CE_TF-CE_GRU (>0 favors GRU);P=CE0-CE_final;M_final=M0-(P_TF-P_GRU). No-update SGD lr0/wd0 is initialization-equivalent. Seed labels0-9 block repeatability,not identical weights/minibatches. Pointwise95%CIs use paired seed differences,t9=2.26216; conservative marginal-SD sensitivity is saved. Neither interval covers table/test-sample uncertainty or multiplicity.',
'## No-Update Baselines',
'|Dataset/context|CE0 G1|G2|TF1|TF4|',
'|---|---:|---:|---:|---:|']
for d,L in [('bg_04502b',12),('bg_0bf6d7',24)]:
 b=r['baselines'][d];lines.append('|%s/L%d|%s|'%(d,L,'|'.join('%.6f'%b[s] for s in ['G1','G2','TF1','TF4'])))
lines+=['## Final CE and Actual Test Progress','Each quadruple is G1,G2,TF1,TF4;all cells n10,seeds0-9,fail0. Final train CE and train progress are **unavailable**: verified runner records only full-test CE after updates,not train endpoints. They were not replaced by minibatch loss or fabricated estimates.',
'|Context/T/Delta|Final test CE|Actual test progress|','|---|---|---|']
for c in r['cells']:
 L=12 if c['dataset']=='bg_04502b' else 24
 lines.append('|L%d/%d/%.1f|%s|%s|'%(L,c['T'],c['delta'],','.join('%.6f'%c['final_test_ce'][s] for s in ['G1','G2','TF1','TF4']),','.join('%.6f'%c['test_progress'][s] for s in ['G1','G2','TF1','TF4'])))
lines+=['## Numerical Margins','Entries M[pointwise95%CI],nats;each contrast n10 seed pairs,zero valid-run failures. Source:slow_sgd/grid_*.json plus correct24_*.json;baseline sources zero12/zero24/zero24_tf.json. Full seed outcomes and all source/measurement hashes are retained,not aggregate-only reconstructions.',
'|Context/T/Delta|G1/TF1|G1/TF4|G2/TF1|G2/TF4|','|---|---|---|---|---|']
for c in r['cells']:
 L=12 if c['dataset']=='bg_04502b' else 24
 cs=[a for a in r['contrasts'] if (a['dataset'],a['T'],a['delta'])==(c['dataset'],c['T'],c['delta'])]
 lines.append('|L%d/%d/%.1f|%s|'%(L,c['T'],c['delta'],'|'.join('%.6f[%.6f,%.6f]'%(a['margin'],*a['ci95']) for a in cs)))
lines+=['## Tests, Failures and Boundaries',
'Final signs:46/48 favor GRU;38/48 CIs wholly favor GRU,none wholly favor TF. At L12/Delta.8,G2/TF4 M=-.00502565(T512),-.00507865(T2048),both CIs cross0. The preregistered all-positive point prediction fails;this does not establish a decisive TF phase. At .2/.6 all32 means favor GRU,but some .6 intervals cross0;no continuous .6 cutoff is inferred.',
'All24 duration-paired contrast signs persist;maximum mean margin shift .000195694 nats. Small observed differences are not a formal equivalence test. Retention M_final/M0 ranges[-.03357,.76131]L12 versus[.19600,.85260]L24. Tables and sequence seeds cochange with context,so this is transport heterogeneity,not a causal context effect.',
'TF extra actual test progress is positive in48/48 means AND paired95%CIs. Thus the preregistered progress and duration-sign predictions succeed even where final-rank predictions fail. Initial-gap decomposition is an accounting identity,not evidence that initialization alone caused final rankings.',
'Matched Adam(.9,.999),wd0,Delta.0256,T512 on bg_04502b is a boundary only:CE G1/G2/TF1/TF4=3.032721/3.086577/3.007964/2.940509. M G1/TF4=-.092212,CI[-.110188,-.074235];G1/TF1=-.024757,CI[-.044638,-.004876];G2/TF4=-.146068,CI[-.162228,-.129908]. All four GRU-TF means/CIs favor TF;the preregistered TF4-over-G1 prediction succeeds. Momentum was not tested.',
'There are60 valid configurations,600 finite seed evaluations,zero reported valid-run seed failures. Fourteen invalid L24 TF requests inherited L12 positional tables and raised IndexError;no finite outcomes or seed-failure counts were returned. TF-only corrections explicitly set context24,without repeating valid GRU training. One name-key rejection supplied no measurements.',
'One L24 G1 no-update baseline was inadvertently recomputed despite a surviving identical old mean because the unused GRU context attribute differed;it receives no independent replication credit. This deviates from the no-repeat instruction. Old numerical grids were recovered from frozen artifacts,not rerun. All remaining new valid cells are uncached;cached repeats are never independent evidence.',
'## Interpretation and Unresolved Tests',
'The phase is tested conditional final-gap retention under these exact recipes,not GRU-first or a universal lr*T law. No widened <=.8 law is promoted. The old68/68 supports old cells,not unseen heads,FFs,decay or contexts. K1075 adaptive counterexamples remain separate;consolidation is in this report,not deletion of incompatible optimizer scopes.',
'q_43b634 is L12,T2048,b16,SGD.0003/wd.001,Delta.6144,GRU64x2 versus TF64 depths1-4,heads2/4,FF64/256. This wd0/heads2/FF128 audit is not its exact recipe or a complete-order replay. Its final-loss-versus-progress error is corrected logically;no focal numerical rank or mechanism is inferred from these different recipes.',
'Competing explanations:initial predictive dispersion/bias,gradient conditioning,depth-dependent optimization,positions,gating,capacity and table/split variation. Neither initial CE nor ignored head_scale/position fields establishes causal calibration. Verified temperature/head interventions,train endpoints,actual updates and same-table context controls are unavailable;they remain unresolved.',
'Prospective unmeasured test:next eligible L12/alpha1 table bg_0915b4,identical heads2/FF128,SGD wd0/b16/Delta.2/T2048:predict all four M>0 and TF extra test progress>0. At .8 no universal sign is predicted. An opposite mean refutes the narrow sign prediction;an opposite CI is stronger falsification. This prediction is not a newly validated law.',
'## Reproducibility',
'Run python slow_sgd/analyze.py and python slow_sgd/audit_history.py. Frozen/checkpoint gzip snapshots,prior report,experiment responses,255 SHA256-verified measurement/source artifacts,results.json and contrasts.csv preserve provenance,seed outcomes,paired/conservative CIs,retention,duration effects and failures. Train CE remains explicitly null. Complete here means independently reviewable,not a completed causal theory.'
]
text='\n'.join(lines)+'\n'
with open('slow_sgd/report.md','w') as f:f.write(text)
print('report lines',len(lines),'chars',len(text))
