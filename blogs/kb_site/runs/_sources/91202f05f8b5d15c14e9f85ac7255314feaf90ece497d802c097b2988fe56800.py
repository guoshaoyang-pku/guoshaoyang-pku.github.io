import json, os, math, itertools, shutil
import numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
lab=json.load(open(ROOT+'/lab.json'))
base=[c for c in lab if c['family']=='bigram_lm' and not c['set_id'].startswith('exp:') and c['loss']=={'loss_id':'cross_entropy'}]
pairs=[]
for a,b in itertools.combinations(base,2):
    if a['set_id']!=b['set_id'] or a['model']!=b['model'] or a['budget']!=b['budget']: continue
    if {a['optimizer']['type'],b['optimizer']['type']}!={'Adagrad','RMSprop'}: continue
    ad=a if a['optimizer']['type']=='Adagrad' else b
    rm=b if ad is a else a
    if ad['optimizer']['lr']>0.0001 or rm['optimizer']['lr']!=0.003: continue
    pairs.append({'set':a['set_id'],'width':a.get('d_model'),'ad':ad,'rms':rm,'ratio':rm['mean']/ad['mean'],'margin':rm['mean']-ad['mean']})
json.dump(pairs,open(ROOT+'/base_pairs.json','w'),indent=2)
print('BASE pairs',len(pairs),'sets',len({p['set'] for p in pairs}),'RMS wins',sum(p['margin']<0 for p in pairs))
for p in pairs: print('BASE',p['width'],p['set'],p['ratio'],p['ad']['optimizer'],p['rms']['optimizer'])
rows=[]
for ds in ['bg_87ba9c','bg_9a3dce']:
    folder=ROOT+'/artifacts/'+ds
    for sub in os.listdir(folder):
        p=folder+'/'+sub
        spec=json.load(open(p+'/candidate_spec.json')); s=json.load(open(p+'/results/summary.json'))
        finite=[x for x in s['seed_results'] if not x['failed'] and math.isfinite(x['final_test_ce'])]
        curves=np.load(p+'/results/curves.npz')['curves']
        rows.append({'dataset':ds,'width':spec['model']['d_model'],'optimizer':spec['optimizer']['type'],'mean':s['mean_test_ce'],'sd':s['std_test_ce'],'n':len(finite),'failed':s['failed_seeds'],'seed_ids':[x['seed'] for x in finite],'ce_minus_uniform':s['mean_test_ce']-math.log(24),'test_after_first_update':float(curves[:,0].mean()),'test_last32':float(curves[:,-32:].mean()),'spec':spec,'summary':s,'source':p})
for r in rows: print('CELL',r['dataset'],r['width'],r['optimizer'],r['mean'],r['sd'],r['n'],r['failed'],'uniformdiff',r['ce_minus_uniform'],'test_trajectory',r['test_after_first_update'],r['test_last32'])
comparisons=[]
for ds in ['bg_87ba9c','bg_9a3dce']:
    for w in [64,128]:
        rr={r['optimizer']:r for r in rows if r['dataset']==ds and r['width']==w}
        for rival in ['RMSprop','SGD']:
            a=rr['Adagrad'];b=rr[rival]
            # Same seed identities give descriptive paired differences, not independent dataset replication.
            av={s['seed']:s['final_test_ce'] for s in a['summary']['seed_results'] if not s['failed']}
            bv={s['seed']:s['final_test_ce'] for s in b['summary']['seed_results'] if not s['failed']}
            d=np.array([bv[s]-av[s] for s in av if s in bv])
            c={'dataset':ds,'width':w,'rival':rival,'rival_over_adagrad':b['mean']/a['mean'],'rival_minus_adagrad':b['mean']-a['mean'],'paired_sd':float(d.std(ddof=1)),'paired_margin_interval_t9':(float(d.mean()-2.262*d.std(ddof=1)/math.sqrt(len(d))),float(d.mean()+2.262*d.std(ddof=1)/math.sqrt(len(d)))),'adagrad_seed_wins':int((d>0).sum())}
            comparisons.append(c); print('COMPARE',c)
json.dump({'rows':rows,'comparisons':comparisons},open(ROOT+'/analysis.json','w'),indent=2)

broader=[]
for ad in base:
 if ad['optimizer']['type']!='Adagrad' or ad['optimizer']['lr']>0.0001: continue
 for rm in base:
  if rm['set_id']==ad['set_id'] and rm['budget']==ad['budget'] and rm['model_type']=='transformer_lm' and rm.get('d_model')==64 and rm['optimizer']['type']=='RMSprop' and rm['optimizer']['lr']==0.003:
   broader.append({'ad':ad,'rms':rm,'finite_pair':math.isfinite(ad['mean']) and math.isfinite(rm['mean'])})
wide=[r for r in base if r['model_type']=='transformer_lm' and r.get('d_model')==128 and r['optimizer']['type']=='RMSprop' and r['optimizer']['lr']==0.003]
json.dump({'broader_pairs':broader,'wide_unpaired':wide},open(ROOT+'/base_context.json','w'),indent=2)
