import json, os, hashlib, itertools
import numpy as np
from pathlib import Path
root=Path('rate_shape')
rows=json.load(open(root/'new_cells.json'))
def vec(r): return np.array([s['final_test_ce'] for s in sorted(r['seed_results'],key=lambda s:s['seed'])],float)
def key(r): return (r['dataset_id'],r['model_type'],r['d_model'],r['num_layers'],r['optimizer']['lr'],r['budget']['training_steps'])
cells={key(r):r for r in rows}
assert len(cells)==38
manifest={}
progress=[]
for r in rows:
 v=vec(r)
 assert len(v)==10
 if not r['excluded']:
  assert abs(v.mean()-r['mean'])<1e-12
  assert abs(v.std()-r['std'])<1e-12
 assert r['cached']==False
 assert r['source_provenance']['status']=='verified'
 for name,m in r['measurement_files'].items():
  p=m['repo_path']; data=Path(p).read_bytes()
  assert hashlib.sha256(data).hexdigest()==m['sha256']
  manifest[p]=m['sha256']
 f=np.load(r['measurement_files']['results/curves.npz']['repo_path'])
 curve=f['curves']
 out={'key':key(r),'excluded':r['excluded'],'failed_seeds':r['failed_seeds'],'raw_endpoint_mean':float(v.mean()),'raw_endpoint_sd':float(v.std(ddof=1)),'raw_endpoint_range':[float(v.min()),float(v.max())],'initial_ce':None,'train_ce':None,'progress':{}}
 if curve.shape[1]:
  assert curve.shape==(10,r['budget']['training_steps'])
  assert np.max(np.abs(curve[:,-1]-v))<1e-12
  assert np.array_equal(f['samples'],np.arange(1,curve.shape[1]+1)*32)
  for step in [1,64,128,256,512,1024]:
   if step<=curve.shape[1]:out['progress'][str(step)]={'mean':float(curve[:,step-1].mean()),'sd':float(curve[:,step-1].std(ddof=1)),'seeds':curve[:,step-1].tolist()}
 else:
  assert r['excluded']
 out['curve_shape']=list(curve.shape)
 progress.append(out)
contrasts=[]
def compare(a,b,label,pred=None):
 ra,rb=cells[a],cells[b]; x=vec(ra)-vec(rb)
 accepted=not(ra['excluded'] or rb['excluded'])
 mu=float(x.mean()); half=2.2621571628540993*float(x.std(ddof=1))/np.sqrt(10)
 contrasts.append({'label':label,'a':a,'b':b,'accepted':accepted,'mean':mu,'ci':[mu-half,mu+half],'wins':int((x<0).sum()),'seed_differences':x.tolist(),'prediction':pred,'prediction_succeeded':None if not accepted or pred is None else bool(mu*pred>0)})
for ds,T in [('bg_1864c9',256),('bg_44e9f3',256),('bg_1864c9',1024)]:
 def K(w,d,lr):return(ds,'transformer_lm',w,d,lr,T)
 for w,d in itertools.product([64,128],[1,4]):
  for hi,lo in [(0.0003,0.0001),(0.001,0.0003),(0.001,0.0001)]:
   pred=None
   if T==256 and d==1 and (hi,lo)==(.0003,.0001):pred=-1
   if T==256 and d==4 and (hi,lo)==(.001,.0003):pred=1
   if T==1024 and (hi,lo)==(.0003,.0001):pred=-1 if d==1 else 1
   compare(K(w,d,hi),K(w,d,lo),'rate',pred)
 for d,lr in itertools.product([1,4],[.0001,.0003,.001]):
  pred=(-1 if lr==.0001 else 1 if lr==.001 else None) if T==256 else None
  compare(K(128,d,lr),K(64,d,lr),'width128-minus64',pred)
 for w,lr in itertools.product([64,128],[.0001,.0003,.001]):
  pred=(-1 if lr==.0001 else 1 if lr==.001 else None) if T==256 else None
  compare(K(w,4,lr),K(w,1,lr),'depth4-minus1',pred)
for w,d,lr in itertools.product([64,128],[1,4],[.0001,.0003,.001]):
 compare(('bg_1864c9','transformer_lm',w,d,lr,1024),('bg_1864c9','transformer_lm',w,d,lr,256),'budget1024-minus256')
compare(('bg_1864c9','gru_lm',64,2,.0003,1024),('bg_1864c9','gru_lm',64,2,.0001,1024),'GRU rate',-1)
json.dump(progress,open(root/'progress.json','w'),indent=2)
json.dump(contrasts,open(root/'contrasts.json','w'),indent=2)
json.dump(manifest,open(root/'manifest.json','w'),indent=2)
print('verified',len(rows),'cells',len(manifest),'artifacts',len(contrasts),'contrasts')
for c in contrasts:
 if c['prediction'] is not None or c['label']=='budget1024-minus256':
  print(c['label'],c['a'],c['b'][4],c['accepted'],round(c['mean'],6),[round(z,6) for z in c['ci']],c['wins'],c['prediction_succeeded'])
print('prediction successes',sum(c['prediction_succeeded'] is True for c in contrasts),'rejected',sum(c['prediction_succeeded'] is False for c in contrasts),'unresolved excluded',sum(c['prediction'] is not None and not c['accepted'] for c in contrasts))
