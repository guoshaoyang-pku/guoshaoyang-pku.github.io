import json,itertools
import numpy as np
from pathlib import Path
root=Path('rate_shape')
out=[]
rows=json.load(open(root/'reused_rate_ff.json'))+json.load(open(root/'reused_d32.json'))
def v(r):return np.array([s['final_test_ce'] for s in sorted(r['seed_results'],key=lambda s:s['seed'])])
def contrast(a,b,label):
 x=v(a)-v(b);mu=float(x.mean());h=2.2621571628540993*x.std(ddof=1)/np.sqrt(10)
 out.append({'a':[a['set_id'],a['candidate_id']],'b':[b['set_id'],b['candidate_id']],'table':a['dataset_id'],'label':label,'mean':mu,'ci':[float(mu-h),float(mu+h)],'wins':int((x<0).sum()),'seed_differences':x.tolist()})
for a,b in itertools.combinations(rows,2):
 if a['dataset_id']!=b['dataset_id']:continue
 if a['model']==b['model'] and a['optimizer']['type']==b['optimizer']['type'] and a['optimizer']['weight_decay']==b['optimizer']['weight_decay']:
  contrast(a,b,'reused-rate')
 if a['optimizer']==b['optimizer'] and sum(a['model'].get(k)!=b['model'].get(k) for k in set(a['model'])|set(b['model']))==1:
  contrast(a,b,'reused-shape')
json.dump(out,open(root/'reused_contrasts.json','w'),indent=2)
new=json.load(open(root/'new_cells.json'));prefixes=0
for a,b in itertools.combinations(new,2):
 if a['dataset_id']==b['dataset_id'] and a['model']==b['model'] and a['optimizer']==b['optimizer'] and a['budget']['training_steps']!=b['budget']['training_steps'] and not a['excluded'] and not b['excluded']:
  c=np.load(a['measurement_files']['results/curves.npz']['repo_path'])['curves']
  d=np.load(b['measurement_files']['results/curves.npz']['repo_path'])['curves']
  n=min(c.shape[1],d.shape[1]);assert np.array_equal(c[:,:n],d[:,:n]);prefixes+=1
print('reused contrasts',len(out),'exact verified budget prefixes',prefixes)
