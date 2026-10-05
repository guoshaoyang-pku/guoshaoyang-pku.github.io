import json,hashlib
from pathlib import Path
import numpy as np
rows=json.load(open('rate_shape/reused_adamw_phase.json'))
manifest={};unique={};contrasts=[]
for r in rows:
 k=(r['dataset_id'],json.dumps(r['model'],sort_keys=True),json.dumps(r['optimizer'],sort_keys=True))
 unique[k]=r
 v=np.array([s['final_test_ce'] for s in r['seed_results']])
 if not r['excluded']:assert abs(v.mean()-r['mean'])<1e-12
 for m in r.get('measurement_files',{}).values():
  data=Path(m['path']).read_bytes()
  assert hashlib.sha256(data).hexdigest()==m['sha256']
  p=Path(m['repo_path']);p.parent.mkdir(exist_ok=True);p.write_bytes(data);manifest[str(p)]=m['sha256']
rows=list(unique.values())
for i,a in enumerate(rows):
 for b in rows[i+1:]:
  if a['dataset_id']!=b['dataset_id'] or a['excluded'] or b['excluded']:continue
  vm=lambda r:np.array([s['final_test_ce'] for s in sorted(r['seed_results'],key=lambda z:z['seed'])])
  x=vm(a)-vm(b);mu=float(x.mean());h=float(2.2621571628540993*x.std(ddof=1)/np.sqrt(10))
  contrasts.append({'a':[a['set_id'],a['candidate_id']],'b':[b['set_id'],b['candidate_id']],'mean':mu,'ci':[mu-h,mu+h],'wins':int((x<0).sum())})
json.dump(manifest,open('rate_shape/reused_adamw_manifest.json','w'),indent=2)
json.dump(contrasts,open('rate_shape/reused_adamw_contrasts.json','w'),indent=2)
print('separate AdamW rows',len(rows),'hashed artifacts',len(manifest),'within-table contrasts',len(contrasts))
