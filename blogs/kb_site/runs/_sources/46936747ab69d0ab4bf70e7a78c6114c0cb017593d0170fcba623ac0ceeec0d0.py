import json,hashlib,os
from pathlib import Path
import numpy as np
root=Path('rate_shape')
d32=json.load(open(root/'reused_d32.json'))
old=json.load(open(root/'reused_rate_ff.json'))
manifest={}
unique={}
for r in old+d32:
 unique[(r['dataset_id'],json.dumps(r['model'],sort_keys=True),json.dumps(r['optimizer'],sort_keys=True),json.dumps(r['budget'],sort_keys=True))]=r
 for m in r['measurement_files'].values():
  p=Path(m['repo_path']);data=Path(m['path']).read_bytes()
  assert hashlib.sha256(data).hexdigest()==m['sha256']
  p.parent.mkdir(exist_ok=True);p.write_bytes(data);manifest[str(p)]=m['sha256']
 v=np.array([s['final_test_ce'] for s in r['seed_results']])
 assert abs(v.mean()-r['mean'])<1e-12
 f=np.load(r['measurement_files']['results/curves.npz']['repo_path'])
 if f['curves'].shape[1]:assert np.max(np.abs(f['curves'][:,-1]-v))<1e-12
json.dump(manifest,open(root/'reused_manifest.json','w'),indent=2)
print('reused rate/FF rows',len(old),'d32 rows',len(d32),'unique',len(unique),'verified artifacts',len(manifest))
for ds in ['bg_1de357','bg_313ea4','bg_05e970','bg_22dc04']:
 rr=[r for r in old if r['dataset_id']==ds and r['d_ff']==64 and r['optimizer']['type']=='RMSprop']
 print(ds,sorted(set((r['optimizer']['lr'],round(r['mean'],6)) for r in rr)))
for ds in ['bg_08f706','bg_20ea99']:
 print(ds,[(r['d_model'],r['num_layers'],r['num_heads'],round(r['mean'],6)) for r in d32 if r['dataset_id']==ds])
