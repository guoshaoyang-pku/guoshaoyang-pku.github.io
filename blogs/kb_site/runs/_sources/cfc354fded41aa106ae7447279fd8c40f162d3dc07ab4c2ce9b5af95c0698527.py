import json, pathlib, hashlib
import numpy as np
P=pathlib.Path(__file__).resolve().parent
data=json.loads((P/'resolved.json').read_text())
out=[]
for d in data:
 r=d['results']
 for x in r:
  assert x['n_seeds']==10 and x['failed_seeds']==0 and not x['cached']
  v=x['variant']
  assert v['loss.loss_id']=='mse' and v['budget.batch_size']==32
  vals=[s['final_test_mse'] for s in x['seed_results']]
  assert np.isclose(np.mean(vals),x['mean'])
 def effect(i,j):
  a,b=r[i],r[j]
  av=np.array([s['final_test_mse'] for s in a['seed_results']])
  bv=np.array([s['final_test_mse'] for s in b['seed_results']])
  diff=av-bv
  return dict(i=i,j=j,margin=float(diff.mean()),ratio=a['mean']/b['mean'],first_wins=int(sum(av<bv)),sd_diff=float(diff.std(ddof=1)),ci95=(diff.mean()+np.array([-1,1])*2.262157*diff.std(ddof=1)/np.sqrt(10)).tolist())
 out.append(dict(dataset=d['dataset'],full=effect(7,0),optimizer_recipe=[effect(i+1,i) for i in [0,2,4,6]],residual=[effect(i,i+2) for i in [0,1,4,5]],algorithm=[effect(i,j) for i,j in [(8,0),(1,9),(10,2),(3,11)]]))
(P/'effects.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
