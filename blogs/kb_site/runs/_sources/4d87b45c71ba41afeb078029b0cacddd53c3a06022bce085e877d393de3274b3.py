import itertools,collections,json,math
from pathlib import Path
l=load_lab();g=collections.defaultdict(list)
for x in l:
 if x['family']=='xor_classification' and x['model']['input_dim']<8 and math.isfinite(x['mean']) and not x.get('excluded'):g[x['set_id']].append(x)
ps=[]
for v in g.values():
 for a,b in itertools.combinations(v,2):
  if a['residual']==b['residual'] or (a['optimizer']['type'],a['optimizer']['lr'])!=(b['optimizer']['type'],b['optimizer']['lr']):continue
  p,r=(a,b) if not a['residual'] else (b,a)
  if p['depth']<4 or not p['layer_norm'][-1] or p['n_ln']<r['n_ln']:continue
  o=p['optimizer'];t=p['budget']['training_steps']
  d=o['lr']*t if o['type'] in ['Adam','AdamW','RMSprop'] else 2*o['lr']*math.sqrt(t) if o['type']=='Adagrad' else o['lr']*t*.01/(1-o.get('momentum',0))
  ps.append(dict(set_id=p['set_id'],dataset_id=p['dataset_id'],p=p['candidate_id'],r=r['candidate_id'],delta=d,win=p['mean']<r['mean'],ratio=p['mean']/r['mean']))
Path('research/xor_lab_loose_pairs.json').write_text(json.dumps(ps,indent=2))
for low in [True,False]:
 a=[x for x in ps if (x['delta']<.05)==low]
 print(low,sum(x['win'] for x in a),len(a),len({x['set_id'] for x in a}),len({x['dataset_id'] for x in a}))
