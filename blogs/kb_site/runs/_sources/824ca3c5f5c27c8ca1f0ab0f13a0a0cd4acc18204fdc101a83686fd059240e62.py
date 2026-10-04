import collections,itertools,json,math
from pathlib import Path

def delta(x):
 o=x['optimizer'];t=x['budget']['training_steps']
 if o['type'] in ('Adam','AdamW','RMSprop'): return o['lr']*t
 if o['type']=='Adagrad': return 2*o['lr']*math.sqrt(t)
 g=.03 if x['family'].endswith('regression') and o.get('momentum',0)==.9 else .01
 return o['lr']*t*g/(1-o.get('momentum',0))

def pairs(lab,omit):
 groups=collections.defaultdict(list)
 for x in lab:
  if x['model_type']!='mlp' or x.get('excluded') or x.get('failed_seeds') or not math.isfinite(x['mean']) or x['mean']<=0: continue
  m={k:v for k,v in x['model'].items() if k not in omit}
  key=json.dumps([x['set_id'],m,x['optimizer'],x['budget'],x['loss'],x.get('init')],sort_keys=True)
  groups[key].append(x)
 for g in groups.values(): yield from itertools.combinations(g,2)

def summary(rows):
 if not rows:return {'wins':0,'n':0}
 import numpy as np
 return {'wins':sum(r['win'] for r in rows),'n':len(rows),'sets':len({r['set_id'] for r in rows}),'datasets':len({(r['family'],r['dataset_id']) for r in rows}),'median_ratio':float(np.median([r['ratio'] for r in rows]))}

def audit(lab):
 rows=[]
 for kind,omit in [('placement',{'layer_norm'}),('residual',{'residual'}),('ln_count',{'layer_norm'}),('smooth',{'activation','leaky_slope'})]:
  for a,b in pairs(lab,omit):
   if kind=='placement':
    if a['n_ln']!=b['n_ln'] or not a['n_ln']:continue
    la=max(i for i,f in enumerate(a['layer_norm']) if f);lb=max(i for i,f in enumerate(b['layer_norm']) if f)
    if la==lb:continue
    a,b=(a,b) if la>lb else (b,a)
   elif kind=='residual':
    if a['residual']==b['residual']:continue
    a,b=(a,b) if a['residual'] else (b,a)
   elif kind=='ln_count':
    if a['n_ln']==b['n_ln']:continue
    a,b=(a,b) if a['n_ln']>b['n_ln'] else (b,a)
   else:
    sa=a['activation'] in ('gelu','silu');sb=b['activation'] in ('gelu','silu')
    if sa==sb:continue
    a,b=(a,b) if sa else (b,a)
   rows.append(dict(kind=kind,family=a['family'],dataset_id=a['dataset_id'],set_id=a['set_id'],a=a['candidate_id'],b=b['candidate_id'],delta=delta(a),raw_lrT=a['optimizer']['lr']*a['budget']['training_steps'],optimizer=a['optimizer']['type'],residual=a['residual'],activation=a['activation'],depth=a['depth'],width=a['width'],ln_a=a['layer_norm'],ln_b=b['layer_norm'],samples=a['budget']['total_samples_seen'],ratio=a['mean']/b['mean'],win=a['mean']<b['mean'],mean_a=a['mean'],mean_b=b['mean'],std_a=a['std'],std_b=b['std']))
 return rows

rows=audit(load_lab());out={}
for kind in ['placement','residual','ln_count','smooth']:
 for family in sorted({r['family'] for r in rows}):
  for band in ['low','high']:
   subset=[r for r in rows if r['kind']==kind and r['family']==family and (r['delta']<.1)==(band=='low')]
   if subset:out[f'{kind}/{family}/{band}']=summary(subset)
for band in ['below','above']:
 subset=[r for r in rows if r['kind']=='placement' and r['family']=='multivariate_regression' and r['optimizer'] in ('Adam','AdamW') and (r['raw_lrT']<.03)==(band=='below')]
 out['placement/multivariate/Adam-family/'+band+'-.03']=summary(subset)
Path('research/strict_lab_pairs.json').write_text(json.dumps(rows,indent=2))
Path('research/strict_lab_summary.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
