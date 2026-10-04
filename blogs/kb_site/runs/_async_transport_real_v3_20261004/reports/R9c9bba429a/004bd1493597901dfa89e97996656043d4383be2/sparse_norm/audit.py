import json, math, collections, csv
from pathlib import Path
P=Path(__file__).parent
lab=json.loads((P/'lab_snapshot.json').read_text())
def delta(x):
 o=x['optimizer']; t=x['budget']['training_steps'];lr=o['lr']; typ=o['type']
 if typ=='Adagrad':return 2*lr*math.sqrt(t)
 if typ=='SGD':
  m=o.get('momentum',0);g=0.03 if m and 'regression' in x['family'] else 0.01
  return lr*t*g/(1-m)
 return lr*t

def tail(x):
 ln=x['layer_norm'];return len(ln)-1-max(i for i,z in enumerate(ln) if z)
sets=collections.defaultdict(list)
for x in lab:
 if not x['set_id'].startswith('exp:') and x['family'] in ['univariate_regression','multivariate_regression'] and not x.get('excluded') and not x.get('failed_seeds') and x['loss']['loss_id']=='mse':sets[x['set_id']].append(x)
rows=[]
for sid,xs in sets.items():
 for a in xs:
  if a['depth']<4 or a['n_ln']<1:continue
  for b in xs:
   if b['depth']>2 or a['n_ln']<b['n_ln']:continue
   ratio_delta=delta(a)/delta(b)
   matched=all(a.get(key)==b.get(key) for key in ['width','activation','residual','init']) and a['optimizer']==b['optimizer'] and a['budget']==b['budget'] and a['loss']==b['loss']
   rows.append(dict(set_id=sid,dataset=a['dataset_id'],family=a['family'],deep=a['candidate_id'],shallow=b['candidate_id'],optimizer=a['optimizer']['type'],shallow_optimizer=b['optimizer']['type'],lr=a['optimizer']['lr'],activation=a['activation'],shallow_activation=b['activation'],residual=a['residual'],shallow_residual=b['residual'],ln=a['n_ln'],shallow_ln=b['n_ln'],tail=tail(a),depth=a['depth'],width=a['width'],shallow_width=b['width'],steps=a['budget']['training_steps'],delta=delta(a),delta_ratio=ratio_delta,loss_ratio=a['mean']/b['mean'],deep_sd=a['std'],shallow_sd=b['std'],matched=matched))
lookup={(x['set_id'],x['candidate_id']):x for x in lab}
for r in rows:
 for role in ['deep','shallow']:
  x=lookup[(r['set_id'],r[role])]
  for field in ['model','optimizer','budget','loss','init']:
   r[role+'_'+field+'_json']=json.dumps(x.get(field),sort_keys=True)
with (P/'observational_pairs.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
groups=collections.defaultdict(list)
for r in rows:
 if r['delta_ratio']>=1/3:
  key=tuple(r[k] for k in ['family','optimizer','shallow_optimizer','activation','residual','ln','tail','depth','width','shallow_width','lr','steps'])
  groups[key].append(r)
out=[]
for key,rs in sorted(groups.items(),key=lambda z:str(z[0])):
 out.append(dict(cell=key,n_pairs=len(rs),n_sets=len({r['set_id'] for r in rs}),n_datasets=len({r['dataset'] for r in rs}),deep_wins=sum(r['loss_ratio']<1 for r in rs),geometric_mean_ratio=math.exp(sum(math.log(r['loss_ratio']) for r in rs)/len(rs))))
(P/'descriptive_strata.json').write_text(json.dumps(out,indent=2))
exact=[]
for r in rows:
 if r['delta_ratio']>=1/3:
  exact.append({k:r[k] for k in r if k not in ['matched']})
(P/'support_cells.json').write_text(json.dumps(exact,indent=2))
matched=[r for r in rows if r['matched']]
(P/'nuisance_matched_pairs.json').write_text(json.dumps(matched,indent=2))
print('eligible',len(rows),'stratified cells',len(out),'nuisance-matched',len(matched))
for fam in ['univariate_regression','multivariate_regression']:
 for gate in [True,False]:
  rs=[r for r in rows if r['family']==fam and (r['delta_ratio']>=1/3)==gate]
  print(fam,'delta>=1/3',gate,'pairs',len(rs),'wins',sum(r['loss_ratio']<1 for r in rs),'datasets',len({r['dataset'] for r in rs}))
print('matched cells',matched)
