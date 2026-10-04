import json,gzip,math,collections
with gzip.open('analysis/lab_snapshot.json.gz','rt') as f:L=json.load(f)
index={(x['set_id'],x['candidate_id']):x for x in L}
summaries={}
for name in ['strict_arch_pairs','strict_sgd_pairs','strict_spiral_pairs']:
 with open('analysis/'+name+'.json') as f:ps=json.load(f)
 groups=collections.defaultdict(list)
 for p in ps:
  a=index[(p['set'],p['a'])];b=index[(p['set'],p['b'])]
  assert all(a[k]==b[k] for k in ['dataset','budget','loss','init'])
  assert math.isfinite(a['mean']) and math.isfinite(b['mean'])
  if name=='strict_arch_pairs':
   assert a['optimizer']==b['optimizer']
   ma=dict(a['model']);mb=dict(b['model']);factor=p['factor']
   for m in [ma,mb]:
    m.pop(factor,None)
    if factor=='activation':m.pop('leaky_slope',None)
    if factor=='depth':m.pop('layer_norm',None)
   assert ma==mb
  else:assert a['model']==b['model']
  assert p['win']==(a['mean']<b['mean'])
  key=str((p['family'],p['region'],p['factor'],p['residual'] if p['factor']=='depth' else 'all'))
  groups[key].append(p)
 summaries[name]={k:{'wins':sum(p['win'] for p in xs),'n':len(xs),'sets':len(set(p['set'] for p in xs))} for k,xs in groups.items()}
with open('analysis/verified_initial_summary.json','w') as f:json.dump(summaries,f,indent=2)
print('Verified all saved pre-test pairs against frozen candidate measurements; no refreshed candidates pooled.')
print(json.dumps(summaries,indent=2))
