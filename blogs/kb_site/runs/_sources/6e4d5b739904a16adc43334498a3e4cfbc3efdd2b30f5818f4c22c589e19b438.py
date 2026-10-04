import json, collections, math
import glob
lab=[x for f in glob.glob('audit/sets/*.json') for x in json.load(open(f))]
def delta(x):
 o=x['optimizer'];t=x['budget']['training_steps'];lr=o['lr'];typ=o['type']
 if typ=='SGD': return lr*t*(.03 if 'regression' in x['family'] and o.get('momentum',0)==.9 else .01)/(1-o.get('momentum',0))
 if typ=='Adagrad':return 2*lr*math.sqrt(t)
 return lr*t
sets=collections.defaultdict(list)
for x in lab:
 if not x.get('excluded') and not x.get('failed_seeds'):sets[x['set_id']].append(x)
pairs=[]
for sid, xs in sets.items():
 for x in xs:
  if x['model_type']!='mlp' or x['optimizer']['type']!='SGD' or x.get('n_ln')!=0 or x['depth']<3:continue
  for y in xs:
   if y['optimizer']['type'] not in ['Adam','AdamW','RMSprop','Adagrad']:continue
   if delta(y)>=.1 or delta(x)<3*delta(y):continue
   pairs.append({'set':sid,'sgd':x['candidate_id'],'rival':y['candidate_id'],'family':x['family'],'residual':x['residual'],'act':x['activation'],'width':x['width'],'init':x.get('init'),'sgd_opt':x['optimizer'],'rival_opt':y['optimizer'],'budget':x['budget'],'rival_budget':y['budget'],'metric':x['metric'],'sgd_mean':x['mean'],'rival_mean':y['mean'],'difference':x['mean']-y['mean'],'ratio':x['mean']/y['mean'],'win':x['mean']<y['mean'],'exact_model':x['model']==y['model'],'same_loss':x['loss']==y['loss'],'same_budget':x['budget']==y['budget'],'rival_model':y['model'],'rival_init':y.get('init')})
json.dump(pairs,open('audit/sgd_pairs.json','w'),indent=2)
print('SGD pairs',len(pairs),'wins',sum(p['win'] for p in pairs),'sets',len(set(p['set'] for p in pairs)))
for keys in [('family',),('residual',),('act',),('width',),('init',),('exact_model','same_loss','same_budget')]:
 groups=collections.defaultdict(list)
 for p in pairs:groups[tuple(p[k] for k in keys)].append(p)
 for key, ps in groups.items():print(keys,key,'n',len(ps),'wins',sum(p['win'] for p in ps),'sets',len(set(p['set'] for p in ps)),'median_diff',round(__import__('numpy').median([p['difference'] for p in ps]),6),'median_ratio',round(__import__('numpy').median([p['ratio'] for p in ps]),4))
print('representative pairs')
for p in pairs[:3]:print(json.dumps(p))
