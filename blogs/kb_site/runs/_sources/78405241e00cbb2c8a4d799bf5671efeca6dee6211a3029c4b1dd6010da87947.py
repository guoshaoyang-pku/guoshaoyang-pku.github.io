import math, json, itertools, collections
import numpy as np

def delta(x):
 o=x['optimizer']; t=x['budget']['training_steps']; lr=o['lr']; typ=o['type'].lower()
 if typ=='adagrad': return 2*lr*math.sqrt(t)
 if typ=='sgd': return lr*t*(.03 if x['family'] in ['multivariate_regression','univariate_regression'] and o.get('momentum',0)>0 else .01)/(1-o.get('momentum',0))
 return lr*t

def summary(ps):
 return dict(n=len(ps),sets=len({p['a']['set_id'] for p in ps}),datasets=len({p['a']['dataset_id'] for p in ps}),wins=sum(p['win'] for p in ps),median=float(np.median([p['ratio'] for p in ps])) if ps else None)

def make_pairs(L):
 P=[]
 for sid,rs in itertools.groupby(sorted(L,key=lambda x:x['set_id']),lambda x:x['set_id']):
  rs=list(rs)
  for a,b in itertools.permutations(rs,2):
   if a['family'] not in ['multivariate_regression','univariate_regression'] or a.get('model_type')!='mlp' or b.get('model_type')!='mlp':continue
   if a['loss']['loss_id']!='mse' or b['loss']['loss_id']!='mse':continue
   if a['residual'] or not b['residual'] or a['n_ln']!=b['n_ln']+1:continue
   da,db=delta(a),delta(b)
   if not .5<=da/db<=2:continue
   P.append(dict(a=a,b=b,da=da,db=db,ratio=a['mean']/b['mean'],win=a['mean']<b['mean']))
 return P

P=make_pairs([x for x in load_lab() if not x.get('excluded')])
for fam in ['multivariate_regression','univariate_regression']:
 for low in [True,False]:
  ps=[p for p in P if p['a']['family']==fam and (max(p['da'],p['db'])<.1)==low]
  print(fam,low,summary(ps))
open('pairs.json','w').write(json.dumps(P,indent=2))
