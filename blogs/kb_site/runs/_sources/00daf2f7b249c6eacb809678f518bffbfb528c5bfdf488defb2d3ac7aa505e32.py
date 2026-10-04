import json,gzip,glob,itertools,hashlib,os
import numpy as np
ROOT='xor_audit'
def margin(a,b):
 x=np.array([s['final_test_ce'] for s in a['seed_results']])-np.array([s['final_test_ce'] for s in b['seed_results']])
 m=float(x.mean());h=2.262157*float(x.std(ddof=1))/np.sqrt(len(x))
 return {'mean_difference':m,'ci95':[m-h,m+h],'first_wins':int((x<0).sum()),'n':len(x)}
def key(v):
 d=v['model.depth']; mask=v['model.layer_norm']
 return (d,v['model.residual'],v['model.activation'],'zero' if not any(mask) else 'all' if all(mask) else 'one')
out={};phase=[];verified=0
for ds in ['xorcls_12d3d2','xorcls_13a317']:
 rows=sum([json.load(open(p))['results'] for p in sorted(glob.glob(ROOT+'/receipt_'+ds+'_*.json'))],[])
 D={key(r['variant']):r for r in rows}; assert len(D)==36
 for r in rows:
  for f in r['measurement_files'].values():
   assert hashlib.sha256(open(f['repo_path'],'rb').read()).hexdigest()==f['sha256'];verified+=1
  c=np.load(r['measurement_files']['results/curves.npz']['repo_path'])
  assert c['curves'].shape==(10,256)
  assert np.allclose(c['curves'][:,-1],[s['final_test_ce'] for s in r['seed_results']],atol=1e-10)
  r['update1_mean']=float(c['curves'][:,0].mean())
  src=os.path.dirname(r['measurement_files']['candidate_spec.json']['path'])
  dest=ROOT+'/recipes/'+ds+'/'+r['measurement_files']['candidate_spec.json']['sha256']
  os.makedirs(dest,exist_ok=True)
  for f in ['model.py','optimizer.py','loss.py','train.py']:
   open(dest+'/'+f,'w').write(open(src+'/'+f).read())
 tests={p:[] for p in ['P1','P2','P3','P4','P5']}
 def add(p,ka,kb):
  tests[p].append({'first':ka,'second':kb,**margin(D[ka],D[kb])})
 for d,r,m,a in itertools.product([2,4],[False,True],['zero','one','all'],['silu','gelu']):
  add('P1',(d,r,'leaky_relu',m),(d,r,a,m))
 for d,a,m in itertools.product([2,4],['silu','gelu','leaky_relu'],['zero','one','all']):
  add('P2',(d,True,a,m),(d,False,a,m))
 for d,r,a in itertools.product([2,4],[False,True],['silu','gelu']):
  add('P3',(d,r,a,'one'),(d,r,a,'zero'))
 for a in ['silu','gelu']:add('P4',(2,False,a,'one'),(4,False,a,'all'))
 for a,r in itertools.product(['silu','gelu'],[False,True]):
  for k in D:
   if k[2]==a and k!=(4,r,a,'zero'):add('P5',k,(4,r,a,'zero'))
 out[ds]={'tests':tests,'cells':[{'key':k,'mean':r['mean'],'std':r['std'],'update1_mean':r['update1_mean'],'failed':r['failed_seeds']} for k,r in D.items()]}
 for p,ts in tests.items():print(ds,p,sum(t['mean_difference']<0 for t in ts),'/',len(ts),'CI decisive',sum(t['ci95'][1]<0 for t in ts),sum(t['ci95'][0]>0 for t in ts))
 for a in ['silu','gelu']:
  print(ds,'P4',a,margin(D[(4,False,a,'all')],D[(2,False,a,'one')]))
 for d,r in itertools.product([2,4],[False,True]):
  phase.append([ds,d,r]+[round(D[(d,r,a,m)]['mean'],9) for a in ['silu','gelu','leaky_relu'] for m in ['zero','one','all']])
print('verified',verified)
json.dump(out,open(ROOT+'/statistics.json','w'),indent=2)
json.dump(phase,open(ROOT+'/phase.json','w'),indent=2)
z=json.load(gzip.open(ROOT+'/frozen.json.gz','rt'))
# Freeze base-only rows: prior experiments are not independent backtests.
lab=[r for r in z['lab'] if r['family']=='xor_classification' and r['model_type']=='mlp' and not r.get('excluded',False) and r.get('failed_seeds',0)==0 and not r['candidate_id'].startswith('x_') and r['dataset']['input_dim']==2 and r['optimizer']['type']=='SGD' and r['optimizer']['lr']==.0003 and r['optimizer'].get('momentum',0)==0 and r['budget']['training_steps']==256 and r['metric']=='test_ce' and r['loss']['loss_id']=='cross_entropy']
pairs=[]
for a,b in itertools.combinations(lab,2):
 if a['set_id']!=b['set_id']:continue
 full=a['optimizer']==b['optimizer'] and a['budget']==b['budget']
 # Low-rate contextual pair predicates; all remain bundled unless single-factor matching.
 eq=lambda fields: all(a.get(f)==b.get(f) for f in fields)
 for p in ['P1','P2','P3']:
  first=second=None; isolated=False
  if p=='P1' and (a['activation'] in ['relu','leaky_relu']) != (b['activation'] in ['relu','leaky_relu']) and {a['activation'],b['activation']} & {'silu','gelu'}:
   first,second=(a,b) if a['activation'] in ['relu','leaky_relu'] else (b,a)
   isolated=eq(['depth','width','residual','layer_norm','init'])
  if p=='P2' and a['residual']!=b['residual']:
   first,second=(a,b) if a['residual'] else (b,a);isolated=eq(['depth','width','activation','layer_norm','init'])
  if p=='P3' and a['activation']==b['activation'] and a['activation'] in ['silu','gelu'] and {a['n_ln'],b['n_ln']}=={0,1}:
   first,second=(a,b) if a['n_ln']==1 else (b,a)
   if not first['layer_norm'][-1]:continue
   isolated=eq(['depth','width','activation','residual','init'])
  if first:
   pairs.append({'prediction':p,'set_id':a['set_id'],'dataset_id':a['dataset_id'],'first':first['candidate_id'],'second':second['candidate_id'],'margin':first['mean']-second['mean'],'full_optimizer_budget':full,'isolated':isolated})
json.dump(pairs,open(ROOT+'/backtest_lab.json','w'),indent=2)
for p in ['P1','P2','P3']:
 for tag,fun in [('type_lr',lambda x:True),('full_config',lambda x:x['full_optimizer_budget']),('isolated_full',lambda x:x['full_optimizer_budget'] and x['isolated'])]:
  q=[x for x in pairs if x['prediction']==p and fun(x)];print('lab',p,tag,sum(x['margin']<0 for x in q),len(q),'sets',len({x['set_id'] for x in q}),'datasets',len({x['dataset_id'] for x in q}),'candidates',len({(x['set_id'],c) for x in q for c in [x['first'],x['second']]}))
