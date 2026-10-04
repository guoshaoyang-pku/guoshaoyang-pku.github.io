import json,numpy as np,hashlib,os
def margin(a,b):
 x=np.array([s['final_test_ce'] for s in a['seed_results']])-np.array([s['final_test_ce'] for s in b['seed_results']])
 m=float(x.mean());h=2.262157*float(x.std(ddof=1))/np.sqrt(len(x))
 return {'mean_difference':m,'ci95':[m-h,m+h],'first_wins':int((x<0).sum()),'n':len(x)}
def key(v):
 d=v['model.depth']; mask=v['model.layer_norm']
 return (d,v['model.residual'],v['model.activation'],'zero' if not any(mask) else 'all' if all(mask) else 'one')

out={}
for ds in ['xorcls_12d3d2','xorcls_13a317']:
 r=json.load(open('xor_audit/boundary_'+ds+'.json'));D={key(x['variant']):x for x in r['results']}
 tests=[]
 for x in D.values():
  for f in x['measurement_files'].values():assert hashlib.sha256(open(f['repo_path'],'rb').read()).hexdigest()==f['sha256']
 for a in ['silu','gelu']:
  for ka,kb in [((4,False,a,'all'),(2,False,a,'one')),((4,True,a,'zero'),(4,False,a,'zero'))]:
   tests.append({'first':ka,'second':kb,**margin(D[ka],D[kb])})
 out[ds]=tests;print(ds,tests)
json.dump(out,open('xor_audit/boundary_statistics.json','w'),indent=2)
