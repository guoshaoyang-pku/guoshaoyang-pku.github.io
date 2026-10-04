import json,os,hashlib,numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
cells=json.load(open(ROOT+'/cells.json'));manifest=json.load(open(ROOT+'/manifest.json'))
for x in manifest:assert hashlib.sha256(open(ROOT+'/'+x['saved'],'rb').read()).hexdigest()==x['sha256']
unique={}
for c in cells:
 s=c['spec'];o=s['optimizer'];key=(c['dataset'],s['budget']['training_steps'],o['type'],o['lr'],o['weight_decay'],tuple(o.get('betas',[])) if o['type']!='RMSprop' else ())
 if key in unique:assert c['summary']['seed_results']==unique[key]['summary']['seed_results']
 else:unique[key]=c
prefixes=0
for key,c in unique.items():
 sm=c['summary'];vals=np.array([r['final_test_ce'] for r in sm['seed_results']])
 assert len(vals)==10 and sm['failed_seeds']==0 and not sm['excluded']
 assert abs(vals.mean()-sm['mean_test_ce'])<1e-12
 path=ROOT+'/artifacts/'+c['dataset']+'/'+c['spec']['candidate_id']
 z=np.load(path+'/results/curves.npz')['curves'];assert np.allclose(z[:,-1],vals,rtol=0,atol=0)
 for t in [256,512]:
  if t>=key[1]:continue
  k=(key[0],t,*key[2:])
  if k in unique:
   p=unique[k];zz=np.load(ROOT+'/artifacts/'+p['dataset']+'/'+p['spec']['candidate_id']+'/results/curves.npz')['curves']
   assert np.array_equal(z[:,:t],zz);prefixes+=1
json.dump(list(unique.values()),open(ROOT+'/unique_cells.json','w'),indent=2)
print('Validated',len(manifest),'saved hashes;',len(unique),'unique operational cells;',prefixes,'exact budget prefixes; zero flags')
for ds in ['bg_1d9caf','bg_209ff2']:
 print(ds)
 for t in [256,512,1024]:
  cs=[x for k,x in unique.items() if k[0]==ds and k[1]==t]
  if t==256:
   print([(x['spec']['optimizer']['type'],x['spec']['optimizer']['lr'],round(x['summary']['mean_test_ce'],6)) for x in cs])
