import json, hashlib, pathlib, numpy as np
selected=json.load(open('selected_blocks.json'))
ids={r['dataset_id'] for r in selected}|{'mvar_0fbae4'}
rows=[json.loads(s) for s in open(EXP_PATH) if s.strip()]
rows=[r for r in rows if r['dataset_id'] in ids and r.get('why','').startswith('Preregistered') and r.get('model',{}).get('width')==192 and r.get('model',{}).get('depth')==3 and r.get('budget',{}).get('batch_size')==64]
unique={}
for r in rows: unique[(r['set_id'],r['candidate_id'])]=r
rows=list(unique.values())
json.dump(rows,open('experiment_records.json','w'),indent=2)
out=[]
for d in sorted(ids):
 rr=[r for r in rows if r['dataset_id']==d]
 if len(rr)!=2: continue
 s=next(r for r in rr if r['optimizer']['type']=='SGD')
 a=next(r for r in rr if r['optimizer']['type']=='Adam')
 for r in rr:
  for f,m in r['measurement_files'].items():
   p=pathlib.Path(m['repo_path'])
   assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
  spec=json.load(open(r['measurement_files']['candidate_spec.json']['repo_path']))
  expected={'type':'mlp','depth':3,'width':192,'residual':False,'activation':'gelu','layer_norm':[False,True,False],'input_dim':8,'output_dim':1}
  assert all(spec['model'].get(k)==v for k,v in expected.items())
  assert spec['budget']=={'training_steps':256,'batch_size':64,'total_samples_seen':16384}
  assert spec['optimizer']['weight_decay']==.0001 and spec['loss']['loss_id']=='mse'
  z=np.load(r['measurement_files']['results/curves.npz']['repo_path'])
  if not r['excluded']:
   assert z['curves'].shape==(10,256) and z['samples'][-1]==16384
 if s['excluded'] or a['excluded']: continue
 sv=np.array([x['final_test_mse'] for x in sorted(s['seed_results'],key=lambda x:x['seed'])])
 av=np.array([x['final_test_mse'] for x in sorted(a['seed_results'],key=lambda x:x['seed'])])
 delta=sv-av
 assert np.isclose(sv.mean(),s['mean']) and np.isclose(av.mean(),a['mean'])
 half=2.262157*delta.std(ddof=1)/np.sqrt(10)
 item={'dataset':d,'set_id':s['set_id'],'sgd':sv.tolist(),'adam':av.tolist(),'sgd_mean':sv.mean(),'adam_mean':av.mean(),'margin':delta.mean(),'paired_95ci':[delta.mean()-half,delta.mean()+half],'sgd_seed_wins':int(sum(delta<0)),'sgd_std':sv.std(ddof=1),'adam_std':av.std(ddof=1)}
 out.append(item)
 print(d,s['set_id'],*[round(v,6) for v in [sv.mean(),av.mean(),delta.mean(),delta.mean()-half,delta.mean()+half]],'wins',sum(delta<0))
print('valid',len(out),'SGD block wins',sum(r['margin']<0 for r in out),'seed wins',sum(r['sgd_seed_wins'] for r in out))
print('distinct expressions',len({r['dataset']['expression'] for r in rows if not r['excluded']}))
json.dump(out,open('analysis_results.json','w'),indent=2)
