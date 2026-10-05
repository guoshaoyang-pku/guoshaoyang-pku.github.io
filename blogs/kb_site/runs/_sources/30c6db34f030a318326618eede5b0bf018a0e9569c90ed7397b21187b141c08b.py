import json, hashlib, pathlib, numpy as np
ROOT=pathlib.Path(__file__).resolve().parent
REPO=ROOT.parent
rows=json.loads((ROOT/'new_rows.json').read_text())
rows.sort(key=lambda r:r['optimizer']['weight_decay'])
STEPS=[1,64,256,1024,2048]
def read(r,key): return json.loads((REPO/r['measurement_files'][key]['repo_path']).read_text())
def ci(v):
 v=np.asarray(v,dtype=float); m=float(v.mean()); sd=float(v.std(ddof=1)); half=2.262157*sd/np.sqrt(len(v))
 return {'mean':m,'sd':sd,'ci95':[m-half,m+half],'n':len(v)}
checks=[]; docs=[]; evals=[]; metrics=[]; trajectories=[]
for r in rows:
 assert r['dataset_id']=='spiralcls_0717d7' and r['cached'] is False
 assert r['source_provenance']['status']=='verified' and r['process_provenance']['status']=='verified'
 assert r['n_seeds']==10 and not r['excluded'] and r['failed_seeds']==0
 for key,f in r['measurement_files'].items():
  b=(REPO/f['repo_path']).read_bytes()
  assert hashlib.sha256(b).hexdigest()==f['sha256'],key
  assert len(b)==f['bytes'],key
  checks.append({'candidate':r['candidate_id'],'artifact':key,'sha256':f['sha256'],'bytes':len(b)})
 spec=read(r,'candidate_spec.json')
 assert spec['model']=={'type':'mlp','depth':3,'width':48,'residual':False,'activation':'silu','layer_norm':[True]*3,'input_dim':2,'output_dim':2}
 assert spec['loss']=={'loss_id':'cross_entropy'}
 assert spec['budget']=={'training_steps':2048,'batch_size':16,'total_samples_seen':32768}
 assert spec['optimizer']['type']=='Adam' and spec['optimizer']['lr']==.001 and spec['optimizer']['betas']==[.9,.95]
 assert spec['process']=={'version':'adam_ce_v1','steps':STEPS,'max_eval_samples':4096}
 assert r['init']=='default' and r['dataset']['train_size']==1024 and r['dataset']['test_size']==2048
 manifest=read(r,'results/execution_manifest.json')
 for key,f in manifest['sources'].items(): assert f['sha256']==r['measurement_files']['executed/'+key]['sha256']
 for key,f in manifest['measurements'].items(): assert f['sha256']==r['measurement_files'][key]['sha256']
 wd=spec['optimizer']['weight_decay']; ds=[]; zs=[]
 for seed in range(10):
  j=read(r,f'results/process/seed_{seed}.json'); ds.append(j)
  assert j['seed']==seed and j['status']=='completed' and j['decay_mode']=='coupled'
  assert j['optimizer_defaults']['decoupled_weight_decay'] is False
  assert j['optimizer_defaults']['weight_decay']==wd
  assert [s['step'] for s in j['steps']]==STEPS
  assert j['inputs']['train_x']['shape']==[1024,2] and j['inputs']['test_x']['shape']==[2048,2]
  z=dict(np.load(REPO/r['measurement_files'][f'results/process/seed_{seed}.npz']['repo_path']));zs.append(z)
  l=z['logits'].astype(np.float64);y=z['targets']; pred=l.argmax(1); margin=l[np.arange(len(y)),y]-l[np.arange(len(y)),1-y]
  ce=np.logaddexp(0,-margin).mean(); err=(pred!=y)
  assert np.array_equal(pred,z['predictions']) and np.array_equal(err,z['errors'])
  assert np.allclose(margin,z['true_class_margins'],rtol=1e-6,atol=1e-6)
  assert abs(ce-j['final_metrics']['ce'])<1e-6
  zs[-1]['margin64']=margin
  metrics.append({'wd':wd,'seed':seed,'ce':float(ce),'accuracy':float((~err).mean()),'errors':int(err.sum()),'error_rate':float(err.mean()),'margin_mean':float(margin.mean()),'margin_quantiles':np.quantile(margin,[0,.05,.25,.5,.75,.95,1]).tolist()})
  for st in j['steps']:
   groups={}
   for e in st['parameters']:
    assert e['active'] and e['adam_step_check']['passed']
    assert abs(e['decay_gradient_norm']-wd*e['parameter_norm'])<=1e-6
    g,d=e['data_gradient_norm'],e['decay_gradient_norm']
    angle=e['data_decay_cosine']['value'] or 0
    assert abs(np.sqrt(max(0,g*g+d*d+2*g*d*angle))-e['coupled_gradient_norm'])<1e-6
    name=e['name']; group='input' if name.startswith('net.0.') else 'head' if name.startswith('net.5.') else '.'.join(name.split('.')[:3])
    groups.setdefault(group,[]).append(e)
   groups['hidden_all']=[e for e in st['parameters'] if e['role']=='hidden']
   groups['all']=st['parameters']
   for group,es in groups.items():
    norm=lambda key:float(np.sqrt(sum(e[key]**2 for e in es)))
    W,G,D,U,Q=map(norm,['parameter_norm','data_gradient_norm','decay_gradient_norm','descent_norm','preconditioned_direction_norm'])
    def cos(key,x,y):
     num=sum((e[key]['value'] or 0)*e[x]*e[y] for e in es)
     den=norm(x)*norm(y)
     return num/den if den else None
    trajectories.append({'wd':wd,'seed':seed,'step':st['step'],'group':group,'W':W,'G':G,'D':D,'r':D/G if G else None,'update':U,'relative_update':U/W if W else None,'direction':Q,'direction_data_cos':cos('preconditioned_data_cosine','preconditioned_direction_norm','data_gradient_norm'),'direction_parameter_cos':cos('preconditioned_parameter_cosine','preconditioned_direction_norm','parameter_norm'),'data_parameter_cos':cos('parameter_data_cosine','parameter_norm','data_gradient_norm')})
 docs.append(ds); evals.append(zs)
pairs=[]
for seed,(a,b) in enumerate(zip(evals[0],evals[1])):
 assert np.array_equal(a['targets'],b['targets']) and np.array_equal(a['indices'],b['indices'])
 assert docs[0][seed]['inputs']==docs[1][seed]['inputs']
 assert docs[0][seed]['minibatch_stream_sha256']==docs[1][seed]['minibatch_stream_sha256']
 for s0,s1 in zip(docs[0][seed]['steps'],docs[1][seed]['steps']): assert s0['minibatch']==s1['minibatch']
 for p0,p1 in zip(docs[0][seed]['steps'][0]['parameters'],docs[1][seed]['steps'][0]['parameters']):
  assert p0['parameter_norm']==p1['parameter_norm'] and p0['data_gradient_norm']==p1['data_gradient_norm']
 m0,m1=a['margin64'],b['margin64'];scale=max(0,float(np.dot(m0,m1)/np.dot(m0,m0)))
 residual=float(np.linalg.norm(m1-scale*m0)/np.linalg.norm(m1));e0=a['errors'];e1=b['errors']
 pairs.append({'seed':seed,'scale':scale,'normalized_residual':residual,'disagreement':int(np.sum(a['predictions']!=b['predictions'])),'new_wrong':int(np.sum(~e0&e1)),'corrected':int(np.sum(e0&~e1)),'both_wrong':int(np.sum(e0&e1)),'delta_ce':metrics[10+seed]['ce']-metrics[seed]['ce'],'delta_error':float(e1.mean()-e0.mean()),'delta_margin_mean':float(m1.mean()-m0.mean())})
summary={'cells':{},'paired':{},'process':{},'verification':{'artifacts':len(checks),'adam_step_checks':1600,'paired_streams_and_inputs':10,'recipe':'verified'}}
for wd in [0,.001]: summary['cells'][str(wd)]={key:ci([m[key] for m in metrics if m['wd']==wd]) for key in ['ce','accuracy','errors','error_rate','margin_mean']}
for key in pairs[0]:
 if key!='seed': summary['paired'][key]=ci([p[key] for p in pairs])
summary['paired']['positive_ce_seeds']=sum(p['delta_ce']>0 for p in pairs)
summary['paired']['ratio_of_ce_means']=summary['cells']['0.001']['ce']['mean']/summary['cells']['0']['ce']['mean']
for wd in [0,.001]:
 for group in sorted(set(t['group'] for t in trajectories)):
  for step in STEPS:
   vals=[t for t in trajectories if t['wd']==wd and t['group']==group and t['step']==step]
   summary['process'][f'{wd}/{group}/{step}']={k:ci([t[k] for t in vals]) for k in ['W','G','D','r','update','relative_update','direction','direction_data_cos','direction_parameter_cos','data_parameter_cos'] if all(t[k] is not None for t in vals)}
for name,data in [('hash_verification',checks),('seed_metrics',metrics),('paired_metrics',pairs),('role_trajectories',trajectories),('summary',summary)]:
 (ROOT/(name+'.json')).write_text(json.dumps(data,indent=2))
print(json.dumps({'verification':summary['verification'],'cells':summary['cells'],'paired':summary['paired']},indent=2))
