import json,csv,math,collections,pathlib
ROOT=pathlib.Path(__file__).resolve().parent
def unpack(filename):
 raw=json.loads((ROOT/filename).read_text())
 rows=[]
 for call in raw:
  for c in call.get('content',[]):
   if c.get('type')!='text': continue
   try: d=json.loads(c['text'])
   except ValueError: continue
   for r in d.get('results',[]):
    v=r['variant'];o={k[10:]:val for k,val in v.items() if k.startswith('optimizer.')}
    rows.append(dict(dataset=d['dataset'],dim=d['dataset_params']['input_dim'],depth=v['model.depth'],width=v['model.width'],activation=v['model.activation'],residual=v['model.residual'],ln=''.join(str(int(x)) for x in v['model.layer_norm']),optimizer=o,T=v['budget.training_steps'],batch=v['budget.batch_size'],mean=r.get('mean'),std=r.get('std'),error=r.get('error'),file=filename,variant=v))
 return rows
def proxy(o,T):
 if o['type']=='SGD':return .01*o['lr']*T/(1-o.get('momentum',0))
 if o['type']=='Adagrad':return 2*o['lr']*math.sqrt(T)
 return o['lr']*T
def contrast(p,r):
 diff=p['mean']-r['mean']
 return dict(plain=p['mean'],rival=r['mean'],plain_sd=p['std'],rival_sd=r['std'],difference=diff,ratio=p['mean']/r['mean'],upper_sd_t_halfwidth=2.262*(p['std']+r['std'])/math.sqrt(10))
files=['experiments.json','transport_experiments.json','adjacent_experiments.json','calibration_experiments.json','budget_experiments.json','replication_experiments.json']
rows=[]
for f in files:
 if (ROOT/f).exists():rows+=unpack(f)
for r in rows:r['proxy']=proxy(r['optimizer'],r['T'])
valid=[r for r in rows if not r['error'] and r['mean'] is not None]
summary={'cells':len(rows),'successful_cells':len(valid),'datasets':sorted(set(r['dataset'] for r in rows)),'nominal_seed_fits':10*len(valid),'twin_groups':[],'target_adjacent':[],'budget':[],'calibration':[]}
for dim in [2,4,8]:
 for lr in [3e-5,3e-4]:
  rs=[r for r in valid if r['file'] in files[:2] and r['dim']==dim and r['optimizer']['lr']==lr]
  pairs=[]
  for p in rs:
   if p['residual']:continue
   r=next((r for r in rs if r['residual'] and (r['depth'],r['width'],r['activation'],r['ln'])==(p['depth'],p['width'],p['activation'],p['ln'])),None)
   if r:pairs.append(dict(ln=p['ln'],width=p['width'],activation=p['activation'],**contrast(p,r)))
  for group in ['all','last','early']:
   sub=[p for p in pairs if group=='all' or (p['ln'][-1]=='1')==(group=='last')]
   if not sub:continue
   summary['twin_groups'].append(dict(dim=dim,lr=lr,group=group,pairs=len(sub),wins=sum(p['difference']<0 for p in sub),mean_diff=sum(p['difference'] for p in sub)/len(sub),min_diff=min(p['difference'] for p in sub),max_diff=max(p['difference'] for p in sub),mean_ratio=sum(p['ratio'] for p in sub)/len(sub),pairs_detail=sub))
for file,key in [('adjacent_experiments.json','target_adjacent'),('budget_experiments.json','budget'),('calibration_experiments.json','calibration'),('replication_experiments.json','replication')]:
 summary[key]=[r for r in valid if r['file']==file]
summary['fewer_ln_controls']=[]
for dim in [2,4,8]:
 for lr in [3e-5,3e-4]:
  ps=next(g['pairs_detail'] for g in summary['twin_groups'] if g['dim']==dim and g['lr']==lr and g['group']=='all')
  ds=[]
  for p in ps:
   if p['ln']!='0111':continue
   q=next((q for q in ps if q['ln']=='1111' and q['width']==p['width'] and q['activation']==p['activation']),None)
   if q:ds.append(dict(width=p['width'],activation=p['activation'],difference=p['plain']-q['rival'],ratio=p['plain']/q['rival'],upper_sd_t_halfwidth=2.262*(p['plain_sd']+q['rival_sd'])/math.sqrt(10)))
  summary['fewer_ln_controls'].append(dict(dim=dim,lr=lr,pairs=len(ds),wins=sum(d['difference']<0 for d in ds),details=ds))

(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
with (ROOT/'cells.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['file','dataset','dim','depth','width','activation','residual','ln','optimizer','T','batch','proxy','mean','std','error'])
 w.writeheader()
 for r in rows:w.writerow({k:r[k] for k in w.fieldnames})
summary['unique_variant_specs']=len(set((r['dataset'],json.dumps(r['variant'],sort_keys=True)) for r in valid))
(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
print('UNIQUE_VARIANT_SPECS',summary['unique_variant_specs'])
print('CELLS',summary['cells'],'SUCCESS',summary['successful_cells'],'DATASETS',summary['datasets'],'NOMINAL_SEED_FITS',summary['nominal_seed_fits'])
for s in summary['twin_groups']:print({k:v for k,v in s.items() if k!='pairs_detail'})
for k in ['target_adjacent','calibration','budget','replication']:
 print(k)
 for r in summary[k]:print(r['dataset'],r['T'],r['batch'],r['optimizer'],r['depth'],r['width'],r['activation'],r['residual'],r['ln'],round(r['mean'],6),round(r['std'],6))
