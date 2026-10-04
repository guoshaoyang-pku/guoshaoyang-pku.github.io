import json,gzip,collections,itertools,re,os,glob,hashlib,shutil,numpy as np
ROOT=os.path.dirname(os.path.abspath(__file__))
def read(n):
 with gzip.open(os.path.join(ROOT,n+'.json.gz'),'rt') as f:return json.load(f)
lab=read('lab')
pairs=[]
for setid,rs in itertools.groupby(sorted(lab,key=lambda x:x['set_id']),key=lambda x:x['set_id']):
 rs=list(rs)
 for a,b in itertools.combinations(rs,2):
  if any(x.get('excluded') or not np.isfinite(x['mean']) for x in [a,b]):continue
  if any(x.get('model_type')!='gru_lm' or x['loss'].get('loss_id')!='cross_entropy' for x in [a,b]):continue
  if a['model']!=b['model'] or a['budget']!=b['budget']:continue
  hi,lo=(a,b) if a['optimizer']['lr']>b['optimizer']['lr'] else (b,a)
  if hi['optimizer']['type'] not in ['Adam','AdamW','RMSprop'] or lo['optimizer']['type'] not in ['Adam','AdamW']:continue
  if hi['optimizer']['lr'] not in [.001,.003] or lo['optimizer']['lr'] not in [.0001,.0003]:continue
  pairs.append(dict(set_id=setid,dataset=a['dataset_id'],T=a['budget']['training_steps'],model=a['model'],budget=a['budget'],high=hi['optimizer'],low=lo['optimizer'],margin=hi['mean']-lo['mean'],source=hi.get('source','baseline'),ids=[hi['candidate_id'],lo['candidate_id']]))
json.dump(pairs,open(os.path.join(ROOT,'lab_pairs.json'),'w'),indent=2)
for T in [256,512,1024]:
 p=[x for x in pairs if x['T']==T and not x['set_id'].startswith('exp:')]
 print('BASE',T,sum(x['margin']<0 for x in p),len(p),'sets',len(set(x['set_id'] for x in p)))
 print([(x['dataset'],x['model']['d_model'],x['model']['num_layers'],x['high'],x['low'],round(x['margin'],6)) for x in p])
cells=json.load(open(os.path.join(ROOT,'unique_cells.json')))
manifests=json.load(open(os.path.join(ROOT,'manifest.json')))
def seeds(c):return np.array([x['final_test_ce'] for x in sorted(c['summary']['seed_results'],key=lambda x:x['seed'])])
contrasts=[]
for ds in ['bg_1d9caf','bg_209ff2']:
 cs=[c for c in cells if c['dataset']==ds]
 def get(o,lr,t):return next(c for c in cs if c['spec']['optimizer']['type']==o and c['spec']['optimizer']['lr']==lr and c['spec']['budget']['training_steps']==t)
 print('CELLS',ds,[(c['spec']['optimizer']['type'],c['spec']['optimizer']['lr'],c['spec']['budget']['training_steps'],round(c['summary']['mean_test_ce'],6),c['summary']['failed_seeds']) for c in cs])
 for t in [256,512,1024]:
  for a,b in [(('Adam',.003),('RMSprop',.003)),(('Adam',.003),('AdamW',.0003)),(('RMSprop',.003),('AdamW',.0003))]:
   x,y=get(*a,t),get(*b,t);d=seeds(x)-seeds(y);mean=float(d.mean());half=float(2.262157*d.std(ddof=1)/np.sqrt(10))
   z=dict(dataset=ds,T=t,a=a,b=b,margin=mean,interval=[mean-half,mean+half],wins=int((d<0).sum()),ratio=x['summary']['mean_test_ce']/y['summary']['mean_test_ce']);contrasts.append(z);print('TRIO',z)
 for o in ['Adam','RMSprop','AdamW']:
  for hi in [.001,.003]:
   for lo in [.0001,.0003]:
    d=seeds(get(o,hi,256))-seeds(get(o,lo,256));mean=float(d.mean());half=float(2.262157*d.std(ddof=1)/np.sqrt(10));z=dict(dataset=ds,T=256,a=[o,hi],b=[o,lo],margin=mean,interval=[mean-half,mean+half],wins=int((d<0).sum()));contrasts.append(z);print('RATE',z)
json.dump(contrasts,open(os.path.join(ROOT,'contrasts.json'),'w'),indent=2)
print('TOTAL',len(cells),len(manifests),sum(c['summary']['failed_seeds'] for c in cells))
