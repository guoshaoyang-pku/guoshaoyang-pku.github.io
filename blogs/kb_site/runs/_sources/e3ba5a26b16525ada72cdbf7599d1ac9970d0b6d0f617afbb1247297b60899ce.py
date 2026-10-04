import json, math, itertools, collections, os
L=[x for x in load_lab() if x.get('model_type')=='mlp' and not x.get('excluded') and not x.get('failed_seeds') and math.isfinite(x['mean']) and math.isfinite(x['std'])]
H=load_history()
def freeze(x): return json.dumps(x,sort_keys=True,separators=(',',':'))
def delta(x):
 o=x['optimizer'];t=x['budget']['training_steps'];lr=o['lr'];typ=o['type']
 if typ=='Adagrad': return 2*lr*math.sqrt(t)
 if typ=='SGD':
  g=.03 if 'regression' in x['family'] and o.get('momentum',0)==.9 else .01
  return lr*t*g/(1-o.get('momentum',0))
 return lr*t
def region(x):
 f=x['family'];d=delta(x)
 if f=='xor_classification': return 'nf'+('>=8' if x['dataset']['input_dim']>=8 else '<8')+' D'+('>=.05' if d>=.05 else '<.05')
 if f=='synthetic_tabular_classification': return x['dataset']['rule_family']+' E'+str(x['budget']['total_samples_seen']/1024)+' D'+('>=.1' if d>=.1 else '<.1')
 return str(x['dataset'].get('spiral_turns',''))
def summarize(ps):
 if not ps:return {'wins':0,'n':0,'sets':0}
 ratios=[p['ratio'] for p in ps];n=len(ps);w=sum(p['win'] for p in ps);z=1.96;den=1+z*z/n;ctr=(w/n+z*z/(2*n))/den;rad=z*math.sqrt(w/n*(1-w/n)/n+z*z/(4*n*n))/den
 return {'wins':w,'n':n,'sets':len(set(p['set'] for p in ps)),'median_loss_ratio':float(__import__('numpy').median(ratios)),'wilson_descriptive':[round(ctr-rad,3),round(ctr+rad,3)],'clear_mean_gap':sum(p['clear'] for p in ps)}
def pair(a,b,factor):
 return {'set':a['set_id'],'a':a['candidate_id'],'b':b['candidate_id'],'family':a['family'],'region':region(a),'residual':a['residual'],'optimizer':a['optimizer'],'width':a['width'],'depth':a['depth'],'n_ln':a['n_ln'],'activation':a['activation'],'delta':delta(a),'factor':factor,'win':a['mean']<b['mean'],'ratio':a['mean']/b['mean'],'clear':abs(a['mean']-b['mean'])>1.96*math.sqrt((a['std']**2+b['std']**2)/10)}
pairs=[]
for factor in ['activation','layer_norm','depth','width']:
 groups=collections.defaultdict(list)
 for x in L:
  if x['family'] not in ['xor_classification','synthetic_tabular_classification']: continue
  m=dict(x['model']);m.pop(factor,None)
  if factor=='activation':m.pop('leaky_slope',None)
  if factor=='depth':
   ln=m.pop('layer_norm')
   if not (all(ln) or not any(ln)):continue
   m['uniform_ln']=bool(any(ln))
  key=freeze([x['set_id'],x['dataset'],x['budget'],x['optimizer'],x['loss'],x['init'],m])
  groups[key].append(x)
 for xs in groups.values():
  for a,b in itertools.combinations(xs,2):
   if factor=='activation':
    sa=a['activation'] in ['gelu','silu'];sb=b['activation'] in ['gelu','silu']
    if sa==sb:continue
    if not sa:a,b=b,a
   elif factor=='layer_norm':
    if a['n_ln']==b['n_ln']:continue
    if a['n_ln']>b['n_ln']:a,b=b,a
   elif factor=='depth':
    if a['depth']==b['depth']:continue
    if a['depth']>b['depth']:a,b=b,a
   else:
    if a['width']==b['width']:continue
    if a['width']<b['width']:a,b=b,a
   p=pair(a,b,factor);p['gap']=abs(a.get('depth',0)-b.get('depth',0));pairs.append(p)
sgd=[];spiral=[]
sets=collections.defaultdict(list)
for x in L:sets[x['set_id']].append(x)
for xs in sets.values():
 for a,b in itertools.combinations(xs,2):
  if a['optimizer']['type']!='SGD' and b['optimizer']['type']=='SGD':a,b=b,a
  common=all(a[k]==b[k] for k in ['dataset','budget','loss','model','init'])
  if not common:continue
  if a['optimizer']['type']=='SGD' and b['optimizer']['type'] in ['Adam','AdamW','RMSprop','Adagrad'] and a['optimizer'].get('weight_decay',0)==b['optimizer'].get('weight_decay',0) and delta(a)+1e-12>=3*delta(b) and delta(b)<.1:
   p=pair(a,b,'SGD larger delta');p['rival_optimizer']=b['optimizer'];p['rival_delta']=delta(b);sgd.append(p)
  if a['family']=='spiral_classification' and set([a['optimizer']['type'],b['optimizer']['type']])==set(['Adam','AdamW']):
   if a['optimizer']['type']=='Adam':a,b=b,a
   oa=dict(a['optimizer']);ob=dict(b['optimizer']);oa.pop('type');ob.pop('type')
   if oa==ob and oa.get('weight_decay',0)>=.001:
    p=pair(a,b,'AdamW vs Adam');p['turns']=a['dataset']['spiral_turns'];p['means']=[a['mean'],b['mean']];spiral.append(p)
out={}
for p in pairs:
 key=(p['family'],p['region'],p['factor'],p['residual'] if p['factor']=='depth' else 'all')
 out.setdefault(str(key),[]).append(p)
print('ARCHITECTURE EXACT MATCHES')
for k,ps in sorted(out.items()):print(k,summarize(ps))
print('SGD EXACT ARCHITECTURE, DATA/BUDGET/LOSS/WD; lr differs intentionally',summarize(sgd))
for k,ps in itertools.groupby(sorted(sgd,key=lambda p:str((p['family'],p['n_ln']>0,p['depth']>=3,p['residual']))),key=lambda p:str((p['family'],p['n_ln']>0,p['depth']>=3,p['residual']))):print(k,summarize(list(ps)))
print('SPIRAL ACTUAL OPTIMIZER TWINS',summarize(spiral));print(json.dumps(spiral)[:7000])
os.makedirs('analysis',exist_ok=True)
for name,obj in [('strict_arch_pairs',pairs),('strict_sgd_pairs',sgd),('strict_spiral_pairs',spiral),('citation_records',[{k:v for k,v in x.items() if k!='question'} for x in H if any(c in ['K1018','K1009','K1038','K1040'] for c in x.get('cited',[]))])]:
 with open('analysis/'+name+'.json','w') as f:json.dump(obj,f,separators=(',',':'))
