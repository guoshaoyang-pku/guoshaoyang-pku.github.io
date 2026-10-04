import json,glob,re,collections,itertools,numpy as np,hashlib,os
def chunks(prefix):
 return [r for p in sorted(glob.glob('research/'+prefix+'_[0-9]*.json')) for r in json.load(open(p))]
h=chunks('history');lab=chunks('lab')
f=next(r for r in h if r['question_id']=='q_85c503')
print('focal epoch/source',f['epoch'],f['source'])
def gate(r):
 return (r.get('dataset',{}).get('rule_family')=='sparse_interaction' and r['dataset'].get('train_size')==1024 and r['budget']['training_steps']==256 and r['optimizer']['type'] in ['Adam','AdamW','RMSprop'] and .01<=r['optimizer']['lr']*256<.1 and not r.get('excluded') and not r.get('failed_seeds'))
def pairs(rows,full=False,twin=False,one=False):
 by=collections.defaultdict(list)
 for r in rows:
  if gate(r):by[r['set_id']].append(r)
 out=[]
 for sid,rs in by.items():
  for s in rs:
   if s.get('activation') not in ['silu','gelu'] or s.get('n_ln',0)<1:continue
   for r in rs:
    if r.get('activation') not in ['relu','leaky_relu'] or s['n_ln']>r['n_ln'] or s['residual']!=r['residual']:continue
    if s['optimizer']['type']!=r['optimizer']['type'] or s['optimizer']['lr']!=r['optimizer']['lr']:continue
    if full and (s['optimizer']!=r['optimizer'] or s['budget']!=r['budget'] or s['loss']!=r['loss'] or s.get('init')!=r.get('init')):continue
    if twin and (s['depth'],s['width'])!=(r['depth'],r['width']):continue
    if one and r['n_ln']-s['n_ln']!=1:continue
    out.append(dict(set_id=sid,dataset_id=s['dataset_id'],smooth=s['candidate_id'],rectifier=r['candidate_id'],ratio=s['mean']/r['mean'],difference=s['mean']-r['mean'],win=s['mean']<r['mean'],smooth_model=s['model'],rectifier_model=r['model'],smooth_optimizer=s['optimizer'],rectifier_optimizer=r['optimizer'],source=s.get('source','base'),cached=[s.get('cached',False),r.get('cached',False)]))
 return out
aud={}
for group,rows in [('base',[r for r in lab if 'source' not in r]),('experiments',[r for r in lab if 'source' in r]),('all',lab)]:
 for label,flags in [('type_lr',(False,False,False)),('full',(True,False,False)),('twin',(True,True,False)),('one_fewer',(True,True,True))]:
  p=pairs(rows,*flags);aud[group+'_'+label]=p
  print(group,label,'wins',sum(x['win'] for x in p),'n',len(p),'sets',len(set(x['set_id'] for x in p)),'datasets',len(set(x['dataset_id'] for x in p)),'median',np.median([x['ratio'] for x in p]) if p else None)
print('BASE SETS',collections.Counter(x['set_id'] for x in aud['base_type_lr']))
json.dump(aud,open('research/lab_audit.json','w'),indent=2)
# Parse the structured natural-language choice fields and optimizer constructor.
def field(q,pattern):
 m=re.search(pattern,q);return m.group(1) if m else None
hist=[]
for rec in h:
 if rec['question_id']=='q_85c503' or rec['epoch']>=f['epoch']:continue
 q=rec['question']
 if 'Rule family: `sparse_interaction`' not in q or '- Train rows: 1024' not in q or '- training_steps: 256' not in q:continue
 cs=[]
 for part in re.split(r'### Choice ',q)[1:]:
  c={'letter':part[0],'activation':field(part,r'- Activation: (\w+)'),'residual':field(part,r'- Residual connections: (\w+)'),'depth':field(part,r'- Depth: (\d+)'),'width':field(part,r'- Width: (\d+)'),'opt':field(part,r'- Optimizer: (\w+)'),'lr':field(part,r'- Learning rate: ([\d.e-]+)'),'constructor':field(part,r'return torch.optim.([^\n]+)')}
  ln=field(part,r'- Layer norm per layer: (\[[^\n]+\])');c['ln']=ln;c['n_ln']=ln.count('True') if ln else 0
  if c['lr']:c['lr']=float(c['lr'])
  cs.append(c)
 for s in cs:
  for r in cs:
   if s['activation'] not in ['silu','gelu'] or r['activation'] not in ['relu','leaky_relu'] or s['n_ln']<1 or s['n_ln']>r['n_ln'] or s['residual']!=r['residual'] or s['opt'] not in ['Adam','AdamW','RMSprop'] or (s['opt'],s['lr'])!=(r['opt'],r['lr']) or not .01<=s['lr']*256<.1:continue
   order=rec['answer'].split('<')
   if len(order)==1 and rec['answer'] not in [s['letter'],r['letter']]:continue
   hist.append(dict(question_id=rec['question_id'],epoch=rec['epoch'],source=rec['source'],smooth=s,rectifier=r,win=(order.index(s['letter'])<order.index(r['letter'])) if len(order)>1 else rec['answer']==s['letter'],full=s['constructor']==r['constructor'],twin=(s['depth'],s['width'])==(r['depth'],r['width']),one_fewer=r['n_ln']-s['n_ln']==1))
json.dump(hist,open('research/history_audit.json','w'),indent=2)
print('HISTORY',json.dumps(hist))
