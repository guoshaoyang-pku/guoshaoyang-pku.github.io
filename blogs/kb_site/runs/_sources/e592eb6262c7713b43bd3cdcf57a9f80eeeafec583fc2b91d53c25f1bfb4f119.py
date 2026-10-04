import json,gzip,re,itertools
z=json.load(gzip.open('xor_audit/frozen.json.gz','rt'))
def get(pattern,s,cast=str,default=None):
 m=re.search(pattern,s);return cast(m.group(1)) if m else default
records=[];pairs=[]
for r in z['history']:
 if r['family']!='xor_classification' or r['question_id']=='q_b204a3':continue
 s=r['question'];dim=get(r'- Input dimension: (\d+)',s,int)
 if dim!=2:continue
 steps=get(r'- training_steps: (\d+)',s,int);bs=get(r'- batch_size: (\d+)',s,int)
 chunks=re.split(r'### Choice ([A-Z])\n',s);cs=[]
 for i in range(1,len(chunks),2):
  letter,t=chunks[i:i+2]
  opt=get(r'- Optimizer: (\w+)',t);lr=get(r'- Learning rate: ([\d.e-]+)',t,float);mom=get(r'- Momentum: ([\d.e-]+)',t,float,0)
  if opt!='SGD' or lr!=.0003 or mom!=0 or steps!=256:continue
  mask=get(r'- Layer norm per layer: (\[[^\n]+\])',t)
  if mask is None:continue
  mask=json.loads(mask.lower())
  cs.append({'letter':letter,'optimizer':(opt,lr,mom,get(r'- Weight decay: ([\d.e-]+)',t,float)), 'budget':(steps,bs),'depth':get(r'- Depth: (\d+)',t,int),'width':get(r'- Width: (\d+)',t,int),'activation':get(r'- Activation: (\w+)',t),'residual':get(r'- Residual connections: (True|False)',t)=='True','mask':mask,'init':'default' if 'Initialization: PyTorch Linear defaults' in t else 'unknown'})
 if cs:records.append({'question_id':r['question_id'],'candidates':cs,'ordinal':r['answer']})
 rank={a:i for i,a in enumerate(re.findall(r'[A-E]',r['answer']))}
 for a,b in itertools.combinations(cs,2):
  full=a['optimizer']==b['optimizer'] and a['budget']==b['budget']
  eq=lambda fs:all(a[f]==b[f] for f in fs)
  for p in ['P1','P2','P3','P4','P5']:
   f=g=None;iso=False
   if p=='P1' and (a['activation'] in ['relu','leaky_relu'])!=(b['activation'] in ['relu','leaky_relu']) and {a['activation'],b['activation']}&{'silu','gelu'}:
    f,g=(a,b) if a['activation'] in ['relu','leaky_relu'] else (b,a);iso=eq(['depth','width','residual','mask','init'])
   if p=='P2' and a['residual']!=b['residual']:
    f,g=(a,b) if a['residual'] else (b,a);iso=eq(['depth','width','activation','mask','init'])
   if p=='P3' and a['activation']==b['activation'] and a['activation'] in ['silu','gelu'] and {sum(a['mask']),sum(b['mask'])}=={0,1}:
    f,g=(a,b) if sum(a['mask']) else (b,a)
    if not f['mask'][-1]:continue
    iso=eq(['depth','width','residual','init'])
   if p=='P4' and a['activation']==b['activation'] and a['activation'] in ['silu','gelu'] and not a['residual'] and not b['residual']:
    f,g=sorted([a,b],key=lambda x:x['depth'])
    if not(f['depth']==2 and sum(f['mask'])==1 and f['mask'][-1] and g['depth']==4 and all(g['mask'])):continue
    iso=eq(['width','init'])
   if p=='P5' and a['activation']==b['activation'] and a['activation'] in ['silu','gelu']:
    anchors=[x for x in [a,b] if x['depth']==4 and not any(x['mask'])]
    if not anchors:continue
    g=anchors[0];f=b if g is a else a;iso=eq(['width','init'])
   if f is not None and f['letter'] in rank and g['letter'] in rank:
    pairs.append({'prediction':p,'question_id':r['question_id'],'first':f['letter'],'second':g['letter'],'win':rank[f['letter']]<rank[g['letter']],'full_optimizer_budget':full,'isolated':iso})
json.dump({'records':records,'pairs':pairs},open('xor_audit/backtest_history.json','w'),indent=2)
print('eligible records',len(records),'multi eligible',sum(len(x['candidates'])>=2 for x in records),'candidates',sum(len(x['candidates']) for x in records))
for p in ['P1','P2','P3','P4','P5']:
 for name,fn in [('bundled_type_lr',lambda x:True),('full_optimizer_budget',lambda x:x['full_optimizer_budget']),('isolated_full',lambda x:x['full_optimizer_budget'] and x['isolated'])]:
  q=[x for x in pairs if x['prediction']==p and fn(x)]
  print(p,name,sum(x['win'] for x in q),len(q),'records',len({x['question_id'] for x in q}),'candidates',len({(x['question_id'],c) for x in q for c in [x['first'],x['second']]}))
