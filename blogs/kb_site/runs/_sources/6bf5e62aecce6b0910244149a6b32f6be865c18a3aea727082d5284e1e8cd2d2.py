
import re,json,itertools,collections,numpy as np
def extract(r):
 q=r['question']; choices=[]
 def get(p,s,cast=str,default=None):
  m=re.search(p,s);return cast(m.group(1)) if m else default
 T=get(r'training_steps: (\d+)',q,int);bs=get(r'batch_size: (\d+)',q,int)
 for letter,s in re.findall(r'### Choice ([A-Z])\n(.*?)(?=### Choice|## Your answer|\Z)',q,re.S):
  ln=get(r'Layer norm per layer: (\[[^\]]*\])',s,lambda v:json.loads(v.lower()))
  if ln is None:continue
  c=dict(letter=letter,depth=get(r'Depth: (\d+)',s,int),width=get(r'Width: (\d+)',s,int),residual=get(r'Residual connections: (True|False)',s,lambda v:v=='True'),layer_norm=ln,n_ln=sum(ln),activation=get(r'Activation: (\w+)',s),init=get(r'Initialization: ([^\n]+)',s),optimizer=dict(type=get(r'Optimizer: (\w+)',s),lr=get(r'Learning rate: ([\de.+-]+)',s,float),weight_decay=get(r'Weight decay: ([\de.+-]+)',s,float)),budget=dict(training_steps=T,batch_size=bs))
  c['optimizer']['betas']=get(r'Betas: (\[[^\]]+\])',s,json.loads)
  c['optimizer']['momentum']=get(r'Momentum: ([\de.+-]+)',s,float,0)
  choices.append(c)
 return dict(question_id=r['question_id'],rule=get(r'Rule family: `(\w+)`',q),answer=r['answer'],choices=choices)
def delta(x,mode='core'):
 o=x['optimizer'];T=x['budget']['training_steps'];lr=o['lr']
 if o['type']=='SGD':return lr*T*.01/(1-o.get('momentum',0))
 if o['type']=='Adagrad' and mode=='core':return 2*lr*T**.5
 return lr*T
def matched(a,b,mode='core'):
 return a['optimizer']['type']==b['optimizer']['type'] and a['optimizer']['lr']==b['optimizer']['lr'] and a['budget']==b['budget'] and .03<=delta(a,mode)<=.1
def order(answer,a,b):
 ans=re.sub(r'<[^>]*>','',answer).strip();ls=ans.split('<')
 if len(ls)>1 and a in ls and b in ls:return ls.index(a)<ls.index(b)
 if ans==a:return True
 if ans==b:return False
 return None
def audit(history,lab,mode='core'):
 hp=[];lp=[];hz=[];lz=[];equal=[]
 for r in history:
  if r['family']!='synthetic_tabular_classification':continue
  e=extract(r)
  for a,b in itertools.combinations(e['choices'],2):
   if not matched(a,b,mode):continue
   if a['depth']!=b['depth'] and a['residual'] and b['residual'] and a['n_ln']>0 and b['n_ln']>0:
    a,b=sorted([a,b],key=lambda x:x['depth']);win=order(e['answer'],b['letter'],a['letter'])
    if win is not None:hp.append(dict(q=e['question_id'],rule=e['rule'],shallow=a,deep=b,deeper_win=win))
   for rect,smooth in [(a,b),(b,a)]:
    if e['rule']=='sparse_interaction' and rect['activation'] in ['relu','leaky_relu'] and smooth['activation'] in ['silu','gelu'] and not rect['residual'] and smooth['residual'] and rect['n_ln']==smooth['n_ln']==0:
     win=order(e['answer'],rect['letter'],smooth['letter'])
     if win is not None:hz.append(dict(q=e['question_id'],rect=rect,smooth=smooth,rect_win=win))
 g=collections.defaultdict(list)
 for x in lab:
  if x['family']=='synthetic_tabular_classification' and not x.get('excluded') and x['loss']['loss_id']=='cross_entropy':g[x['set_id']].append(x)
 for sid,rows in g.items():
  for a,b in itertools.combinations(rows,2):
   if not matched(a,b,mode):continue
   if a['depth']!=b['depth'] and a['residual'] and b['residual'] and a['n_ln']>0 and b['n_ln']>0:
    a,b=sorted([a,b],key=lambda x:x['depth']);lp.append(dict(set=sid,shallow=a,deep=b,deeper_win=b['mean']<a['mean'],ratio=b['mean']/a['mean']))
   for rect,smooth in [(a,b),(b,a)]:
    if rect['activation'] in ['relu','leaky_relu'] and smooth['activation'] in ['silu','gelu'] and rect['n_ln']==smooth['n_ln']:
     z=dict(set=sid,rect=rect,smooth=smooth,rect_win=rect['mean']<smooth['mean'],ratio=rect['mean']/smooth['mean']);equal.append(z)
     if rect['dataset']['rule_family']=='sparse_interaction' and not rect['residual'] and smooth['residual'] and rect['n_ln']==0:lz.append(z)
 return dict(history_depth=hp,lab_depth=lp,history_zero=hz,lab_zero=lz,lab_equal=equal)
