import re, json, collections

def parse_history(x):
 q=x['question']; rows=[]
 for label,block in re.findall(r'### Choice ([A-Z])\s*(.*?)(?=### Choice |## Your answer|\Z)',q,re.S):
  def field(pattern):
   m=re.search(pattern,block,re.I)
   return m.group(1) if m else None
  d=field(r'- Depth: (\d+)');w=field(r'- Width: (\d+)');r=field(r'- Residual connections: (True|False)');ln=field(r'- Layer norm per layer: (\[[^\]]*\])');o=field(r'- Optimizer: (\w+)');lr=field(r'- Learning rate: ([\deE.+-]+)')
  if None in [d,w,r,ln,o,lr]:continue
  rows.append(dict(label=label,depth=int(d),width=int(w),residual=r=='True',n_ln=ln.count('True'),optimizer=o,lr=float(lr)))
 return rows
HP=[]
for x in load_history():
 if x['family']!='multivariate_regression':continue
 rows=parse_history(x); gold=x['answer']; order=re.findall(r'[A-E]',gold)
 for a in rows:
  for b in rows:
   if a['residual'] or not b['residual'] or a['n_ln']!=b['n_ln']+1 or a['optimizer']!=b['optimizer'] or a['lr']!=b['lr']:continue
   if len(order)==1:
    if order[0] not in [a['label'],b['label']]:continue
    win=order[0]==a['label']
   else:
    if a['label'] not in order or b['label'] not in order:continue
    win=order.index(a['label'])<order.index(b['label'])
   HP.append(dict(question_id=x['question_id'],a=a,b=b,win=win,epoch=x['epoch']))
print('history',len(HP),len({p['question_id'] for p in HP}),sum(p['win'] for p in HP))
for o in sorted({p['a']['optimizer'] for p in HP}):
 ps=[p for p in HP if p['a']['optimizer']==o];print(o,len(ps),len({p['question_id'] for p in ps}),sum(p['win'] for p in ps))
open('history_pairs.json','w').write(json.dumps(HP,indent=2))

import math
H={x['question_id']:x for x in load_history()}
for p in HP:
 q=H[p['question_id']]['question'];t=re.search(r'- training_steps: (\d+)',q);t=int(t.group(1)) if t else 0
 a=p['a'];o=a['optimizer'];lr=a['lr'];m=re.search(r'momentum[=: ]+([0-9.]+)',q);m=float(m.group(1)) if m else 0
 p['delta']=2*lr*math.sqrt(t) if o=='Adagrad' else lr*t*(.03 if m>0 else .01)/(1-m) if o=='SGD' else lr*t
prior=[p for p in HP if p['delta']<.1 and p['question_id']!='q_1f04dc']
print('prior low',len(prior),len({p['question_id'] for p in prior}),sum(p['win'] for p in prior))
open('history_pairs.json','w').write(json.dumps(HP,indent=2))
