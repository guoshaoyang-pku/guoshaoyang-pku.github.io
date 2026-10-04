import json,re,collections,numpy as np
from pathlib import Path
P=Path('recipe_audit')
h=json.loads((P/'history.json').read_text()); l=json.loads((P/'lab.json').read_text())
def parse(q):
 ss=re.split(r'### Choice ([A-Z])',q);out=[]
 patterns={'depth':r'- Depth: (\d+)','width':r'- Width: (\d+)','residual':r'- Residual connections: (True|False)','activation':r'- Activation: (\w+)','ln':r'- Layer norm per layer: (\[[^\n]+\])','opt':r'- Optimizer: (\w+)','lr':r'- Learning rate: ([\d.eE+-]+)','steps':r'- training_steps: (\d+)','batch':r'- batch_size: (\d+)'}
 for i in range(1,len(ss),2):
  s=ss[i+1];d={'label':ss[i]}
  for k,p in patterns.items():
   m=re.search(p,s)
   if m:d[k]=m.group(1)
  if len(d)==10:
   for k in ['depth','width','steps','batch']:d[k]=int(d[k])
   d['lr']=float(d['lr']);d['residual']=d['residual']=='True';d['n_ln']=d['ln'].count('True');out.append(d)
 return out
pairs=[]
for x in h:
 if x['family']!='spiral_classification' or x['question_id']=='q_b3f618':continue
 ps=parse(x['question']);m=re.search(r'spiral turns `([\d.]+)`',x['question']);turn=float(m.group(1)) if m else None
 for a in ps:
  for b in ps:
   if a['residual'] and a['activation'] in ['relu','leaky_relu','leakyrelu'] and not b['residual'] and b['activation'] in ['silu','gelu'] and a['opt'] in ['Adam','AdamW'] and b['opt'] in ['Adam','AdamW'] and a['lr']>=3*b['lr']-1e-12 and a['depth']>=b['depth'] and a['n_ln']>=b['n_ln'] and (a['steps'],a['batch'])==(b['steps'],b['batch']) and 256<=a['steps']<=2048:
    gold=x['answer']
    if x['task']=='select':win=True if gold==a['label'] else False if gold==b['label'] else None
    else:
     order=re.findall(r'[A-Z]',gold);win=order.index(a['label'])<order.index(b['label']) if a['label'] in order and b['label'] in order else None
    pairs.append({'id':x['question_id'],'turns':turn,'task':x['task'],'gold':gold,'win':win,'a':a,'b':b,'source':x['source'],'epoch':x['epoch']})
(P/'history_pairs.json').write_text(json.dumps(pairs,indent=2))
print('HISTORY',json.dumps(pairs))
g=collections.defaultdict(list)
for x in l:
 if x['family']=='spiral_classification' and not x['set_id'].startswith('exp:') and not x.get('excluded') and not x.get('failed_seeds'):g[x['set_id']].append(x)
p=[]
for sid,xs in g.items():
 for a in xs:
  for b in xs:
   if a.get('residual') and a.get('activation') in ['relu','leaky_relu','leakyrelu'] and not b.get('residual') and b.get('activation') in ['silu','gelu'] and a['optimizer']['type'] in ['Adam','AdamW'] and b['optimizer']['type'] in ['Adam','AdamW'] and a['optimizer']['lr']>=3*b['optimizer']['lr']-1e-12 and a['depth']>=b['depth'] and a['n_ln']>=b['n_ln'] and a['budget']==b['budget'] and a['loss']==b['loss'] and 256<=a['budget']['training_steps']<=2048:
    p.append({'a':a,'b':b,'ratio':a['mean']/b['mean'],'margin':b['mean']-a['mean']})
(P/'base_pairs.json').write_text(json.dumps(p,indent=2))
print('LAB',[(x['a']['set_id'],x['a']['dataset']['spiral_turns'],x['ratio']) for x in p])
assert len(p)==4 and sum(2<=x['a']['dataset']['spiral_turns']<=3 for x in p)==2
