import re,json,math,itertools,collections
from pathlib import Path

def value(s,name,default=None):
 m=re.search(r'- '+re.escape(name)+r': ([^\n]+)',s)
 return m.group(1) if m else default

def parse(r):
 q=r['question'];t=int(value(q,'training_steps','0'));bs=int(value(q,'batch_size','0'))
 out=[]
 for label,s in re.findall(r'### Choice ([A-E])(.*?)(?=### Choice|## Your answer|\Z)',q,re.S):
  if value(s,'Type')!='MLP':continue
  try:
   ln=json.loads(value(s,'Layer norm per layer').lower());typ=value(s,'Optimizer');lr=float(value(s,'Learning rate'))
   model=dict(depth=int(value(s,'Depth').split()[0]),width=int(value(s,'Width').split()[0]),residual=value(s,'Residual connections')=='True',layer_norm=ln,activation=value(s,'Activation').split()[0],input_dim=int(value(s,'Input dimension','1' if r['family']=='univariate_regression' else '0')),init=value(s,'Initialization'))
   opt=dict(type=typ,lr=lr,weight_decay=value(s,'Weight decay'),betas=value(s,'Betas'),momentum=value(s,'Momentum'))
   out.append(dict(label=label,family=r['family'],question_id=r['question_id'],model=model,optimizer=opt,budget=dict(training_steps=t,batch_size=bs,total_samples_seen=t*bs),n_ln=sum(ln)))
  except (ValueError,TypeError,AttributeError):continue
 return out

records=[];xp=[];count=collections.Counter()
for r in load_history():
 a=parse(r);order=r['answer'].split('<');rank={v:i for i,v in enumerate(order)}
 if len(order)==1:rank={v:(0 if v==r['answer'] else 1) for v in [x['label'] for x in a]}
 for x in a:
  o=x['optimizer'];t=x['budget']['training_steps'];m=float(o.get('momentum') or 0);g=.03 if x['family'].endswith('regression') and m==.9 else .01
  x['delta']=o['lr']*t if o['type'] in ('Adam','AdamW','RMSprop') else 2*o['lr']*math.sqrt(t) if o['type']=='Adagrad' else o['lr']*t*g/(1-m)
 records.append(dict(question_id=r['question_id'],family=r['family'],epoch=r['epoch'],source=r['source'],answer=r['answer'],score=r['score'],cited=r.get('cited',[]),candidates=a))
 if r['family']!='xor_classification':continue
 for x,y in itertools.combinations(a,2):
  if x['optimizer']!=y['optimizer'] or x['budget']!=y['budget'] or x['model']['residual']==y['model']['residual']:continue
  p,v=(x,y) if not x['model']['residual'] else (y,x)
  if p['model']['input_dim']>=8 or p['model']['depth']<4 or not p['model']['layer_norm'][-1] or p['n_ln']<v['n_ln']:continue
  if rank.get(p['label'],1)==rank.get(v['label'],1):continue
  xp.append(dict(question_id=r['question_id'],plain=p['label'],residual=v['label'],delta=p['delta'],win=rank[p['label']]<rank[v['label']],source=r['source']))
Path('research/history_candidates.json').write_text(json.dumps(records,indent=2))
Path('research/xor_history_pairs.json').write_text(json.dumps(xp,indent=2))
print('parsed',sum(bool(r['candidates']) for r in records),'MLP questions')
for low in [True,False]:
 s=[x for x in xp if (x['delta']<.05)==low];print('XOR below .05' if low else 'XOR >=.05',sum(x['win'] for x in s),len(s),'questions',len(set(x['question_id'] for x in s)))
print('pairs',xp)
