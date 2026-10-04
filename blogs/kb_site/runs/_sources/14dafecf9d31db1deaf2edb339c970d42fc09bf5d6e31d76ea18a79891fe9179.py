import json,re,itertools,math,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def stable(x):
 return json.dumps(x,sort_keys=True)
def history_pairs(h):
 pairs=[]; rejected=collections.Counter()
 for r in h:
  if r['family']!='multivariate_regression':continue
  chunks=re.split(r'### Choice ([A-Z])\s*\n',r['question'])
  shared=r['question'].split('## Choices')[0]
  choices=[]
  for i in range(1,len(chunks),2):
   label,body=chunks[i:i+2]
   model=re.search(r'\*\*Model code\*\*\s*\n```python\n(.*?)\n```',body,re.S)
   loss=re.search(r'def loss_fn\(.*?\n(.*?)\n```',body,re.S)
   opt=re.search(r'- Optimizer: (\w+)',body)
   lr=re.search(r'- Learning rate: ([\deE.+-]+)',body)
   schedule=body if '- training_steps:' in body else shared
   vals=[re.search('- '+key+r': (\d+)',schedule) for key in ['training_steps','batch_size','total_samples_seen']]
   init=re.search(r'- Initialization: ([^\n]+)',body)
   if not all([model,loss,opt,lr,init,*vals]):
    rejected['unparsed_choices']+=1;continue
   choices.append(dict(label=label,model=model[1].strip(),init=init[1],loss=loss[1].strip(),
     optimizer=opt[1],lr=float(lr[1]),budget=tuple(int(x[1]) for x in vals)))
  for a,b in itertools.combinations(choices,2):
   if b['optimizer']=='Adagrad':a,b=b,a
   if a['optimizer']!='Adagrad' or b['optimizer'] not in ['Adam','AdamW']:continue
   if a['budget']!=b['budget']:rejected['budget']+=1;continue
   if a['loss']!=b['loss']:rejected['loss']+=1;continue
   if 'return torch.mean((pred - target) ** 2)' not in a['loss']:
    rejected['nonplain_mse']+=1;continue
   if a['model']!=b['model'] or a['init']!=b['init']:rejected['model']+=1;continue
   order=r['answer'].split('<')
   if len(order)>1 and a['label'] in order and b['label'] in order:win=order.index(a['label'])<order.index(b['label'])
   elif r['answer']==a['label']:win=True
   elif r['answer']==b['label']:win=False
   else:rejected['unobserved_select_pair']+=1;continue
   pairs.append(dict(question_id=r['question_id'],epoch=r['epoch'],source=r['source'],
    imported_from=r.get('imported_from'),adagrad=a,adam=b,adagrad_win=win))
 return pairs,dict(rejected)
def lab_pairs(rows):
 groups=collections.defaultdict(list)
 for r in rows:
  if not r['set_id'].startswith('exp:'):groups[r['set_id']].append(r)
 pairs=[]
 for sid,group in groups.items():
  for a,b in itertools.combinations(group,2):
   if b['optimizer']['type']=='Adagrad':a,b=b,a
   if a['optimizer']['type']!='Adagrad' or b['optimizer']['type'] not in ['Adam','AdamW']:continue
   if a['budget']!=b['budget'] or a['loss']!=b['loss']:continue
   pairs.append(dict(set_id=sid,a=a,b=b,same_model=a['model']==b['model'] and a.get('init')==b.get('init'),
    finite=math.isfinite(a['mean']) and math.isfinite(b['mean']),
    not_excluded=not a.get('excluded') and not b.get('excluded'),adagrad_win=a['mean']<b['mean']))
 return pairs
def summarize(p,lab=False):
 key='set_id' if lab else 'question_id'
 return dict(pairs=len(p),sets=len(set(r[key] for r in p)),adagrad_wins=sum(r['adagrad_win'] for r in p))
def analyze():
 h=json.loads((ROOT/'history.json').read_text());m=json.loads((ROOT/'lab_mvar.json').read_text())
 hp,rejected=history_pairs(h)
 out={'history':summarize(hp),'history_rejected':rejected,'history_pairs':hp,
      'history_before_q_epoch':summarize([r for r in hp if r['epoch']<3])}
 lp=lab_pairs(m)
 out['lab']={}
 for rates in ['all','conservative_vs_fast']:
  subset=lp if rates=='all' else [r for r in lp if r['a']['optimizer']['lr']<=.0001 and r['b']['optimizer']['lr']>=.001]
  for policy in ['all','finite','not_excluded']:
   p=[r for r in subset if policy=='all' or r[policy]]
   out['lab'][rates+'/'+policy]=dict(**summarize(p,True),same_model=summarize([r for r in p if r['same_model']],True),
    budgets=sorted(set(tuple(r['a']['budget'][k] for k in ['training_steps','batch_size','total_samples_seen']) for r in p)))
 out['rate_ratio_le3_pre_epoch3']=summarize([r for r in hp if r['epoch']<3 and r['adagrad']['lr']/r['adam']['lr']<=3])
 out['conservative_fast_history']=summarize([r for r in hp if r['adagrad']['lr']<=.0001 and r['adam']['lr']>=.001])
 records={r['question_id']:r for r in h};bias=collections.Counter();errors=[]
 for r in hp:
  a,b=r['adagrad'],r['adam'];pred=records[r['question_id']]['prediction'];order=pred.split('<')
  if len(order)>1 and a['label'] in order and b['label'] in order:pw=order.index(a['label'])<order.index(b['label'])
  elif pred==a['label']:pw=True
  elif pred==b['label']:pw=False
  else:bias['unidentifiable_prediction']+=1;continue
  da=2*a['lr']*a['budget'][0]**.5;db=b['lr']*b['budget'][0]
  if da==db:bias['equal_delta']+=1;continue
  if pw==r['adagrad_win']:bias['correct']+=1;continue
  kind='high_delta_error' if (da>db)==pw else 'low_delta_error';bias[kind]+=1;errors.append((r['question_id'],kind))
 out['restricted_prediction_bias']={'counts':dict(bias),'errors':errors}
 (ROOT/'audit_results.json').write_text(json.dumps(out,indent=2))
 print(json.dumps({k:v for k,v in out.items() if k!='history_pairs'},indent=2))
 print('historical pairs',[(r['question_id'],r['adagrad']['label'],r['adam']['label'],r['adagrad_win'],r['adagrad']['budget']) for r in hp])
if __name__=='__main__':analyze()
