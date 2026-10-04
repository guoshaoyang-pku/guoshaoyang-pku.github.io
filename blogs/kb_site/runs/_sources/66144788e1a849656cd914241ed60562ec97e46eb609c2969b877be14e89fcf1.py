import json,re,itertools,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parent
h=json.load(open(ROOT/'history.json'))
counts=collections.Counter(); errors=[]; controls=[]
for r in h:
 if r['family']!='bigram_lm': continue
 q=r['question']; step=re.search(r'training_steps: (\d+)',q)
 if not step: continue
 T=int(step.group(1))
 choices={}
 for m in re.finditer(r'### Choice ([A-E])\n(.*?)(?=### Choice |## Your answer|\Z)',q,re.S):
  s=m[2];opt=re.search(r'- Optimizer: (\w+)',s);lr=re.search(r'- Learning rate: ([\d.eE+-]+)',s)
  if not opt or not lr: continue
  typ=opt[1];rate=float(lr[1]);mo=re.search(r'- Momentum: ([\d.]+)',s);mom=float(mo[1]) if mo else 0
  delta=2*rate*T**.5 if typ=='Adagrad' else rate*T*.003/(1-mom) if typ=='SGD' else rate*T
  model=s[s.index('**Model (natural language)**'):s.index('**Optimizer**')]
  choices[m[1]]={'optimizer':typ,'delta':delta,'model':model,'lr':rate}
 gold=r['answer'].split('<');pred=r['prediction'].split('<')
 if r['task']=='select':
  pairs=[(gold[0],pred[0])] if gold[0]!=pred[0] else []
 else:
  pairs=[(a,b) for a,b in itertools.combinations(gold,2) if a in pred and b in pred and pred.index(a)>pred.index(b)]
 for a,b in pairs:
  if a not in choices or b not in choices: continue
  ca,cb=choices[a],choices[b]
  direction='higher' if cb['delta']>ca['delta'] else 'lower' if cb['delta']<ca['delta'] else 'equal'
  counts[direction]+=1
  adaptive=(cb['optimizer']!='SGD')-(ca['optimizer']!='SGD')
  counts['adaptive_preference' if adaptive>0 else 'SGD_preference' if adaptive<0 else 'same_optimizer_class']+=1
  errors.append({'question_id':r['question_id'],'task':r['task'],'source':r['source'],'gold_better':a,'pred_better':b,'delta_gold':ca['delta'],'delta_pred':cb['delta'],'direction':direction,'same_model':ca['model']==cb['model'],'adaptive':adaptive})
 for a,b in itertools.combinations(choices,2):
  ca,cb=choices[a],choices[b]
  if ca['model']==cb['model'] and {ca['optimizer'],cb['optimizer']}=={'Adagrad','RMSprop'}:
   if len(gold)>1: winner=a if gold.index(a)<gold.index(b) else b
   elif gold[0] in [a,b]: winner=gold[0]
   else: continue
   controls.append({'question_id':r['question_id'],'choices':[a,b],'winner':winner,'recipes':[ca,cb]})
print('parsed ERROR counts',dict(counts),'questions',len({e['question_id'] for e in errors}))
print('same-model Adagrad/RMS history controls',[(c['question_id'],c['winner'],[(x['optimizer'],x['lr'],x['delta']) for x in c['recipes']]) for c in controls])
json.dump({'counts':dict(counts),'errors':errors,'controls':controls},open(ROOT/'history_audit.json','w'),indent=2)
