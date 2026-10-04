import re,gzip,json,itertools
h=json.load(gzip.open('gru_audit/history.json.gz','rt'));out=[]
for r in h:
 if r.get('family')!='bigram_lm':continue
 s=r['question'];choices=[]
 for c in re.split(r'### Choice ',s)[1:]:
  if 'nn.GRU(' not in c:continue
  def num(p,text=c):
   m=re.search(p,text);return float(m.group(1)) if m else None
  o=re.search(r'- Optimizer: (\w+)',c)
  lr=num(r'- Learning rate: ([\deE.+-]+)')
  t=num(r'training_steps: (\d+)');b=num(r'batch_size: (\d+)')
  if t is None:t=num(r'- training_steps: (\d+)',s)
  if b is None:b=num(r'- batch_size: (\d+)',s)
  plain=('cross_entropy(' in c and 'l1' not in c.lower() and 'l2' not in c.lower())
  if o and lr is not None:choices.append(dict(letter=c[0],o=o.group(1),lr=lr,wd=num(r'- Weight decay: ([\deE.+-]+)'),betas=re.findall(r'- Betas: ([^\n]+)',c),w=num(r'hidden_size=(\d+)'),d=num(r'num_layers=(\d+)'),T=t,b=b,plain=plain))
 order=re.findall(r'\b[A-E]\b',r['answer'])
 for a,b in itertools.combinations(choices,2):
  hi,lo=sorted([a,b],key=lambda x:x['lr'],reverse=True)
  if hi['o'] not in ['Adam','AdamW','RMSprop'] or lo['o'] not in ['Adam','AdamW']:continue
  if hi['lr'] not in [.001,.003] or lo['lr'] not in [.0001,.0003]:continue
  if any(hi[k]!=lo[k] for k in ['w','d','T','b']) or not hi['plain'] or not lo['plain']:continue
  if hi['letter'] in order and lo['letter'] in order:win=order.index(hi['letter'])<order.index(lo['letter'])
  elif len(order)==1 and order[0] in [hi['letter'],lo['letter']]:win=order[0]==hi['letter']
  else:continue
  out.append(dict(q=r['question_id'],high=hi,low=lo,win=win,task=r['task'],source=r['source'],epoch=r['epoch']))
json.dump(out,open('gru_audit/history_pairs.json','w'),indent=2)
for t in [256,512,1024]:
 a=[x for x in out if x['high']['T']==t and x['q']!='q_53593d'];print('HISTORY',t,sum(x['win'] for x in a),len(a),sorted(set(x['q'] for x in a)));print(a)
