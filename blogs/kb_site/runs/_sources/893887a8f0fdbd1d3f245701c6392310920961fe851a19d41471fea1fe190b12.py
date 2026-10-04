import re,json,math,itertools,collections
def delta(o,T):
 if o['type'] in ['Adam','AdamW','RMSprop']: return o['lr']*T
 if o['type']=='Adagrad': return 2*o['lr']*math.sqrt(T)
 if o['type']=='SGD': return o['lr']*T*.01/(1-o.get('momentum',0))
 raise ValueError(o)
def parse(r):
 q=r['question']
 dim=int(re.search(r'Input dimension: (\d+)',q).group(1))
 T=int(re.search(r'training_steps: (\d+)',q).group(1))
 batch=int(re.search(r'batch_size: (\d+)',q).group(1))
 cs=[]
 for s in re.split(r'### Choice ',q)[1:]:
  def field(n): return re.search(r'^- '+n+r': (.+)',s,re.M).group(1)
  if '- Type: MLP' not in s: continue
  opt={'type':field('Optimizer'),'lr':float(field('Learning rate')),'weight_decay':float(field('Weight decay'))}
  for name,key in [('Momentum','momentum'),('Betas','betas')]:
   m=re.search(r'^- '+name+r': (.+)',s,re.M)
   if m: opt[key]=json.loads(m.group(1)) if key=='betas' else float(m.group(1))
  ln=json.loads(field('Layer norm per layer').lower())
  cs.append(dict(letter=s[0],depth=int(field('Depth').split()[0]),width=int(field('Width').split()[0]),residual=field('Residual connections')=='True',activation=field('Activation').split()[0],layer_norm=ln,n_ln=sum(ln),optimizer=opt,dim=dim,T=T,batch=batch,delta=delta(opt,T)))
 return cs
hs=[];errors=[]
for r in json.load(open('research/xor_history.json')):
 if r['family']!='xor_classification':continue
 try:
  cs=parse(r); order=re.sub('[^A-Z]','',r['answer'].split('<answer>')[-1].split('</answer>')[0])
  if r['task']=='select':
   winner=order[0] if order else r['answer'].strip()
   pairs=[(p,s) for p in cs for s in cs if not p['residual'] and s['residual'] and winner in [p['letter'],s['letter']]]
  else:
   pairs=[(p,s) for p in cs for s in cs if not p['residual'] and s['residual']]
  for p,s in pairs:
   if p['depth']<4 or not p['layer_norm'][-1] or p['dim']>=8:continue
   if p['optimizer']['type']!=s['optimizer']['type'] or p['optimizer']['lr']!=s['optimizer']['lr']:continue
   if r['task']=='select': win=winner==p['letter']
   else:
    if p['letter'] not in order or s['letter'] not in order: continue
    win=order.index(p['letter'])<order.index(s['letter'])
   hs.append(dict(question_id=r['question_id'],source=r['source'],task=r['task'],plain=p,rival=s,win=win,full_optimizer_match=p['optimizer']==s['optimizer'],width_match=p['width']==s['width'],activation_match=p['activation']==s['activation'],depth_match=p['depth']==s['depth'],last_match=s['layer_norm'][-1],ln_relation='ge' if p['n_ln']>=s['n_ln'] else 'lt'))
 except Exception as e:errors.append((r['question_id'],str(e)))
ls=[]
groups=collections.defaultdict(list)
for r in json.load(open('research/xor_lab.json')):
 if r['family']=='xor_classification' and not r.get('excluded') and not r.get('failed_seeds'):groups[r['set_id']].append(r)
for sid,rs in groups.items():
 for p,s in itertools.product(rs,rs):
  if p['residual'] or not s['residual'] or p['depth']<4 or not p['layer_norm'][-1] or p['dataset']['input_dim']>=8:continue
  if p['optimizer']['type']!=s['optimizer']['type'] or p['optimizer']['lr']!=s['optimizer']['lr']:continue
  ls.append(dict(set_id=sid,plain=p['candidate_id'],rival=s['candidate_id'],dim=p['dataset']['input_dim'],delta=delta(p['optimizer'],p['budget']['training_steps']),win=p['mean']<s['mean'],diff=p['mean']-s['mean'],ratio=p['mean']/s['mean'],ln_relation='ge' if p['n_ln']>=s['n_ln'] else 'lt',full_optimizer_match=p['optimizer']==s['optimizer'],width_match=p['width']==s['width'],activation_match=p['activation']==s['activation'],depth_match=p['depth']==s['depth'],last_match=s['layer_norm'][-1]))
with open('research/audit.json','w') as f:json.dump(dict(history=hs,lab=ls,errors=errors),f,indent=2)
print('parse errors',errors)
for source,rows in [('history',hs),('lab',ls)]:
 for regime in ['low','high']:
  for ln in ['ge','lt']:
   sub=[r for r in rows if ((r['plain']['delta'] if source=='history' else r['delta'])<.05)==(regime=='low') and r['ln_relation']==ln]
   print(source,regime,ln,'wins',sum(r['win'] for r in sub),'/',len(sub),'units',len(set(r.get('question_id',r.get('set_id')) for r in sub)))
   if sub:
    print('matching', {key:sum(r[key] for r in sub) for key in ['full_optimizer_match','width_match','activation_match','depth_match','last_match']})
    print('fully architecture matched',sum(all(r[key] for key in ['full_optimizer_match','width_match','activation_match','depth_match','last_match']) for r in sub))
    if source=='lab': print('mean difference',sum(r['diff'] for r in sub)/len(sub),'mean ratio',sum(r['ratio'] for r in sub)/len(sub))
