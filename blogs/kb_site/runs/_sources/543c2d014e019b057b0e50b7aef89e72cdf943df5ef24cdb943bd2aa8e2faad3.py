import json,re
records=json.load(open('slow_sgd/history_selected.json'))
out=[]
for r in records:
 if r['question_id']=='q_43b634':continue
 q=r['question'];T=int(re.search(r'- training_steps: (\d+)',q).group(1));b=int(re.search(r'- batch_size: (\d+)',q).group(1))
 choices=[]
 for label,s in re.findall(r'### Choice (\w)(.*?)(?=### Choice|## Your answer|\Z)',q,re.S):
  def num(pattern):return float(re.search(pattern,s).group(1))
  typ='G' if 'nn.GRU(' in s else 'TF'
  lr=num(r'- Learning rate: ([0-9.e-]+)');wd=num(r'- Weight decay: ([0-9.e-]+)')
  width=num(r'input_size=(\d+)') if typ=='G' else num(r'd_model=(\d+)')
  depth=num(r'num_layers=(\d+)')
  choices.append({'label':label,'type':typ,'width':int(width),'depth':int(depth),'lr':lr,'wd':wd,'momentum':num(r'- Momentum: ([0-9.e-]+)'),'heads':int(num(r'nhead=(\d+)')) if typ=='TF' else None,'ff':int(num(r'dim_feedforward=(\d+)')) if typ=='TF' else None})
 order=re.findall(r'[A-E]',r['answer'])
 pairs=[]
 for g in choices:
  if g['type']!='G' or g['width']!=64 or g['depth'] not in [1,2]:continue
  for t in choices:
   if t['type']!='TF' or t['width'] not in [32,64] or (g['lr'],g['wd'],g['momentum'])!=(t['lr'],t['wd'],t['momentum']) or not 0<g['lr']*T<=.8:continue
   pairs.append({'g':g,'tf':t,'gru_lower_ordinal':order.index(g['label'])<order.index(t['label']),'exact_b16_wd0':b==16 and g['wd']==0})
 out.append({'question_id':r['question_id'],'source':r['source'],'T':T,'batch':b,'pairs':pairs})
with open('slow_sgd/history_audit.json','w') as f:json.dump(out,f,indent=2)
print([(r['question_id'],len(r['pairs']),sum(p['gru_lower_ordinal'] for p in r['pairs']),sum(p['exact_b16_wd0'] for p in r['pairs']),r['batch']) for r in out])
