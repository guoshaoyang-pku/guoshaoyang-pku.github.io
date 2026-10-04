import json,gzip,math,itertools,collections
with gzip.open('analysis/lab_snapshot.json.gz','rt') as f:L=json.load(f)
sets={k:[x for x in L if x['set_id']==v] for k,v in {'xor':'exp:0b4125b9ec13','tabular':'exp:afaf9323f634','sgd':'exp:f84504ffe915','spiral':'exp:dd7760207448'}.items()}
def contrast(a,b,label):
 assert a['set_id']==b['set_id']
 d=a['mean']-b['mean'];half=2.262*(a['std']+b['std'])/math.sqrt(10)
 return dict(label=label,set_id=a['set_id'],a=a['candidate_id'],b=b['candidate_id'],difference=d,ratio=a['mean']/b['mean'],halfwidth=half,clear=abs(d)>half,a_mean=a['mean'],b_mean=b['mean'])
out=[]
for topic in ['xor','tabular']:
 cells=sets[topic]
 for a,b in itertools.combinations(cells,2):
  if a['optimizer']!=b['optimizer'] or a['budget']!=b['budget'] or a['residual']!=b['residual'] or a['depth']==b['depth']:continue
  if a['depth']>b['depth']:a,b=b,a
  out.append(contrast(a,b,topic+' shallow '+str(a['depth'])+' vs '+str(b['depth'])+' residual='+str(a['residual'])+' T='+str(a['budget']['training_steps'])+' lr='+str(a['optimizer']['lr'])))
for ln in [False,True]:
 a=next(x for x in sets['sgd'] if x['optimizer']['type']=='SGD' and bool(x['n_ln'])==ln)
 b=next(x for x in sets['sgd'] if x['optimizer']['type']=='Adam' and bool(x['n_ln'])==ln)
 out.append(contrast(a,b,'SGD vs Adam LN='+str(ln)))
for t in [256,2048]:
 cells=[x for x in sets['spiral'] if x['budget']['training_steps']==t]
 a=next(x for x in cells if x['optimizer']['type']=='AdamW' and x['optimizer']['weight_decay']==.001)
 b=next(x for x in cells if x['optimizer']['type']=='Adam' and x['optimizer']['weight_decay']==.001)
 out.append(contrast(a,b,'spiral AdamW vs Adam T='+str(t)))
for x in out:print(x['label'],round(x['difference'],6),'half',round(x['halfwidth'],6),'clear',x['clear'])
with open('analysis/focused_pairs.json','w') as f:json.dump(out,f,indent=2)
with open('analysis/experiment_cells.json','w') as f:json.dump(sets,f,indent=2)
