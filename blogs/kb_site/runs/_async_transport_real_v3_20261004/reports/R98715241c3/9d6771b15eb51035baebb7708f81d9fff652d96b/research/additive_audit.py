import json,itertools
from pathlib import Path
hist={r['question_id']:r for r in load_history()}
parsed=json.loads(Path('research/history_candidates.json').read_text());out=[]
for r in parsed:
 h=hist[r['question_id']]
 if h.get('type')!='architecture_only':continue
 a=r['candidates'];order=h['answer'].split('<')
 ranks={v:i for i,v in enumerate(order)} if len(order)>1 else {x['label']:int(x['label']!=h['answer']) for x in a}
 for x,y in itertools.combinations(a,2):
  if x['optimizer']!=y['optimizer'] or ranks[x['label']]==ranks[y['label']]:continue
  def score(z):
   m=z['model'];return z['n_ln']+int(m['residual'])-int(not m['residual'] and m['depth']>=4 and not m['layer_norm'][-1])
  if score(x)==score(y):continue
  x,y=(x,y) if score(x)>score(y) else (y,x)
  out.append(dict(question_id=r['question_id'],family=r['family'],delta=x['delta'],win=ranks[x['label']]<ranks[y['label']],a=x['label'],b=y['label']))
Path('research/additive_history_pairs.json').write_text(json.dumps(out,indent=2))
for f in sorted({x['family'] for x in out}):
 for low in [True,False]:
  a=[x for x in out if x['family']==f and (x['delta']<.1)==low]
  if a:print(f,'<.1' if low else '>=.1',sum(x['win'] for x in a),len(a),'questions',len({x['question_id'] for x in a}))
