import json,math,itertools,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def extract(name):
 raw=json.loads((ROOT/(name+'.json')).read_text())
 return json.loads(next(c['text'] for c in raw['content'] if c['type']=='text'))
def analyze():
 rows=[];pairs=[]
 for name,sid in [('experiment1','exp:25e340c554d5'),('experiment2','exp:f431df8b43e3')]:
  data=extract(name)
  for r in data['results']:
   v=r['variant'];op=v['optimizer.type'];lr=v['optimizer.lr'];t=v['budget.training_steps']
   rows.append(dict(set_id=sid,dataset=data['dataset'],optimizer=op,lr=lr,weight_decay=v['optimizer.weight_decay'],
    T=t,batch=32,samples=t*32,mean=r['mean'],std=r['std'],seed_count_protocol=10,
    finite_seed_count=None,failed_seed_count=None,error=r['error'],
    finite_aggregate=math.isfinite(r['mean']) and math.isfinite(r['std'])))
  for t in [256,1024]:
   group=[r for r in rows if r['set_id']==sid and r['T']==t]
   for a,b in itertools.combinations(group,2):
    pairs.append(dict(set_id=sid,T=t,a=f"{a['optimizer']}@{a['lr']}",b=f"{b['optimizer']}@{b['lr']}",
      a_minus_b=a['mean']-b['mean'],a_over_b=a['mean']/b['mean'],a_wins=a['mean']<b['mean']))
 (ROOT/'results.json').write_text(json.dumps(rows,indent=2))
 (ROOT/'contrasts.json').write_text(json.dumps(pairs,indent=2))
 with (ROOT/'results.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 for r in rows:print(f"{r['dataset'].split('/')[-1]} T{r['T']} {r['optimizer']} {r['lr']} {r['mean']:.8f} +/-{r['std']:.8f}")
 for p in pairs:
  if p['a']=='Adagrad@0.0001' and p['b']=='AdamW@0.001' or p['a']=='AdamW@0.001' and p['b'] in ['Adagrad@0.001','AdamW@0.0001','SGD@0.003']:
   print(p)
 print('finite aggregate cells',sum(r['finite_aggregate'] for r in rows),'reported cell errors',sum(r['error'] is not None for r in rows))
if __name__=='__main__':analyze()
