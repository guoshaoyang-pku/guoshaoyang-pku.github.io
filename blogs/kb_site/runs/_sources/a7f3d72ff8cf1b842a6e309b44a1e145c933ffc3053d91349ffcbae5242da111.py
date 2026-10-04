import json, math, csv
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT/name).read_text())
lab=read('lab_mvar.json')
groups=defaultdict(list)
for r in lab:
    if not r['set_id'].startswith('exp:'):
        groups[r['set_id']].append(r)
pairs=[]
for sid, rows in groups.items():
    for a in rows:
        if a['optimizer']['type']!='AdamW' or a['optimizer']['lr']!=.001:
            continue
        for b in rows:
            opt=b['optimizer']
            kind=('adagrad' if opt['type']=='Adagrad' and opt['lr']==.003 else
                  'lowadam' if opt['type']=='AdamW' and opt['lr']==3e-5 else None)
            if kind is None:
                continue
            finite=all(math.isfinite(r['mean']) and math.isfinite(r['std']) for r in (a,b))
            pairs.append(dict(type=kind,set=sid,a=a,b=b,
                same_model=a['model']==b['model'],same_loss=a['loss']==b['loss'],
                same_budget=a['budget']==b['budget'],finite=finite,
                clean=all(not r.get('excluded',False) and r.get('failed_seeds')==0 for r in (a,b)),
                difference=a['mean']-b['mean'] if finite else None,
                ratio=a['mean']/b['mean'] if finite and b['mean'] else None))
out=[]
for p in pairs:
    a,b=p['a'],p['b']
    out.append(dict(type=p['type'],set=p['set'],a=a['candidate_id'],b=b['candidate_id'],
        steps=a['budget']['training_steps'],samples=a['budget']['total_samples_seen'],
        input_dim=a['dataset']['input_dim'],expression=a['dataset']['expression'],
        same_model=p['same_model'],same_loss=p['same_loss'],same_budget=p['same_budget'],
        finite=p['finite'],clean=p['clean'],a_failed=a.get('failed_seeds'),
        b_failed=b.get('failed_seeds'),difference=p['difference'],ratio=p['ratio']))
with (ROOT/'historical_effects.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
results=read('new_results.json')
assert len(results)==12
by_set=defaultdict(list)
for r in results:
    assert r['budget']==dict(training_steps=2048,batch_size=16,total_samples_seen=32768)
    assert r['loss']==dict(loss_id='mse')
    assert r['model']['leaky_relu_slope']==.01
    by_set[r['set_id']].append(r)
effects=[]
# Indices follow preregistered six-variant submission order.
contrasts=[('gold-low',0,1),('gold-adagrad',0,2),('rate999',3,1),
           ('rate95',4,5),('beta-high',4,3),('beta-low',5,1),('decay-high95',0,4)]
for sid, rs in by_set.items():
    assert len(rs)==6 and all(r['model']==rs[0]['model'] for r in rs)
    for name,i,j in contrasts:
        a,b=rs[i],rs[j]
        effects.append(dict(set=sid,dataset=a['dataset_id'],contrast=name,
            a_mean=a['mean'],b_mean=b['mean'],difference=a['mean']-b['mean'],
            ratio=a['mean']/b['mean'],pooled_sd_effect=(a['mean']-b['mean'])/
            math.sqrt((a['std']**2+b['std']**2)/2),
            protocol_n=10,finite_seeds=None,failed_seeds=None,paired_covariance=None))
(ROOT/'effects.json').write_text(json.dumps(effects,indent=2))
with (ROOT/'effects.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(effects[0]));w.writeheader();w.writerows(effects)
for kind in ('adagrad','lowadam'):
    ps=[p for p in pairs if p['type']==kind]
    print(kind,'pairs',len(ps),'sets',len({p['set'] for p in ps}),
          'exact-model',sum(p['same_model'] for p in ps),
          'finite',sum(p['finite'] for p in ps),'clean',sum(p['finite'] and p['clean'] for p in ps))
for e in effects:
    print(e['dataset'],e['contrast'],round(e['difference'],8),round(e['ratio'],5),
          round(e['pooled_sd_effect'],3))
