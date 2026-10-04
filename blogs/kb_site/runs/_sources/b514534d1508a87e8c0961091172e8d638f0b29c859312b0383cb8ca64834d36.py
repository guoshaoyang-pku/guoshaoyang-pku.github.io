import collections, itertools, json
from pathlib import Path
import numpy as np
import runpy
history_protocol=runpy.run_path('analyze.py')['history_protocol']

lab=json.loads(Path('research/tabular_lab_initial.json').read_text())
base=[x for x in lab if '/set_' in x['set_id'] and x.get('model_type')=='mlp']
pairs=[]
for sid,xs in itertools.groupby(sorted(base,key=lambda x:x['set_id']),key=lambda x:x['set_id']):
    for a,b in itertools.combinations(list(xs),2):
        if any(a['optimizer'][k]!=b['optimizer'][k] for k in ['type','lr']): continue
        if a['budget']['training_steps']!=b['budget']['training_steps']: continue
        opt=a['optimizer']['type']; lr=a['optimizer']['lr']; t=a['budget']['training_steps']
        if opt not in ['Adam','AdamW','RMSprop','Adagrad']: continue
        d=2*lr*t**.5 if opt=='Adagrad' else lr*t
        if d<.1 or (a['activation'] in ['silu','gelu'])==(b['activation'] in ['silu','gelu']): continue
        if a['excluded'] or b['excluded']: continue
        s,r=(a,b) if a['activation'] in ['silu','gelu'] else (b,a)
        pairs.append(dict(set=sid,rule=a['dataset']['rule_family'],smooth=s['candidate_id'],rectifier=r['candidate_id'],win=s['mean']<r['mean'],ratio=s['mean']/r['mean'],opt=opt,lr=lr,T=t,delta=d))
Path('research/base_highband_pairs.json').write_text(json.dumps(pairs,indent=2))
for rule in ['piecewise_boundary','smooth_additive','sparse_interaction']:
    rows=[p for p in pairs if p['rule']==rule]
    print(rule, len(rows),sum(p['win'] for p in rows),len({p['set'] for p in rows}),np.median([p['ratio'] for p in rows]))
for t in [256,1024]:
    rows=[x for x in base if x['optimizer']['type']=='RMSprop' and x['optimizer']['lr']==.0001 and x['budget']['training_steps']==t]
    print('RMSprop .0001',t,len(rows),dict(collections.Counter(x['dataset']['rule_family'] for x in rows)))
print('Relevant checkpoint protocols',history_protocol(json.loads(Path('research/history_relevant_checkpoint.json').read_text())))
