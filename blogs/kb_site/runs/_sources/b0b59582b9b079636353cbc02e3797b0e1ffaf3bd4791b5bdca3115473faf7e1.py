import json,re,runpy
from pathlib import Path
history_protocol=runpy.run_path('analyze.py')['history_protocol']
h=json.loads(Path('research/history_relevant_checkpoint.json').read_text());hp=history_protocol(h);idx={x['question_id']:x for x in h};pairs=[]
for row in hp:
    opt,lr,t=row['opt'],row['lr'],row['T']
    if opt not in ['Adam','AdamW','RMSprop','Adagrad']:continue
    d=2*lr*t**.5 if opt=='Adagrad' else lr*t
    if d<.1:continue
    cs=re.split(r'### Choice ([A-Z])',idx[row['id']]['question'])[1:];acts={}
    for label,body in zip(cs[::2],cs[1::2]):
        z=re.search(r'Activation: ([a-z_]+)',body)
        if z:acts[label]=z.group(1)
    gold=row['answer'].split('<')
    for a,act in acts.items():
        if act not in ['silu','gelu']:continue
        for b,actb in acts.items():
            if actb not in ['relu','leaky_relu']:continue
            win=None
            if len(gold)>1 and a in gold and b in gold:win=gold.index(a)<gold.index(b)
            elif gold==[a]:win=True
            elif gold==[b]:win=False
            if win is not None:pairs.append(dict(id=row['id'],rule=row['rule'],smooth=a,rectifier=b,win=win,opt=opt,lr=lr,T=t,delta=d))
Path('research/history_highband_checkpoint.json').write_text(json.dumps(pairs,indent=2))
for rule in ['piecewise_boundary','smooth_additive','sparse_interaction']:
    z=[x for x in pairs if x['rule']==rule];print(rule,len(z),sum(x['win'] for x in z),len({x['id'] for x in z}))
