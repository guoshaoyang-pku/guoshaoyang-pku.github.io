import json,re,itertools,collections,pathlib
R=pathlib.Path(__file__).parent
def parse(q):
    out=[]
    for letter,t in re.findall(r'### Choice ([A-Z])\n(.*?)(?=### Choice|## Your answer|\Z)',q['question'],re.S):
        m=re.findall(r'\*\*Model code\*\*\s*```python\n(.*?)```',t,re.S)
        opt=re.search(r'- Optimizer: (\w+)',t);lr=re.search(r'- Learning rate: ([\deE.+-]+)',t)
        loss=re.findall(r'\*\*Loss\*\*.*?```python\n(.*?)```',t,re.S)
        if m and opt and lr and loss:out.append(dict(letter=letter,model=m[0],opt=opt[1],lr=float(lr[1]),loss=loss[0]))
    return out
hp=[]
for path in (R/'history').glob('*.json'):
    q=json.loads(path.read_text());st=re.search(r'training_steps: (\d+)',q['question']);T=int(st[1]) if st else 0
    order=q['answer'].replace('<answer>','').replace('</answer>','').strip().split('<')
    for a,b in itertools.combinations(parse(q),2):
        if a['model']!=b['model'] or a['loss']!=b['loss'] or 'torch.mean((pred - target) ** 2)' not in a['loss']:continue
        if a['opt']=='RMSprop':a,b=b,a
        if a['opt'] not in ['Adam','AdamW'] or b['opt']!='RMSprop':continue
        if a['letter'] not in order or b['letter'] not in order:continue
        hp.append(dict(qid=q['question_id'],T=T,ratio=a['lr']/b['lr'],win=order.index(a['letter'])<order.index(b['letter'])))
g=collections.defaultdict(list)
for x in json.loads((R/'mvar_lab.json').read_text()):
    if not x.get('excluded') and x.get('failed_seeds',0)==0:g[x['set_id']].append(x)
lp=[]
for sid,rows in g.items():
    for a,b in itertools.combinations(rows,2):
        if a['model']!=b['model'] or a['loss']!=b['loss'] or a['budget']!=b['budget']:continue
        if a['optimizer']['type']=='RMSprop':a,b=b,a
        if a['optimizer']['type'] not in ['Adam','AdamW'] or b['optimizer']['type']!='RMSprop':continue
        lp.append(dict(set_id=sid,ratio=a['optimizer']['lr']/b['optimizer']['lr'],win=a['mean']<b['mean']))
for T in [1024,2048]:
    for mode in ['high','equal']:
        p=[x for x in hp if x['qid']!='q_45fadb' and 256<=x['T']<=T and (x['ratio']>=3-1e-8 if mode=='high' else abs(x['ratio']-1)<1e-8)]
        print('history',T,mode,sum(x['win'] for x in p),len(p),len({x['qid'] for x in p}))
for mode in ['high','equal']:
    p=[x for x in lp if (x['ratio']>=3-1e-8 if mode=='high' else abs(x['ratio']-1)<1e-8)]
    print('lab',mode,sum(x['win'] for x in p),len(p),len({x['set_id'] for x in p}))
