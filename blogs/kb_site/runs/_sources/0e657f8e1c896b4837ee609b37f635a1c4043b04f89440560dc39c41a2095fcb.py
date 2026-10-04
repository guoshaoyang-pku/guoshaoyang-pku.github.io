import ast, re, json, math, itertools, glob
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parent

def canon(code):
    return ast.dump(ast.parse(code),include_attributes=False)

def parse(r):
    q=r['question']; out=[]
    budget=tuple(re.findall(r'- (training_steps|batch_size|total_samples_seen): (\d+)',q.split('## Choices')[0]))
    for letter,section in re.findall(r'### Choice ([A-E])\n(.*?)(?=### Choice |## Your answer|\Z)',q,re.S):
        bd=dict(budget); bd.update(dict(re.findall(r'- (training_steps|batch_size|total_samples_seen): (\d+)',section))); local_budget=tuple(sorted(bd.items()))
        blocks=re.findall(r'```python\n(.*?)```',section,re.S)
        if len(blocks)!=3 or 'torch.optim.Adagrad(' not in blocks[1]: continue
        opt=ast.parse(blocks[1]); calls=[n for n in ast.walk(opt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Adagrad']
        if len(calls)!=1: continue
        kws={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
        wd=kws.get('weight_decay',0); lr=kws['lr']
        for k in calls[0].keywords:
            if k.arg=='weight_decay': k.value=ast.Constant(0)
        out.append(dict(letter=letter,wd=wd,lr=lr,key=(canon(blocks[0]),ast.dump(opt,include_attributes=False),canon(blocks[2]),local_budget),default=not bool(re.search(r'nn\.init\.|reset_parameters|manual_seed',blocks[0]))))
    return out,budget

h=[json.load(open(p)) for p in sorted(ROOT.glob('history/*.json'))]; pairs=[]; errors=[]
for r in h:
    try: choices,budget=parse(r)
    except Exception as e: errors.append([r['question_id'],str(e)]); continue
    for a,b in itertools.combinations(choices,2):
        if a['key']!=b['key'] or a['wd']==b['wd']: continue
        a,b=sorted([a,b],key=lambda x:x['wd'])
        ans=re.findall(r'[A-E]',r['answer'])
        if r['task']=='ranking' and a['letter'] in ans and b['letter'] in ans: win=ans.index(a['letter'])<ans.index(b['letter'])
        elif r['task']=='select' and len(ans)==1 and ans[0] in [a['letter'],b['letter']]: win=ans[0]==a['letter']
        else: continue
        bd=dict(a['key'][-1]); T=int(bd.get('training_steps',0)); proxy=2*a['lr']*math.sqrt(T)
        pairs.append(dict(question_id=r['question_id'],epoch=r['epoch'],source=r['source'],family=r['family'],low=a['letter'],high=b['letter'],wd=[a['wd'],b['wd']],lr=a['lr'],T=T,batch=int(bd.get('batch_size',0)),proxy=proxy,default=a['default'] and b['default'],win=win))
json.dump(dict(pairs=pairs,parse_errors=errors),open(ROOT/'history_pairs.json','w'),indent=2)
for name,ps in [('all',pairs),('excluding focal',[p for p in pairs if p['question_id']!='q_f03f18']),('prior low proxy',[p for p in pairs if p['epoch']<14 and p['T']<=1024 and p['proxy']<=.01 and p['default'] and 0<=p['wd'][0]<p['wd'][1]<=.001])]:
    print(name,'wins',sum(p['win'] for p in ps),'pairs',len(ps),'questions',len({p['question_id'] for p in ps}));print(ps)
print('parse errors',errors)
lab=json.load(open(ROOT/'lab_adagrad_prior.json')); lp=[]
for a,b in itertools.combinations(lab,2):
    if a['set_id']!=b['set_id']: continue
    oa={k:v for k,v in a['optimizer'].items() if k!='weight_decay'}; ob={k:v for k,v in b['optimizer'].items() if k!='weight_decay'}
    if oa!=ob or any(a.get(k)!=b.get(k) for k in ['model','init','loss','budget']): continue
    if a['optimizer'].get('weight_decay',0)==b['optimizer'].get('weight_decay',0): continue
    a,b=sorted([a,b],key=lambda x:x['optimizer'].get('weight_decay',0))
    lp.append(dict(set_id=a['set_id'],dataset_id=a['dataset_id'],family=a['family'],candidates=[a['candidate_id'],b['candidate_id']],lr=a['optimizer']['lr'],wd=[a['optimizer'].get('weight_decay',0),b['optimizer'].get('weight_decay',0)],T=a['budget']['training_steps'],batch=a['budget']['batch_size'],model=a['model'],init=a.get('init'),loss=a['loss'],mean=[a['mean'],b['mean']],std=[a['std'],b['std']],failed=[a.get('failed_seeds'),b.get('failed_seeds')],excluded=[a.get('excluded'),b.get('excluded')],win=a['mean']<b['mean']))
json.dump(lp,open(ROOT/'lab_pairs.json','w'),indent=2)
print('lab wins/pairs/sets/datasets',sum(p['win'] for p in lp),len(lp),len({p['set_id'] for p in lp}),len({p['dataset_id'] for p in lp})); print(lp)
ex=json.load(open(ROOT/'exact_experiments.json')); results=[]
for ds in sorted({r['dataset_id'] for r in ex}):
    a,b=sorted([r for r in ex if r['dataset_id']==ds],key=lambda x:x['optimizer']['weight_decay'])
    assert a['set_id']==b['set_id']
    assert all(a[k]==b[k] for k in ['model','loss','budget','init'])
    sa={r['seed']:r for r in a['seed_results']}; sb={r['seed']:r for r in b['seed_results']}
    assert sa.keys()==sb.keys() and len(sa)==10
    assert not any(r['failed'] for r in list(sa.values())+list(sb.values()))
    d=np.array([sb[s]['final_test_ce']-sa[s]['final_test_ce'] for s in sorted(sa)])
    se=float(d.std(ddof=1)/np.sqrt(len(d))); mean=float(d.mean()); ci=[mean-2.2621571627409915*se,mean+2.2621571627409915*se]
    res=dict(dataset=ds,set_id=a['set_id'],mean=[a['mean'],b['mean']],sd=[a['std'],b['std']],margin=mean,ci95=ci,paired_sd=float(d.std(ddof=1)),seed_differences=d.tolist(),wins=int(sum(d>0)),ties=int(sum(d==0)),losses=int(sum(d<0)),cached=[a['cached'],b['cached']],failures=[a['failed_seeds'],b['failed_seeds']],relative_margin=mean/a['mean'])
    results.append(res)
json.dump(results,open(ROOT/'paired_results.json','w'),indent=2); print('fresh',results)
