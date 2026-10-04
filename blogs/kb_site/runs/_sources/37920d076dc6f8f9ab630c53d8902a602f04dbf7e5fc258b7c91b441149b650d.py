import collections, hashlib, itertools, json, re
from pathlib import Path
import numpy as np

def history_protocol(history):
    rows=[]
    for x in history:
        q=x['question']; cs=re.split(r'### Choice [A-Z]',q)[1:]
        if x['family']!='synthetic_tabular_classification' or not cs: continue
        opts=[re.search(r'Optimizer: (\w+)',c) for c in cs]
        rates=[re.search(r'Learning rate: ([\deE.\-]+)',c) for c in cs]
        ts=re.findall(r'training_steps: (\d+)',q)
        rule=re.search(r'Rule family: `([^`]+)`',q)
        if not all(opts) or not all(rates) or not ts: continue
        opts=[z.group(1) for z in opts]; rates=[float(z.group(1)) for z in rates]
        if len(set(opts))==len(set(rates))==len(set(ts))==1:
            rows.append(dict(id=x['question_id'],rule=rule.group(1) if rule else 'unknown',opt=opts[0],lr=rates[0],T=int(ts[0]),answer=x['answer'],prediction=x['prediction']))
    return rows

def analyze():
    specs={}; summaries={}; manifest=[]
    for p in Path('measurements').iterdir():
        digest=hashlib.sha256(p.read_bytes()).hexdigest()
        assert digest==p.stem, p
        manifest.append(dict(path=str(p),sha256=digest))
        if p.suffix!='.json': continue
        x=json.loads(p.read_text())
        if 'model' in x: specs[x['candidate_id']]=x
        elif 'seed_results' in x: summaries[x['candidate_id']]=x
    groups=collections.defaultdict(dict)
    for cid,spec in specs.items():
        label={3:'A',4:'B',1:'C'}[spec['model']['depth']]
        groups[(spec['dataset_id'],spec['budget']['training_steps'])][label]=summaries[cid]
    result=[]; rng=np.random.default_rng(913)
    for (ds,t),cells in sorted(groups.items()):
        b=np.array([r['final_test_ce'] for r in cells['B']['seed_results']]); contrasts={}
        for label in ['A','C']:
            other=np.array([r['final_test_ce'] for r in cells[label]['seed_results']]); d=b-other
            boot=d[rng.integers(0,10,size=(10000,10))].mean(axis=1)
            contrasts[label]=dict(difference=float(d.mean()),ratio=float(b.mean()/other.mean()),wins=int((d<0).sum()),se=float(d.std(ddof=1)/np.sqrt(10)),bootstrap95=np.quantile(boot,[.025,.975]).tolist())
        result.append(dict(dataset=ds,T=t,cells=cells,B_minus=contrasts))
    output=dict(groups=result,manifest=manifest,hash_verified=len(manifest))
    Path('research/analysis_results.json').write_text(json.dumps(output,indent=2))
    return output

if __name__=='__main__':
    for g in analyze()['groups']: print(g['dataset'],g['T'],g['B_minus'])
