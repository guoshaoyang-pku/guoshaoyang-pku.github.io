import json, hashlib
from pathlib import Path
import numpy as np

def contrast(a,b):
    x=np.array([s['final_test_mse'] for s in a['seed_results']])
    y=np.array([s['final_test_mse'] for s in b['seed_results']])
    d=x-y
    assert len(d)==10 and np.isfinite(d).all()
    half=2.2621571627409915*d.std(ddof=1)/np.sqrt(10)
    return dict(margin=float(d.mean()),ci=[float(d.mean()-half),float(d.mean()+half)],wins=int((d<0).sum()))

effects=[]
for i in range(4):
    receipt=json.loads(Path(f'audit/receipt{i}.json').read_text())
    rows=receipt['results']
    assert len(rows)==9
    for r in rows:
        assert r['n_seeds']==10 and not r['excluded'] and r['failed_seeds']==0
        values=[s['final_test_mse'] for s in r['seed_results']]
        assert np.isclose(np.mean(values),r['mean'])
        for f in r['measurement_files'].values():
            assert hashlib.sha256(Path(f['repo_path']).read_bytes()).hexdigest()==f['sha256']
    e=dict(dataset=receipt['dataset'],means=[r['mean'] for r in rows],std=[r['std'] for r in rows],cached=[r['cached'] for r in rows])
    e['full']=[contrast(rows[0],rows[j]) for j in [1,2]]
    e['zero_decay']=[contrast(rows[3],rows[j]) for j in [4,5]]
    e['low_rate']=[contrast(rows[6],rows[j]) for j in [7,8]]
    e['high_minus_low']=[contrast(rows[j],rows[j+6]) for j in range(3)]
    e['decay_minus_zero']=[contrast(rows[j],rows[j+3]) for j in range(3)]
    effects.append(e)
Path('audit/effects.json').write_text(json.dumps(effects,indent=2))
print(json.dumps(effects,indent=2))
