import json, hashlib, os, numpy as np
from target_predictor import parse, features
rows=json.load(open('inherited.json'))+json.load(open('fresh_results.json'))
for r in rows:
    for item in r['measurement_files'].values():
        assert hashlib.sha256(open(item['repo_path'],'rb').read()).hexdigest()==item['sha256']
    c=np.load(r['measurement_files']['results/curves.npz']['repo_path'])
    assert c['curves'].shape==(10,256) and c['samples'][-1]==16384 and c['batch_size']==64
    assert np.allclose(c['curves'][:,-1],[s['final_test_mse'] for s in r['seed_results']])
p=json.load(open('preregistration.json')); result=[]
for f in p['fresh']:
    pair={r['optimizer']['type']:r for r in rows if r['dataset_id']==f['dataset']}
    a={s['seed']:s['final_test_mse'] for s in pair['SGD']['seed_results']};b={s['seed']:s['final_test_mse'] for s in pair['Adam']['seed_results']}
    assert sorted(a)==sorted(b)==list(range(10))
    d=np.array([a[s]-b[s] for s in range(10)]);h=2.262157*np.std(d,ddof=1)/np.sqrt(10)
    result.append(dict(dataset=f['dataset'],set_id=pair['SGD']['set_id'],sgd=pair['SGD']['mean'],adam=pair['Adam']['mean'],delta=float(d.mean()),ci=[float(d.mean()-h),float(d.mean()+h)],sgd_seed_wins=int(np.sum(d<0)),correct=bool((d.mean()<0)==f['pred_sgd'])))
json.dump(result,open('prospective_analysis.json','w'),indent=2)
print(json.dumps(result,indent=2))
for s in ['__import__("os")','x0.__class__','x0[0]','sin(x0,x1)','unknown(x0)']:
    try: parse(s)
    except ValueError: continue
    raise AssertionError('unsafe accepted')
print('reproduction and parser checks passed')
