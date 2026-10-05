import json, pathlib, hashlib
import numpy as np
ROOT=pathlib.Path(__file__).parent
rows=[]
verified=[]
for path in sorted(ROOT.glob('*_response.json')):
    raw=json.loads(path.read_text())
    data=json.loads(raw['content'][0]['text'])
    results=data['results']
    curves=[]
    for r in results:
        assert r['failed_seeds']==0 and not r['excluded']
        for name,m in r['measurement_files'].items():
            p=pathlib.Path(m['path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
        spec=json.loads(pathlib.Path(r['measurement_files']['candidate_spec.json']['path']).read_text())
        code=pathlib.Path(r['measurement_files']['executed/train.py']['path']).read_text()
        assert 'optimizer.step()' in code and 'scheduler' not in code
        z=np.load(r['measurement_files']['results/curves.npz']['path'])
        assert z['curves'].shape==(10,1024)
        assert np.array_equal(z['samples'],np.arange(1,1025)*16)
        c=z['curves']
        assert abs(c[:,-1].mean()-r['mean'])<1e-12
        curves.append(c)
        verified.append(dict(group=path.stem,spec=spec,files=r['measurement_files'],cached=r['cached']))
    rms=curves[0]
    for j,ratio in enumerate([1,3,10],1):
        for T in [256,1024]:
            d=curves[j][:,T-1]-rms[:,T-1]
            rng=np.random.default_rng(2000+T+j)
            boot=d[rng.integers(0,10,(20000,10))].mean(axis=1)
            ci=np.quantile(boot,[.025,.975])
            rows.append(dict(group=path.stem.replace('_response',''),T=T,ratio=ratio,
                 rms_mean=float(rms[:,T-1].mean()),adam_mean=float(curves[j][:,T-1].mean()),
                 rms_sd=float(rms[:,T-1].std()),adam_sd=float(curves[j][:,T-1].std()),
                 D=float(d.mean()),paired_SE=float(d.std(ddof=1)/np.sqrt(10)),
                 CI=ci.tolist(),adam_seed_wins=int((d<0).sum()),
                 phase='A' if ci[1]<0 else 'R' if ci[0]>0 else '?',
                 predicted='R' if ratio==1 else 'A',
                 early_D={str(t):float((curves[j][:,t-1]-rms[:,t-1]).mean()) for t in [1,16,64]},
                 late_sd_rms=float(rms[:,960:].std(axis=1).mean()),
                 late_sd_adam=float(curves[j][:,960:].std(axis=1).mean())))
(ROOT/'phase.json').write_text(json.dumps(rows,indent=2))
(ROOT/'verified_specs.json').write_text(json.dumps(verified))
import csv
with (ROOT/'phase.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for x in rows:
    print(x['group'],x['T'],x['ratio'],'means',format(x['adam_mean'],'.6g'),format(x['rms_mean'],'.6g'),
          'D',format(x['D'],'.6g'),'CI',*[format(v,'.6g') for v in x['CI']],
          x['phase'],'seedwins',x['adam_seed_wins'])
print('verified candidates',len(verified))
