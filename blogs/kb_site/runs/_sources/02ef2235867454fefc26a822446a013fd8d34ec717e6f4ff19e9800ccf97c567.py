import gzip,hashlib,itertools,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def contrast(a,b):
    assert a['set_id']==b['set_id']
    assert not a['failed_seeds'] and not b['failed_seeds']
    x={s['seed']:s['final_test_mse'] for s in a['seed_results']}
    y={s['seed']:s['final_test_mse'] for s in b['seed_results']}
    assert sorted(x)==sorted(y)==list(range(10))
    d=np.array([x[s]-y[s] for s in sorted(x)])
    m=float(d.mean());h=float(2.2621571627409915*d.std(ddof=1)/np.sqrt(10))
    return dict(a=a['candidate_id'],b=b['candidate_id'],set_id=a['set_id'],dataset=a['dataset_id'],mean_a=a['mean'],mean_b=b['mean'],ratio=a['mean']/b['mean'],margin=m,ci95=[m-h,m+h],wins=int((d<0).sum()),n=10)
def main():
    rows=json.loads((ROOT/'cells.json').read_text());cs=[];cv=[];mf=[]
    for r in rows:
        assert r['n_seeds']==10 and r['failed_seeds']==0
        for label,info in r['measurement_files'].items():
            p=ROOT.parent/info['repo_path']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==info['sha256']
            mf.append(dict(dataset=r['dataset_id'],set_id=r['set_id'],candidate=r['candidate_id'],label=label,**info))
        c=np.load(ROOT.parent/r['measurement_files']['results/curves.npz']['repo_path'])['curves']
        ends=np.array([s['final_test_mse'] for s in r['seed_results']])
        assert np.allclose(c[:,-1],ends) and np.isclose(ends.mean(),r['mean'])
        cv.append(dict(dataset=r['dataset_id'],candidate=r['candidate_id'],optimizer=r['optimizer'],budget=r['budget'],first=float(c[:,0].mean()),final=float(c[:,-1].mean()),last64=float(c[:,-64:].mean()),late_temporal_sd=float(c[:,-64:].std(axis=1).mean()),minimum=float(c.mean(axis=0).min()),steps={str(i):float(c[:,i-1].mean()) for i in [1,16,64,128,256,512] if i<=c.shape[1]}))
    for a,b in itertools.combinations(rows,2):
        if a['set_id']==b['set_id'] and a['budget']==b['budget'] and a['model']==b['model'] and a['loss']==b['loss']:
            x=contrast(a,b);x.update(optimizer_a=a['optimizer'],optimizer_b=b['optimizer'],budget=a['budget']);cs.append(x)
    for n,o in [('contrasts',cs),('curves',cv),('manifest',mf)]:
        (ROOT/(n+'.json')).write_text(json.dumps(o,indent=2))
    print(dict(cells=len(rows),new_seed_attempts=sum(not r["cached"] for r in rows)*10, cached_rows=sum(r["cached"] for r in rows),failures=sum(r['failed_seeds'] for r in rows),cached=sum(r['cached'] for r in rows),sets=sorted(set(r['set_id'] for r in rows)),contrasts=len(cs),verified_files=len(mf)))
if __name__=='__main__':main()
