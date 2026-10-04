import json,gzip,hashlib,math,itertools,collections
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def contrast(a,b):
    aa={s['seed']:s for s in a['seed_results']}
    bb={s['seed']:s for s in b['seed_results']}
    ids=sorted(set(aa)&set(bb))
    ids=[i for i in ids if not aa[i]['failed'] and not bb[i]['failed']]
    d=np.array([bb[i]['final_test_ce']-aa[i]['final_test_ce'] for i in ids])
    assert len(d)==10 and np.isfinite(d).all()
    mean=float(d.mean());se=float(d.std(ddof=1)/math.sqrt(len(d)))
    return dict(effect=mean,ci=[mean-2.2621571627409915*se,mean+2.2621571627409915*se],a_wins=int((d>0).sum()),n=len(d),a_losses=[i for i,v in zip(ids,d) if v<0])
def analyze():
    out=[]; hash_count=0;source_manifest=[]
    for ix in range(1,4):
        raw=json.loads((ROOT/f'run{ix}.json').read_text());rs=raw['results']; ds=raw['dataset']
        for r in rs:
            assert len(r['seed_results'])==10 and r['failed_seeds']==0 and not r['excluded']
            assert abs(np.mean([s['final_test_ce'] for s in r['seed_results']])-r['mean'])<1e-12
            for name,f in r['measurement_files'].items():
                p=ROOT.parent/f['repo_path']
                assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
                hash_count+=1
            srcdir=Path(r['measurement_files']['candidate_spec.json']['path']).parent
            dst=ROOT/'sources'/ds.split('/')[-1]/srcdir.name
            dst.mkdir(parents=True,exist_ok=True)
            for name in ['model.py','optimizer.py','loss.py','train.py']:
                data=(srcdir/name).read_bytes();(dst/name).write_bytes(data)
                source_manifest.append(dict(path=str(dst.relative_to(ROOT)),sha256=hashlib.sha256(data).hexdigest(),execution_pinned=r['source_provenance'] is not None))
        for offset in [0,4,8]:
            cells=rs[offset:offset+4];a,ah,w,wh=cells
            row=dict(dataset=ds,width=a['variant']['model.width'],batch=a['variant']['budget.batch_size'],
                means=[r['mean'] for r in cells],sd=[r['std'] for r in cells],
                accuracy=[float(np.mean([s['final_test_accuracy'] for s in r['seed_results']])) for r in cells],
                full=contrast(a,wh),adam_rate=contrast(a,ah),adamw_rate=contrast(w,wh),
                low_family=contrast(a,w),high_family=contrast(ah,wh))
            out.append(row)
    result=dict(rows=out,verified_files=hash_count,source_files=source_manifest,
        scheduled_seeds=360,finite_seeds=360,failures=0,excluded=0,cached=0)
    (ROOT/'statistics.json').write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    r=analyze()
    for row in r['rows']: print(json.dumps(row))
    print('verified_files',r['verified_files'])
