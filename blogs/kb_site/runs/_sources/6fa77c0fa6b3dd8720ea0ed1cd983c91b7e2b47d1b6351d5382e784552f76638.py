"""Recompute phase statistics from frozen, allowed measurements; no training."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
cells=json.loads((P/'cells.json').read_text())
manifest=json.loads((P/'manifest.json').read_text())
for f in manifest:
    p=P.parent/f['local']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256']
for c in cells:
    y=np.load(P.parent/c['paths']['results/curves.npz'])['curves'][:,c['T']-1]
    assert np.allclose(y,c['seeds'],rtol=0,atol=0)
    assert abs(y.mean()-c['mean'])<1e-12 and abs(y.std()-c['sd'])<1e-12
effects=[]; contrasts=[]
for id in sorted({c['dataset'] for c in cells}):
    for batch in [16,32]:
        for T in [256,512,1024,2048]:
            groups=sorted({z for c in cells if c['dataset']==id and c['batch']==batch and c['T']==T for z in c['source_sets']})
            for source_set in groups:
                group=[c for c in cells if c['dataset']==id and c['batch']==batch and c['T']==T and source_set in c['source_sets']]
                rates=sorted({c['lr'] for c in group})
                chosen={}
                for lr in rates:
                    options=[c for c in group if c['lr']==lr]
                    chosen[lr]=max(options,key=lambda c:len(np.load(P.parent/c['paths']['results/curves.npz'])['samples']))
                    # Same canonical training prefixes may occur in inherited endpoints.
                    # They are reused and never added as replications.
                    for c in options:
                        assert np.allclose(c['seeds'],chosen[lr]['seeds'],rtol=0,atol=0)
                for i,low in enumerate(rates):
                    for high in rates[i+1:]:
                        a,b=chosen[low],chosen[high]
                        assert a['seed_ids']==b['seed_ids']==list(range(10))
                        d=np.array(b['seeds'])-a['seeds']
                        seed=int(hashlib.sha256(f'{id}:{batch}:{T}:{low}:{high}'.encode()).hexdigest()[:8],16)
                        rng=np.random.default_rng(seed)
                        boot=d[rng.integers(0,10,(20000,10))].mean(1)
                        ci=np.quantile(boot,[.025,.975]).tolist()
                        e=dict(source_set=source_set,dataset=id,T=T,batch=batch,source=a['source'],low_lr=low,high_lr=high,low=a['mean'],high=b['mean'],ratio=b['mean']/a['mean'],diff=float(d.mean()),paired_se=float(d.std(ddof=1)/np.sqrt(10)),ci_boot=ci,high_wins=int((d<0).sum()),median_diff=float(np.median(d)),loo_diff=[float(np.delete(d,i).mean()) for i in range(10)],phase='H' if ci[1]<0 else ('L' if ci[0]>0 else '?'))
                        contrasts.append(e)
                        if low==3e-5 and high==.001:effects.append(e)
(P/'effects.json').write_text(json.dumps(effects,indent=2))
(P/'all_contrasts.json').write_text(json.dumps(contrasts,indent=2))
for name,rows,keys in [('phase.csv',cells,['dataset','source','candidate','T','batch','lr','mean','sd']),('effects.csv',effects,['dataset','source','T','batch','low','high','ratio','diff','paired_se','high_wins','median_diff','phase'])]:
    with (P/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
print('Verified frozen files:',len(manifest),'checkpoint cells:',len(cells),'extreme contrasts:',len(effects))
for id in sorted({c['dataset'] for c in cells}):
    for b in [16,32]:
        es=[e for e in effects if e['dataset']==id and e['batch']==b]
        if es:print(id,b,[(e['T'],e['phase'],round(e['ratio'],6)) for e in es])
print('Late fresh uncertainty:',json.dumps([e for e in effects if e['source'].startswith('fresh:') and e['T']==2048]))
