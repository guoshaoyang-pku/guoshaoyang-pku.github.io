import json, os, hashlib, shutil, csv
from pathlib import Path
import numpy as np
P=Path('phase')
root=Path(os.path.dirname(LAB_PATH))/'experiments'/'multivariate_regression'
manifest=[]; cells=[]
def freeze(p):
    p=Path(p); data=p.read_bytes(); sha=hashlib.sha256(data).hexdigest()
    dest=P/'evidence'/sha[:16]/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(data)
    manifest.append(dict(path=str(p),sha256=sha,local=str(dest),bytes=len(data)))
    return str(dest)
def add(id,p,source,receipt=None):
    s=json.load(open(p/'candidate_spec.json')); su=json.load(open(p/'results/summary.json'))
    paths={n:freeze(p/n) for n in ['candidate_spec.json','results/summary.json','results/curves.npz','model.py','train.py','optimizer.py','loss.py']}
    a=np.load(paths['results/curves.npz']); y=a['curves']; seeds=su['seed_results']
    assert y.shape[0]==10 and np.isfinite(y).all()
    assert abs(y[:,-1].mean()-su['mean_test_mse'])<1e-12
    assert abs(y[:,-1].std()-su['std_test_mse'])<1e-12
    if receipt:
        assert abs(su['mean_test_mse']-receipt['mean'])<1e-12
        for n,info in receipt['measurement_files'].items():
            assert hashlib.sha256((p/n).read_bytes()).hexdigest()==info['sha256']
    assert s['model']['activation']=='leaky_relu' and s['model']['leaky_relu_slope']==.01
    assert s['model']['layer_norm']==[True,False,True] and s['model']['width']==256 and s['model']['depth']==3 and not s['model']['residual']
    assert s['optimizer']['betas']==[.9,.999] and s['optimizer']['weight_decay']==1e-5
    assert s['loss']['loss_id']=='mse'
    assert 'return torch.mean((pred - target) ** 2)' in (p/'loss.py').read_text()
    assert len(set((p/'loss.py').read_text().split('return ')))==2
    assert su['failed_seeds']==0 and not su['excluded']
    for T in [256,512,1024,2048]:
        if T<=y.shape[1]:
            v=y[:,T-1]; cells.append(dict(dataset=id,source=source,candidate=p.name,T=T,batch=int(a['batch_size']),lr=s['optimizer']['lr'],mean=float(v.mean()),sd=float(v.std()),seeds=v.tolist(),paths=paths,seed_ids=[z['seed'] for z in seeds]))
for name in ['poly','nested','heldout','offset','nested_offset']:
    r=json.load(open(P/(name+'_receipt.json')));id=r['dataset'].split('/')[-1]
    for result in r['results']:
        p=Path(result['measurement_files']['candidate_spec.json']['path']).parent
        add(id,p,'fresh:'+name,result)
for id in ['mvar_5a829e','mvar_93e6af','mvar_0eb811','mvar_133302']:
    for p in sorted((root/id).iterdir()):
        s=json.load(open(p/'candidate_spec.json'));m=s['model'];o=s['optimizer'];b=s['budget']
        if m.get('depth')==3 and m.get('width')==256 and not m.get('residual') and m.get('activation')=='leaky_relu' and m.get('layer_norm')==[True,False,True] and o.get('type')=='AdamW' and o.get('betas')==[.9,.999] and o.get('weight_decay')==1e-5 and b['batch_size']==16 and s['loss']['loss_id']=='mse' and s['loss'].get('lambda',0)==0:
            add(id,p,'REUSED')
json.dump(manifest,open(P/'manifest.json','w'),indent=2)
json.dump(cells,open(P/'cells.json','w'),indent=2)
# Never combine source sets: fresh receipts are joint grids; old pairs grouped by original training budget.
effects=[]
rng=np.random.default_rng(314159)
for id in sorted({c['dataset'] for c in cells}):
    for T in [256,512,1024,2048]:
        group=[c for c in cells if c['dataset']==id and c['T']==T]
        # Prefer the longest old path to avoid duplicate prefixes; verify duplicates, do not count them.
        low=[c for c in group if c['lr']==3e-5]; high=[c for c in group if c['lr']==.001]
        if not low or not high:continue
        a=max(low,key=lambda c:len(np.load(c['paths']['results/curves.npz'])['samples']))
        b=max(high,key=lambda c:len(np.load(c['paths']['results/curves.npz'])['samples']))
        assert a['seed_ids']==b['seed_ids']==list(range(10))
        d=np.array(b['seeds'])-a['seeds']; boot=d[rng.integers(0,10,(20000,10))].mean(1)
        e=dict(dataset=id,T=T,batch=a['batch'],source=a['source'],low=a['mean'],high=b['mean'],ratio=b['mean']/a['mean'],diff=float(d.mean()),paired_se=float(d.std(ddof=1)/np.sqrt(10)),ci_boot=np.quantile(boot,[.025,.975]).tolist(),high_wins=int((d<0).sum()),median_diff=float(np.median(d)),loo_diff=[float(np.delete(d,i).mean()) for i in range(10)])
        effects.append(e)
json.dump(effects,open(P/'effects.json','w'),indent=2)
for name,rows,keys in [('phase.csv',cells,['dataset','source','candidate','T','batch','lr','mean','sd']),('effects.csv',effects,['dataset','source','T','batch','low','high','ratio','diff','paired_se','high_wins','median_diff'])]:
    with open(P/name,'w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rows])
print('verified',len(manifest),'files',len(cells),'cells',len(effects),'effects')
for id in sorted({c['dataset'] for c in cells}):
    rows=[e for e in effects if e['dataset']==id]
    print(id,[(e['T'],round(e['ratio'],6),round(e['diff'],9),round(e['paired_se'],9),e['high_wins'],e['ci_boot']) for e in rows])
for name in ['poly','nested','heldout','offset','nested_offset']:
    id=json.load(open(P/(name+'_receipt.json')))['dataset'].split('/')[-1]
    print('GRID',id,[(T,[round(c['mean'],9) for c in sorted([c for c in cells if c['dataset']==id and c['T']==T],key=lambda c:c['lr'])]) for T in [256,512,1024,2048]])
