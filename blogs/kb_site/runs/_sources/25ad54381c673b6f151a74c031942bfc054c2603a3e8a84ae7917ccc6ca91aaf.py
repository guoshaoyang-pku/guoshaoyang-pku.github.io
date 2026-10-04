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
for name in ['poly','nested','heldout','offset','nested_offset','batch']:
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
# Attach comparison-set provenance after freezing.
lab=load_lab()
for c in cells:
 s=json.load(open(c['paths']['candidate_spec.json']));su=json.load(open(c['paths']['results/summary.json']))
 c['source_sets']=sorted({r['set_id'] for r in lab if r['dataset_id']==c['dataset'] and r['budget']==s['budget'] and r['optimizer']==s['optimizer'] and abs(r['mean']-su['mean_test_mse'])<1e-12})
 assert c['source_sets']
json.dump(cells,open(P/'cells.json','w'),indent=2)
