import hashlib,json
from pathlib import Path
import numpy as np

def main():
    for row in json.load(open('artifact_manifest.json')):
        assert hashlib.sha256(Path(row['local']).read_bytes()).hexdigest()==row['sha256']
    data=json.load(open('recovered_results.json')); cells={}; means=[]
    for cid,s in data['specs'].items():
        r=data['summaries'][cid]; mask=''.join(str(int(v)) for v in s['model']['layer_norm'])
        key=(s['dataset_id'],s['budget']['training_steps'],mask)
        seeds=sorted(r['seed_results'],key=lambda x:x['seed'])
        assert [x['seed'] for x in seeds]==list(range(10))
        assert not r['excluded'] and r['failed_seeds']==0 and all(not x['failed'] for x in seeds)
        y=np.array([x['final_test_mse'] for x in seeds])
        curves=np.load('archive/'+cid+'/results_curves.npz')['curves']
        assert curves.shape==(10,key[1]) and np.allclose(curves[:,-1],y)
        assert np.isclose(y.mean(),r['mean_test_mse'])
        m=json.load(open('archive/'+cid+'/results_execution_manifest.json'))
        for f,info in m['sources'].items():
            raw=Path('archive/'+cid+'/'+f).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==info['sha256']
            src=Path(info.get('path','archive/'+cid+'/'+f))
        model=Path('archive/'+cid+'/model.py').read_text()
        assert 'nn.LeakyReLU(0.01)' in model
        opt=Path('archive/'+cid+'/optimizer.py').read_text()
        assert 'torch.optim.Adagrad(params, lr=0.001, weight_decay=0)' in opt
        cells[key]=y;means.append(dict(dataset=key[0],steps=key[1],mask=mask,mean=float(y.mean()),sd=float(y.std())))
    rng=np.random.default_rng(9371);idx=rng.integers(0,10,(20000,10));results=[]
    for (dataset,t,mask),y in sorted(cells.items()):
        if mask=='11':continue
        for rival in (['11','110','011'] if t==512 and mask=='101' else ['11']):
            x=cells[(dataset,t,rival)];d=y-x
            results.append(dict(dataset=dataset,steps=t,mask=mask,rival=rival,difference=float(d.mean()),ratio=float(y.mean()/x.mean()),wins=int((y<x).sum()),difference_ci=np.quantile(d[idx].mean(axis=1),[.025,.975]).tolist(),ratio_ci=np.quantile(y[idx].mean(axis=1)/x[idx].mean(axis=1),[.025,.975]).tolist()))
    json.dump({'means':means,'contrasts':results},open('statistics.json','w'),indent=2)
    for r in means:print('MEAN',r)
    for r in results:print('CONTRAST',r)
if __name__=='__main__': main()
