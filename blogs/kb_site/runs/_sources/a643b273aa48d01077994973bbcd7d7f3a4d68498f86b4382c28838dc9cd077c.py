import json, os, hashlib, shutil
import numpy as np
ROOT = os.path.dirname(__file__)
receipts = json.load(open(os.path.join(ROOT, 'development_receipts.json')))
if os.path.exists(os.path.join(ROOT, 'validation_receipts.json')):
    receipts += json.load(open(os.path.join(ROOT, 'validation_receipts.json')))
rows = []
for receipt in receipts:
    obj = json.loads(receipt['raw'])
    for r in obj['results']:
        r = dict(r, dataset=obj['dataset'].split('/')[-1], receipt=receipt['key'])
        rows.append(r)
        for meta in r['measurement_files'].values():
            raw = open(meta['path'], 'rb').read()
            assert hashlib.sha256(raw).hexdigest() == meta['sha256']
            out = os.path.join(ROOT, 'sources', meta['sha256'] + os.path.splitext(meta['path'])[1])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            open(out, 'wb').write(raw)
        p = os.path.dirname(r['measurement_files']['candidate_spec.json']['path'])
        for name in ['model.py','optimizer.py','train.py','loss.py']:
            raw = open(os.path.join(p,name),'rb').read()
            out = os.path.join(ROOT,'sources',hashlib.sha256(raw).hexdigest()+'.py')
            open(out,'wb').write(raw)
def seeds(r):
    return np.array([x['final_test_mse'] for x in sorted(r['seed_results'],key=lambda s:s['seed'])])
def contrast(a,b):
    d=seeds(a)-seeds(b)
    e=2.2621571628540993*d.std(ddof=1)/np.sqrt(len(d))
    return {'paired_sample_sd':float(d.std(ddof=1)),'standard_error':float(d.std(ddof=1)/np.sqrt(len(d))),'delta':float(d.mean()),'ci':[float(d.mean()-e),float(d.mean()+e)],'a_wins':int((d<0).sum()),'ratio':float(seeds(a).mean()/seeds(b).mean())}
results=[]
for ds in sorted({r['dataset'] for r in rows}):
    group=[r for r in rows if r['dataset']==ds]
    cells={(r['variant'].get('optimizer.type','RMSprop'),r['variant']['optimizer.lr'],r['variant']['budget.training_steps']):r for r in group}
    for lr in [3e-5,1e-4]:
        for t in [128,256,384,512]:
            a,b=cells['RMSprop',lr,t],cells['Adam',lr,t]
            results.append(dict(dataset=ds,lr=lr,T=t,R=a['mean'],A=b['mean'],family=contrast(a,b)))
    for t in [128,256,384,512]:
        results.append(dict(dataset=ds,T=t,rate=contrast(cells['RMSprop',1e-4,t],cells['RMSprop',3e-5,t])))
    baseline=cells['RMSprop',0,128]
    results.append(dict(dataset=ds,baseline_raw_mean=float(seeds(baseline).mean()),baseline_flags=baseline['failed_seeds']))
    # Verify all non-baseline budgets are prefixes of the T512 seed curves.
    for typ in ['RMSprop','Adam']:
        for lr in [3e-5,1e-4]:
            end=np.load(cells[typ,lr,512]['measurement_files']['results/curves.npz']['path'])['curves']
            for t in [128,256,384]:
                cur=np.load(cells[typ,lr,t]['measurement_files']['results/curves.npz']['path'])['curves']
                assert np.array_equal(cur,end[:,:t])
    for r in group:
        if r['variant']['optimizer.lr']==0:
            cur=np.load(r['measurement_files']['results/curves.npz']['path'])['curves']
            if cur.shape[1]: assert np.all(cur==cur[:,0,None])
json.dump(results,open(os.path.join(ROOT,'analysis.json'),'w'),indent=2)
json.dump(rows,open(os.path.join(ROOT,'rows.json'),'w'),indent=2)
print('rows',len(rows),'cached',sum(r['cached'] for r in rows),'failed',sum(r['failed_seeds'] for r in rows),'prefix checks passed')

