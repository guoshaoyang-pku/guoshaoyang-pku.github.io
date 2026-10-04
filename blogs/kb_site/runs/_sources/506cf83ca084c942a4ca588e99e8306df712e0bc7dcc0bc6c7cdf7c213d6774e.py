import json, glob, os, hashlib, shutil
import numpy as np
from scipy.stats import t
effects=[]
manifest=[]
for receipt in sorted(glob.glob('experiments/*.json')):
    exp=json.load(open(receipt))
    r=exp['result']
    if isinstance(r,str): r=json.loads(r)
    print('resultkeys',list(r))
    rows=r['results']
    dataset=r['dataset']
    for row in rows:
        for name,f in row['measurement_files'].items():
            p=f['repo_path']
            assert hashlib.sha256(open(p,'rb').read()).hexdigest()==f['sha256']
            manifest.append(f)
        source=os.path.dirname(row['measurement_files']['candidate_spec.json']['path'])
        dest='transport/code/'+dataset.split('/')[-1]+'/'+os.path.basename(source)
        os.makedirs(dest,exist_ok=True)
        for name in ['model.py','optimizer.py','loss.py','train.py']:
            open(dest+'/'+name,'w').write(open(source+'/'+name).read())
    def contrast(i,j,label):
        a,b=rows[i],rows[j]
        x=np.array([s['final_test_mse'] for s in a['seed_results']])
        y=np.array([s['final_test_mse'] for s in b['seed_results']])
        d=x-y; se=d.std(ddof=1)/np.sqrt(10)
        out=dict(dataset=dataset,label=label,mean_a=a['mean'],sd_a=a['std'],mean_b=b['mean'],sd_b=b['std'],difference=float(d.mean()),ratio=a['mean']/b['mean'],a_seed_wins=int((x<y).sum()),n=10,ci95=[float(d.mean()-t.ppf(.975,9)*se),float(d.mean()+t.ppf(.975,9)*se)],ci_family6=[float(d.mean()-t.ppf(1-.05/12,9)*se),float(d.mean()+t.ppf(1-.05/12,9)*se)])
        effects.append(out)
        print(json.dumps(out))
    for i,j,label in [(0,1,'512 E-D'),(0,2,'512 SiLU SGD-AdamW'),(3,1,'512 ReLU SGD-AdamW'),(4,5,'256 E-D'),(4,6,'256 SiLU SGD-AdamW'),(7,5,'256 ReLU SGD-AdamW'),(0,4,'E 512-256'),(1,5,'D 512-256'),(2,6,'SiLU AdamW 512-256'),(3,7,'ReLU SGD 512-256')]:
        contrast(i,j,label)
json.dump(effects,open('transport/effects.json','w'),indent=2)
json.dump(manifest,open('transport/manifest.json','w'),indent=2)
print('validated artifacts',len(manifest))
