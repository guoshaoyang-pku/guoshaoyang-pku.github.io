import json,gzip,os,hashlib
import numpy as np
from pathlib import Path

root=Path(__file__).resolve().parent.parent
os.chdir(root)
receipts=json.load(open('transport/receipts.json'))
out=[]
curves={}
for key,receipt in receipts.items():
    rows={}
    for r in receipt['results']:
        t=r['variant']['budget.training_steps']
        opt=r['variant']['optimizer.type']
        y=np.array([s['final_test_ce'] for s in sorted(r['seed_results'],key=lambda s:s['seed'])])
        assert len(y)==10 and np.isfinite(y).all()
        good=np.array([not s['failed'] for s in sorted(r['seed_results'],key=lambda s:s['seed'])])
        if good.any(): assert abs(y[good].mean()-r['mean'])<1e-12
        rows[t,opt]=(r,y)
        for source in r['measurement_files'].values():
            p=Path(source['repo_path'])
            assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256']
        c=np.load(r['measurement_files']['results/curves.npz']['repo_path'])['curves']
        if c.shape==(10,t) and not r['failed_seeds']:
            assert np.array_equal(c[:,-1],y)
            curves[key,t,opt]=c
    for t in sorted(set(t for t,opt in rows)):
        rr,ry=rows[t,'RMSprop'];aa,ay=rows[t,'AdamW']
        d=ry-ay; se=d.std(ddof=1)/np.sqrt(len(d))
        out.append(dict(dataset=receipt['dataset'],T=t,R=rr['mean'],A=aa['mean'],
            SD_R=rr['std'],SD_A=aa['std'],margin=float(d.mean()),
            ratio=float(ry.mean()/ay.mean()),raw_all_seed_mean_R=float(ry.mean()),excluded=rr['excluded'],failed_R=rr['failed_seeds'],paired_SD=float(d.std(ddof=1)),
            CI95=[float(d.mean()-2.262157*se),float(d.mean()+2.262157*se)],
            R_seed_wins=int((d<0).sum())))
prefix=[]
trajectory=[]
for key in receipts:
    for opt in ['RMSprop','AdamW']:
        ts=sorted(t for k,t,o in curves if k==key and o==opt)
        if 1024 not in ts: continue
        for t in ts[:-1]:
            prefix.append(dict(dataset=key,opt=opt,T=t,identical=bool(np.array_equal(curves[key,t,opt],curves[key,1024,opt][:,:t]))))
        m=curves[key,1024,opt].mean(axis=0)
        trajectory.append(dict(dataset=key,opt=opt,step16=float(m[15]),step128=float(m[127]),
            minimum=float(m.min()),minimum_step=int(m.argmin()+1),
            endpoint=float(m[-1]),late_rise=float(m[-1]-m.min())))
json.dump(dict(contrasts=out,prefix=prefix,trajectory=trajectory),open('transport/analysis.json','w'),indent=2)
for x in out: print(json.dumps(x))
print('prefix',prefix)
print('trajectory',trajectory)
