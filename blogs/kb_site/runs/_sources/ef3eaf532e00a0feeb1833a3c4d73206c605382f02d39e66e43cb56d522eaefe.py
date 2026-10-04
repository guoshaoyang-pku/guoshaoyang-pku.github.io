import json, glob, hashlib, os, numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def contrast(a, b):
    x = np.array([s['final_test_mse'] for s in a['seed_results']])
    y = np.array([s['final_test_mse'] for s in b['seed_results']])
    d = x-y
    radius = 2.262157163*np.std(d, ddof=1)/np.sqrt(len(d))
    return dict(margin=float(d.mean()), ci=[float(d.mean()-radius),float(d.mean()+radius)],
                sgd_seed_wins=int((d<0).sum()), n=len(d),
                ratio=a['mean']/b['mean'])

def main():
    effects=[]
    for p in sorted((ROOT/'experiments').glob('*.json')):
        exp=json.load(open(p))
        rs=exp['result']['results']
        ds=exp['result']['dataset']
        for r in rs:
            assert r['failed_seeds']==0 and r['n_seeds']==10 and not r['excluded']
            assert abs(np.mean([s['final_test_mse'] for s in r['seed_results']])-r['mean'])<1e-10
            for m in r['measurement_files'].values():
                f=ROOT/m['repo_path']
                assert hashlib.sha256(f.read_bytes()).hexdigest()==m['sha256']
        for act in ['silu','leaky_relu']:
            for lr in [.001,.0006]:
                a=next(r for r in rs if r['candidate']['model']['activation']==act and r['candidate']['optimizer']['type']=='SGD' and r['candidate']['optimizer']['lr']==lr)
                b=next(r for r in rs if r['candidate']['model']['activation']==act and r['candidate']['optimizer']['type']=='RMSprop')
                effects.append(dict(dataset=ds,activation=act,lr=lr,sgd_mean=a['mean'],sgd_sd=a['std'],rms_mean=b['mean'],rms_sd=b['std'],**contrast(a,b)))
        a=next(r for r in rs if r['candidate']['model']['activation']=='silu' and r['candidate']['optimizer']['type']=='SGD' and r['candidate']['optimizer']['lr']==.001)
        b=next(r for r in rs if r['candidate']['model']['activation']=='leaky_relu' and r['candidate']['optimizer']['type']=='RMSprop')
        effects.append(dict(dataset=ds,activation='silu_vs_leaky_relu',lr=.001,**contrast(a,b)))
    json.dump(effects,open(ROOT/'boundary/effects.json','w'),indent=2)
    for e in effects: print(e)
    pairs=json.load(open(ROOT/'boundary/audit_pairs.json'))
    strict=[p for p in pairs if any(p['r']['model']['layer_norm'])]
    for name,pp in [('supplied',pairs),('both_normalized',strict)]:
        print(name,len(pp),sum(p['ratio']<1 for p in pp),len(set(p['s']['set_id'] for p in pp)),np.median([p['ratio'] for p in pp]))
    strata=[]
    for p in pairs:
        s,r=p['s'],p['r']
        strata.append(dict(set_id=s['set_id'],sgd_id=s['candidate_id'],rms_id=r['candidate_id'],target=s['dataset'],sgd_model=s['model'],rms_model=r['model'],sgd_optimizer=s['optimizer'],rms_optimizer=r['optimizer'],sgd_budget=s['budget'],rms_budget=r['budget'],sgd_mean=s['mean'],rms_mean=r['mean'],ratio=p['ratio'],excluded_activation=p['excluded_activation']))
    json.dump(strata,open(ROOT/'boundary/strata.json','w'),indent=2)

if __name__=='__main__': main()
