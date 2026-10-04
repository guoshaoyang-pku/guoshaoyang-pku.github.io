import json, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT/name).read_text())
def experiment(name):
    wrapper=read(name)
    return json.loads(next(c['text'] for c in wrapper['content'] if c['type']=='text'))
out=[]
for name in ['exp1.json','exp2_corrected.json']:
    e=experiment(name)
    for i in range(0,len(e['results']),2):
        p,m=e['results'][i:i+2]
        pv,mv=p['variant'],m['variant']
        assert pv['optimizer.lr']==.001 and mv['optimizer.lr']==.0001
        assert pv['optimizer.momentum']==0 and mv['optimizer.momentum']==.9
        assert {k:v for k,v in pv.items() if k not in ['optimizer.lr','optimizer.momentum']}=={k:v for k,v in mv.items() if k not in ['optimizer.lr','optimizer.momentum']}
        assert p['error'] is None and m['error'] is None
        assert math.isfinite(p['mean']) and math.isfinite(m['mean'])
        row=dict(dataset=e['dataset'],T=pv['budget.training_steps'],wd=pv['optimizer.weight_decay'],plain=p['mean'],plain_sd=p['std'],momentum=m['mean'],momentum_sd=m['std'],ratio=p['mean']/m['mean'],margin=p['mean']-m['mean'],plain_win=p['mean']<m['mean'],finite_seed_count=None,failed_seed_count=None)
        out.append(row)
        print(json.dumps(row))
assert len(out)==8
(ROOT/'computed_results.json').write_text(json.dumps(out,indent=2))
pairs=read('parity_pairs.json')
misses=[x for x in pairs if x['p']['family']=='multivariate_regression' and x['p']['mean']>=x['m']['mean']]
(ROOT/'base_multivariate_misses.json').write_text(json.dumps(misses,indent=2))
print('MULTIVARIATE MISSES',[(x['p']['set_id'],x['p']['candidate_id'],x['m']['candidate_id'],x['p']['mean']/x['m']['mean']) for x in misses])
failed=experiment('exp2.json')
assert len(failed['results'])==8 and all(r['mean'] is None and r['error'] for r in failed['results'])
print('VALIDATED',sum(r['plain_win'] for r in out),'of',len(out),'successful comparisons; eight initial protocol failures')
