import json,glob,collections,numpy as np,hashlib
rows=[json.load(open(p)) for p in sorted(glob.glob('research/new_*.json'))]
by=collections.defaultdict(dict)
for r in rows:by[r['dataset_id']][(r['activation'],''.join(str(int(x)) for x in r['layer_norm']))]=r
out={}
def contrast(a,b):
 aa={x['seed']:x['final_test_ce'] for x in a['seed_results'] if not x['failed']}
 bb={x['seed']:x['final_test_ce'] for x in b['seed_results'] if not x['failed']}
 seeds=sorted(set(aa)&set(bb));d=np.array([aa[s]-bb[s] for s in seeds])
 rng=np.random.default_rng(913);boot=d[rng.integers(0,len(d),(10000,len(d)))].mean(1)
 return dict(difference=float(d.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),wins=int((d<0).sum()),n_pairs=len(d),ratio=a['mean']/b['mean'],ids=[a['candidate_id'],b['candidate_id']])
for ds,c in by.items():
 result={'set_id':next(iter(c.values()))['set_id'],'cells':{a+'_'+m:dict(mean=r['mean'],std=r['std'],finite=int(sum(np.isfinite(x.get('final_test_ce',float('nan'))) for x in r['seed_results'])),failures=r['failed_seeds'],cached=r['cached']) for (a,m),r in c.items()}}
 result['B_minus_C']=contrast(c['silu','010'],c['leaky_relu','011'])
 result['activation']={m:contrast(c['silu',m],c['leaky_relu',m]) for m in ['000','010','011','111']}
 result['LN_011_minus_010']={a:contrast(c[a,'011'],c[a,'010']) for a in ['silu','leaky_relu']}
 out[ds]=result
 print(ds,json.dumps(result))
verified=0
for r in rows:
 for f in r['measurement_files'].values():
  assert hashlib.sha256(open(f['repo_path'],'rb').read()).hexdigest()==f['sha256'];verified+=1
print('verified',verified)
json.dump(out,open('research/new_analysis.json','w'),indent=2)
