import ast,json,hashlib,re
from pathlib import Path
import numpy as np
R=Path(__file__).parent
ex=json.load(open(R/'exact_experiments.json'))
q=json.load(open(R/'history/q_f03f18.json'))['question']
sections=dict(re.findall(r'### Choice ([A-E])\n(.*?)(?=### Choice |## Your answer|\Z)',q,re.S))
qmodel=re.findall(r'```python\n(.*?)```',sections['D'],re.S)[0]
def classes(s):
 return [ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.ClassDef)]
artifacts=set()
for row in ex:
 assert row['n_seeds']==10 and row['failed_seeds']==0 and not row['excluded'] and not row['cached']
 assert row['budget']==dict(training_steps=512,batch_size=32,total_samples_seen=16384)
 assert row['source_provenance']['status']=='verified'
 for name,f in row['measurement_files'].items():
  p=Path(f['repo_path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'];artifacts.add(str(p))
 assert classes(Path(row['measurement_files']['executed/model.py']['repo_path']).read_text())==classes(qmodel)
 losses=[s['final_test_ce'] for s in row['seed_results']]
 assert len(losses)==10 and np.isclose(np.mean(losses),row['mean'],atol=1e-14,rtol=0)
 curves=np.load(row['measurement_files']['results/curves.npz']['repo_path'])['curves']
 assert curves.shape==(10,512) and np.array_equal(curves[:,-1],losses)
for p in R.glob('history/*.json'):json.load(open(p))
for p in R.glob('*.json'):json.load(open(p))
print('Verified',len(ex),'cells,',len(artifacts),'artifacts,1253 history records; exact question model, budgets, summaries, curves and provenance.')
