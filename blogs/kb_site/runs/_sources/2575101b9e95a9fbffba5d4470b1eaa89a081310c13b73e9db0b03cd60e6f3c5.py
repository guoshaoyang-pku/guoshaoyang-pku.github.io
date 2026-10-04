import json,math
from pathlib import Path
P=Path('recipe_audit')
rows=json.loads((P/'experiment_rows.json').read_text())
out=[]
for ds in sorted({x['dataset_id'] for x in rows}):
 for t in [256,2048]:
  z=[x for x in rows if x['dataset_id']==ds and x['budget']['training_steps']==t]
  assert len(z)==3
  by={x['activation']:x for x in z};a=by['silu'];c=by['relu']
  assert all(math.isfinite(x['mean']) and math.isfinite(x['std']) for x in z)
  out.append({'dataset':ds,'steps':t,'set_id':a['set_id'],'A':{'mean':a['mean'],'sd':a['std']},'B':{'mean':by['gelu']['mean'],'sd':by['gelu']['std']},'C':{'mean':c['mean'],'sd':c['std']},'margin_A_minus_C':a['mean']-c['mean'],'ratio_C_over_A':c['mean']/a['mean'] if a['mean'] else None,'ratio_status':'finite' if a['mean'] else 'zero_denominator','finite_summary':True,'scheduled_seeds':30,'per_seed_losses_available':False,'failed_seed_count_available':False})
(P/'results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
