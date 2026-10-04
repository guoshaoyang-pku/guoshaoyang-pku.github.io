import json
from pathlib import Path
import numpy as np
rows=json.loads(Path("study/recipe_results.json").read_text()); out=[]
for i,j in [(0,1),(0,5),(6,1),(5,1),(0,6),(2,7),(1,2),(0,2)]:
 a,b=rows[i]["result"],rows[j]["result"]
 assert a["failed_seeds"]==b["failed_seeds"]==0
 va={s["seed"]:s["final_test_ce"] for s in a["seed_results"]}; vb={s["seed"]:s["final_test_ce"] for s in b["seed_results"]}
 assert va.keys()==vb.keys()
 d=np.array([va[k]-vb[k] for k in sorted(va)]); se=d.std(ddof=1)/np.sqrt(len(d))
 out.append(dict(a=i,b=j,n=len(d),margin=float(d.mean()),ci95=[float(d.mean()-2.262157*se),float(d.mean()+2.262157*se)],wins=int((d<0).sum())))
Path("study/recipe_contrasts.json").write_text(json.dumps(out,indent=2))
print(out)
