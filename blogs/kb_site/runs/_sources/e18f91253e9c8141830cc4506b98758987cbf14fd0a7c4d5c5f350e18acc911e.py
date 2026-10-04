import json
from pathlib import Path
import hashlib
import numpy as np
raw = json.loads(Path("recipe_results.json").read_text())
out = []
def contrast(a, b):
    delta = b-a
    half = 2.2621571628540993*delta.std(ddof=1)/np.sqrt(len(delta))
    return {"margin_b_minus_a":float(delta.mean()),"paired_t95":[float(delta.mean()-half),float(delta.mean()+half)],"a_wins":int((delta>0).sum()),"b_wins":int((delta<0).sum()),"ties":int((delta==0).sum()),"a_loss_seeds":np.where(delta<0)[0].tolist()}
for data in raw:
    arr=np.array([[s["final_test_ce"] for s in r["seed_results"]] for r in data["results"]])
    for r in data["results"]:
        for f in r["measurement_files"].values():
            p=Path(f["repo_path"])
            assert hashlib.sha256(p.read_bytes()).hexdigest()==f["sha256"]
    out.append({"dataset":data["dataset"],"means":arr.mean(axis=1).tolist(),"sample_sds":arr.std(axis=1,ddof=1).tolist(),"pairs":{str((i,j)):contrast(arr[i],arr[j]) for i in range(len(arr)) for j in range(i+1,len(arr))},"full_order":int(((arr[0]<arr[1])&(arr[1]<arr[2])).sum()) if len(arr)==3 else None})
# The matched factorial reuses B/C from primary dataset 3; no new replication.
primary=raw[2]["results"]
added=raw[5]["results"]
arr=np.array([[s["final_test_ce"] for s in r["seed_results"]] for r in [added[0],primary[1],primary[2],added[1]]])
factorial={"order":["plain110","residual110","plain111","residual111"],"pairs":{str((i,j)):contrast(arr[i],arr[j]) for i,j in [(0,2),(0,1),(2,3),(1,3)]}}
ln_plain=arr[0]-arr[2]
ln_residual=arr[1]-arr[3]
factorial["ln_improvement_plain_minus_residual"]=contrast(ln_residual,ln_plain)
Path("audit_statistics.json").write_text(json.dumps({"datasets":out,"factorial":factorial},indent=2))
print(json.dumps({"datasets":out,"factorial":factorial},indent=2))
