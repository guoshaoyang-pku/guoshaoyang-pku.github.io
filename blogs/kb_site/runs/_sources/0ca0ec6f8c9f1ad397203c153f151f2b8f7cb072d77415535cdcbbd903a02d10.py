import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
artifacts = json.loads((ROOT / "verified_artifacts.json").read_text())
values = {}
rows = []
for a in artifacts:
    row = a["row"]
    seeds = a["summary"]["seed_results"]
    assert [s["seed"] for s in seeds] == list(range(10))
    x = np.array([s["final_test_ce"] for s in seeds])
    assert np.isfinite(x).all()
    values[row["candidate_id"]] = x
    rows.append(dict(candidate_id=row["candidate_id"], vocab=row["model"]["vocab_size"], mean=float(x.mean()), sample_sd=float(x.std(ddof=1)), ci95_half=float(2.262157*x.std(ddof=1)/np.sqrt(10)), n=10, flagged=a["summary"]["failed_seeds"], source=a["path"], hashes=a["source_hashes"]))
contrasts = []
for high, rivals in [("24117cfb",["f5c1fe44","30ea40fd","abe9743b","6afca670"]),("aa379e73",["cbd9a504","2ad8f1cf","5131c830","07c45000"])]:
    for rival in rivals:
        d=values[high]-values[rival]
        contrasts.append(dict(high=high,rival=rival,mean=float(d.mean()),ci95_half=float(2.262157*d.std(ddof=1)/np.sqrt(10)),positive=int((d>0).sum()),n=len(d)))
output=dict(rows=rows,paired_contrasts=contrasts,uncertainty="Pointwise t9 seed intervals conditional on one fixed dataset, not dataset-population intervals.")
(ROOT / "analysis_results.json").write_text(json.dumps(output,indent=2))
print("Verified",len(rows),"rows and",len(contrasts),"paired contrasts")
