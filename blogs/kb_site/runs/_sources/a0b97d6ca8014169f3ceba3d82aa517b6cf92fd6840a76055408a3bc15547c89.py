import json, glob, itertools, collections, math, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def frozen(name):
 return [r for p in sorted(ROOT.glob(name+"_[0-9]*.json")) for r in json.loads(p.read_text())]
lab=frozen("lab")
groups=collections.defaultdict(dict)
for r in lab:
 if r["family"]!="bigram_lm" or r["model_type"]!="transformer_lm" or r["loss"]!={"loss_id":"cross_entropy"} or r.get("excluded") or not math.isfinite(r["mean"]): continue
 key=json.dumps({k:r[k] for k in ["model","optimizer","budget","loss"]},sort_keys=True)
 groups[r["set_id"]][key]=r
pairs={}
for sid,rs in groups.items():
 for a,b in itertools.combinations(rs.values(),2):
  if any(a[k]!=b[k] for k in ["optimizer","budget","loss"]): continue
  diff=[k for k in set(a["model"])|set(b["model"]) if a["model"].get(k)!=b["model"].get(k)]
  if diff!=["d_model"]: continue
  a,b=sorted([a,b],key=lambda x:x["model"]["d_model"])
  key=json.dumps([a["dataset_id"],a["model"],b["model"],a["optimizer"],a["budget"],a["loss"]],sort_keys=True)
  rec={"dataset":a["dataset_id"],"sets":[sid],"optimizer":a["optimizer"]["type"],"widths":[a["model"]["d_model"],b["model"]["d_model"]],"delta":a["optimizer"]["lr"]*a["budget"]["training_steps"],"small_wins":a["mean"]<b["mean"],"ids":[a["candidate_id"],b["candidate_id"]]}
  if key in pairs: pairs[key]["sets"].append(sid)
  else: pairs[key]=rec
(ROOT/"deduplicated_width_pairs.json").write_text(json.dumps(list(pairs.values()),indent=2))
for band in ["low","high"]:
 rows=[r for r in pairs.values() if r["optimizer"] in ["Adam","AdamW","RMSprop"] and ((.02<=r["delta"]<=.1) if band=="low" else r["delta"]>=.5)]
 strata=collections.defaultdict(list)
 for r in rows:strata[(r["optimizer"],tuple(r["widths"]))].append(r)
 print("AUDIT",band,[(k,sum(r["small_wins"] for r in v),len(v),len(set(r["dataset"] for r in v))) for k,v in sorted(strata.items())])
results=[json.loads(p.read_text()) for p in sorted(ROOT.glob("result_[0-9]*.json"))]
contrasts=[]
for dataset in sorted(set(r["dataset_id"] for r in results)):
 rr=[r for r in results if r["dataset_id"]==dataset]
 cells={}
 for r in rr:
  m=r["model"]; label="W" if m["d_model"]==64 else "C" if m["num_layers"]==3 else "B" if m["num_heads"]==2 else "A"
  cells[label]=r
  seeds=np.array([x["final_test_ce"] for x in sorted(r["seed_results"],key=lambda x:x["seed"])])
  assert len(seeds)==10 and np.isfinite(seeds).all()
  assert abs(seeds.mean()-r["mean"])<1e-12
  for f in r["measurement_files"].values():
   p=ROOT.parent/f["repo_path"];assert hashlib.sha256(p.read_bytes()).hexdigest()==f["sha256"]
 for a,b in [("B","A"),("A","C"),("A","W")]:
  va={x["seed"]:x["final_test_ce"] for x in cells[a]["seed_results"]};vb={x["seed"]:x["final_test_ce"] for x in cells[b]["seed_results"]}
  delta=np.array([va[s]-vb[s] for s in sorted(va)])
  mean=float(delta.mean());half=2.2621571627409915*float(delta.std(ddof=1))/math.sqrt(len(delta))
  rec={"dataset":dataset,"contrast":a+"-"+b,"mean":mean,"ci95":[mean-half,mean+half],"seed_sd":float(delta.std(ddof=1)),"wins":int(sum(delta<0)),"n":len(delta)}
  contrasts.append(rec);print("CONTRAST",rec)
(ROOT/"contrasts.json").write_text(json.dumps(contrasts,indent=2))
print("Verified all means and measurement hashes")
