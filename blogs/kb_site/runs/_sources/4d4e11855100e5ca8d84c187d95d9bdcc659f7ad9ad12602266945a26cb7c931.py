import json, hashlib, os, csv
import numpy as np
data=json.load(open("transfer_raw.json"))
rng=np.random.default_rng(20261005)
audit=[]
rows=[]
def seeds(r):
 return np.array([x["final_test_ce"] for x in r["seed_results"]])
def interval(a,b):
 dif=a[rng.integers(0,len(a),(20000,len(a)))].mean(1)-b[rng.integers(0,len(b),(20000,len(b)))].mean(1)
 return np.quantile(dif,[.025,.975]).tolist()
for kind in ["direct","components"]:
 for batch in data[kind]:
  for i,r in enumerate(batch["results"]):
   p=os.path.dirname(r["measurement_files"]["candidate_spec.json"]["path"])
   source={}
   for f in ["model.py","optimizer.py","loss.py","train.py"]:
    content=open(p+"/"+f).read()
    sha=hashlib.sha256(content.encode()).hexdigest()
    source[f]=sha
    os.makedirs("sources",exist_ok=True)
    open("sources/"+sha+"_"+f,"w").write(content)
   v=r["variant"]; m=open(p+"/model.py").read(); opt=open(p+"/optimizer.py").read()
   assert m.count("MLPBlock(width=")==v["model.depth"]
   assert m.count("use_layer_norm=True)")==sum(v["model.layer_norm"])
   assert ("h = h + x" in m)==v["model.residual"]
   assert "lr="+str(v["optimizer.lr"]) in opt
   assert np.isclose(seeds(r).mean(),r["mean"],rtol=1e-12,atol=1e-16)
   assert r["n_seeds"]==10 and r["base_seed"]==0
   for f,meta in r["measurement_files"].items():
    assert hashlib.sha256(open(meta["repo_path"],"rb").read()).hexdigest()==meta["sha256"]
   audit.append({"kind":kind,"dataset":batch["dataset"],"cell":i,"source_dir":p,"source_sha256":source,"measurement_files":r["measurement_files"]})
for batch in data["direct"]:
 for i,t in [(0,256),(2,512)]:
  s,d=batch["results"][i:i+2]; a,b=seeds(s),seeds(d)
  ci=interval(a,b)
  rows.append({"dataset":batch["dataset"].split("/")[-1],"turns":batch["dataset_params"]["spiral_turns"],"T":t,"S_mean":s["mean"],"S_sd":s["std"],"D_mean":d["mean"],"D_sd":d["std"],"margin":s["mean"]-d["mean"],"ratio":s["mean"]/d["mean"],"ci95":ci,"S_seed_label_wins":int((a<b).sum())})
 # Authenticate deterministic horizon prefix using measured curves, not independent repetitions.
 for i in [0,1]:
  short=batch["results"][i]; long=batch["results"][i+2]
  c=np.load(long["measurement_files"]["results/curves.npz"]["repo_path"])["curves"]
  assert np.array_equal(c[:,255],seeds(short))
component_rows=[]
for batch in data["components"]:
 parent=next(x for x in data["direct"] if x["dataset"]==batch["dataset"])
 for i,r in enumerate(batch["results"]):
  base=parent["results"][2 if i<2 else 3]
  component_rows.append({"dataset":batch["dataset"],"variant":["S_lr.0003","S_d5","D_lr.001","D_d2"][i],"mean":r["mean"],"sd":r["std"],"margin_to_parent":r["mean"]-base["mean"],"ci95":interval(seeds(r),seeds(base))})
json.dump({"direct":rows,"components":component_rows,"audit":audit},open("transfer_analysis.json","w"),indent=2)
with open("transfer_cells.csv","w",newline="") as f:
 writer=csv.writer(f);writer.writerow(["kind","dataset","cell","seed","test_ce","failed","cached","excluded"])
 for kind in ["direct","components"]:
  for batch in data[kind]:
   for i,r in enumerate(batch["results"]):
    for s in r["seed_results"]: writer.writerow([kind,batch["dataset"],i,s["seed"],s["final_test_ce"],s["failed"],r["cached"],r["excluded"]])
print(json.dumps({"direct":rows,"components":component_rows,"audit_cells":len(audit)},indent=2))
