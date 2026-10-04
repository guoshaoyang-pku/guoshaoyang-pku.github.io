import collections, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
lab = json.loads((ROOT / "lab.json").read_text())
groups = collections.defaultdict(list)
for row in lab:
    groups[row["set_id"]].append(row)
def rate(row, b):
    o = row["optimizer"]
    return o["type"] == ("SGD" if b else "AdamW") and o["lr"] == (.0003 if b else .001)
def arch(row, b, exact=False):
    return (row["activation"] == "gelu" and row["residual"] == (not b)
        and row["depth"] == (4 if b else 5)
        and (row["width"] == (128 if b else 96) if exact else True)
        and (row["layer_norm"] == ([True,True,False,True] if b else [False,True,True,True,True])
             if exact else row["n_ln"] >= 1))
def full(row,b):
    o=row["optimizer"]
    return (arch(row,b,True) and row.get("init","default")=="default"
        and o.get("weight_decay",0)==(0 if b else .0001)
        and (o.get("momentum",0)==.9 if b else o.get("betas")==[.9,.95])
        and row["loss"]=={"loss_id":"mse"})
backtest={}
for scope in ["rate","deep_normalized","gelu_residual_direction","exact_arch","full","full_long"]:
    pairs=[]
    for sid,rows in groups.items():
        for b in [r for r in rows if rate(r,True)]:
            for c in [r for r in rows if rate(r,False)]:
                if b["budget"] != c["budget"] or b.get("excluded") or c.get("excluded"): continue
                if scope=="deep_normalized" and not all(r["depth"]>=4 and r["n_ln"]>=1 for r in [b,c]):continue
                if scope=="gelu_residual_direction" and not (arch(b,True) and arch(c,False)):continue
                if scope=="exact_arch" and not (arch(b,True,True) and arch(c,False,True)):continue
                if scope in ["full","full_long"] and not (full(b,True) and full(c,False)):continue
                if scope=="full_long" and b["budget"]!={"training_steps":1024,"batch_size":32,"total_samples_seen":32768}:continue
                pairs.append({"set_id":sid,"B":b["candidate_id"],"C":c["candidate_id"],"Bmean":b["mean"],"Cmean":c["mean"],"ratio":b["mean"]/c["mean"],"Bfailed":b.get("failed_seeds"),"Cfailed":c.get("failed_seeds"),"budget":b["budget"]})
    backtest[scope]=pairs
assert backtest==json.loads((ROOT/"backtest.json").read_text())
newrows=json.loads((ROOT/"new_lab_rows.json").read_text())
effects=[]; specs=[]; moments=[]
rng=np.random.default_rng(20261005)
x=rng.uniform(size=(100000,5))
ys={
"mvar_05883d":x[:,1]*x[:,2]-4*x[:,1]**2+np.cos(2*np.pi*np.sin(2*np.pi*x[:,0]))+np.sin(2*np.pi*x[:,3]**2)-2*np.sin(2*np.pi*x[:,2])+np.tanh(2*x[:,3])+2*x[:,4]+.5,
"mvar_06497c":np.cos(2*np.pi*np.sin(2*np.pi*x[:,1]))**2+np.cos(2*np.pi*x[:,2])*np.sin(2*np.pi*x[:,0])+x[:,0]**2+np.cos(2*np.pi*x[:,2])+np.sin(2*np.pi*x[:,3])+np.sin(2*np.pi*x[:,4])+x[:,2]-.25*x[:,4]-1.5,
"mvar_0f2210":2*x[:,1]**3+x[:,4]**3+x[:,0]*x[:,2]+x[:,3]**2+.5*x[:,0]-2*x[:,2]+1}
for did,y in ys.items():
    moments.append({"dataset":did,"method":"100000 NumPy uniform population probes; NOT fixed Torch split",
        "probe_seed":20261005,"mean":float(y.mean()),"variance":float(y.var()),"mean_mc_se":float(y.std()/np.sqrt(len(y)))})
for fname in ["exp058","exp064","exp0f"]:
    e=json.loads((ROOT/(fname+".json")).read_text());did=e["dataset"].split("/")[-1]
    for r in e["results"]:
        assert not r["cached"] and r["failed_seeds"]==0 and not r["excluded"] and r["n_seeds"]==10
        for name,item in r["measurement_files"].items():
            p=ROOT.parent/item["repo_path"]
            assert hashlib.sha256(p.read_bytes()).hexdigest()==item["sha256"]
        spec=json.loads((ROOT.parent/r["measurement_files"]["candidate_spec.json"]["repo_path"]).read_text())
        for key,value in r["variant"].items():
            parent,field=key.split(".");assert spec[parent][field]==value
        specs.append(spec)
    for i in [0,2]:
        b,c=e["results"][i:i+2]
        bs={r["seed"]:r["final_test_mse"] for r in b["seed_results"]}
        cs={r["seed"]:r["final_test_mse"] for r in c["seed_results"]}
        assert set(bs)==set(cs)==set(range(10))
        d=np.array([bs[s]-cs[s] for s in range(10)])
        matches=[r for r in newrows if r["dataset_id"]==did and r["budget"]["training_steps"]==b["variant"]["budget.training_steps"]]
        assert len(matches)==2 and len({r["set_id"] for r in matches})==1
        ci=2.262157*d.std(ddof=1)/np.sqrt(10)
        effects.append({"dataset":did,"set_id":matches[0]["set_id"],"T":b["variant"]["budget.training_steps"],
          "Bmean":b["mean"],"Cmean":c["mean"],"Bsd":b["std"],"Csd":c["std"],
          "BminusC":float(d.mean()),"BoverC":b["mean"]/c["mean"],"B_seed_wins":int(sum(d<0)),
          "paired_difference_sd":float(d.std(ddof=1)),"paired_t95":[float(d.mean()-ci),float(d.mean()+ci)],
          "seed_differences":d.tolist()})
for name,obj in [("effects",effects),("resolved_specs",specs),("population_moments",moments)]:
    (ROOT/(name+".json")).write_text(json.dumps(obj,indent=2))
print(json.dumps({"effects":effects,"population_moments":moments},indent=2))
