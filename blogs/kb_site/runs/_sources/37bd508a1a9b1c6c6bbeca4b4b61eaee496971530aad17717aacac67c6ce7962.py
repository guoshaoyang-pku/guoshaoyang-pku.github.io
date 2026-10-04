import json, math, itertools, collections
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text())
def canon(x):
    return json.dumps(x, sort_keys=True)
def nominal(r):
    o = r["optimizer"]
    lr, t = o["lr"], r["budget"]["training_steps"]
    if o["type"] == "SGD":
        return lr*t*.003/(1-o.get("momentum",0))
    if o["type"] == "Adagrad":
        return 2*lr*math.sqrt(t)
    return lr*t
def band(d):
    return "<.02" if d < .02 else ".02-.1" if d < .1 else ".1-.5" if d < .5 else ">=.5"
def audit(rows):
    groups = collections.defaultdict(list)
    for r in rows:
        if r["family"] == "bigram_lm" and r["model_type"] == "transformer_lm" and r["loss"].get("loss_id") == "cross_entropy" and not r.get("excluded",False) and math.isfinite(r["mean"]):
            groups[r["set_id"]].append(r)
    pairs = []
    for sid, cells in groups.items():
        for a,b in itertools.combinations(cells,2):
            if a["d_model"] == b["d_model"]:
                continue
            if canon(a["budget"]) != canon(b["budget"]) or canon(a["loss"]) != canon(b["loss"]):
                continue
            if a["optimizer"]["type"] != b["optimizer"]["type"] or a["optimizer"]["lr"] != b["optimizer"]["lr"]:
                continue
            lo,hi = sorted([a,b], key=lambda r:r["d_model"])
            exact_opt = canon(a["optimizer"]) == canon(b["optimizer"])
            ma,mb = dict(a["model"]),dict(b["model"])
            ma.pop("d_model",None);mb.pop("d_model",None)
            strict = exact_opt and canon(ma) == canon(mb)
            pairs.append(dict(set_id=sid, candidates=[lo["candidate_id"],hi["candidate_id"]], optimizer=lo["optimizer"]["type"], delta=nominal(lo), band=band(nominal(lo)), strict=strict, exact_optimizer=exact_opt, wider_wins=hi["mean"]<lo["mean"], margin=hi["mean"]-lo["mean"], shapes=[[r["d_model"],r["num_layers"],r["num_heads"],r["d_ff"]] for r in [lo,hi]], failures=[lo.get("failed_seeds"),hi.get("failed_seeds")]))
    counts = {}
    for matching in ["shared_type_lr_T","full_optimizer","strict_shape"]:
        selected = [p for p in pairs if matching == "shared_type_lr_T" or (p["exact_optimizer"] if matching == "full_optimizer" else p["strict"])]
        strata = collections.defaultdict(list)
        for p in selected:
            strata[(p["optimizer"],p["band"])].append(p)
        counts[matching] = {"/".join(k):dict(wider_wins=sum(p["wider_wins"] for p in v), pairs=len(v), sets=len(set(p["set_id"] for p in v)), incomplete_or_unknown=sum(any(x!=0 for x in p["failures"]) for p in v)) for k,v in sorted(strata.items())}
    return dict(counts=counts,pairs=pairs)
def contrast(a,b):
    # Positive means b has higher held-out CE; seeds are paired, not batches.
    x = np.array([r["final_test_ce"] for r in a["seed_results"]])
    y = np.array([r["final_test_ce"] for r in b["seed_results"]])
    z=y-x
    se=float(np.std(z,ddof=1)/math.sqrt(len(z)))
    m=float(np.mean(z));h=2.2621571628540993*se
    return dict(margin=m,ratio=b["mean"]/a["mean"],ci95=[m-h,m+h],noise_halfwidth=h,b_wins=int(sum(z<0)),n=len(z))
def main():
    frozen=read("lab.json")
    result={"audits":{}}
    result["audits"]["base_only"]=audit([r for r in frozen if r.get("source")!="experiment"])
    result["audits"]["frozen_including_saved_experiments"]=audit(frozen)
    grids={};norms={}
    all_runs=[read("run%d.json"%i) for i in range(1,11)]
    for run in all_runs:
        ds=run["dataset"].split("/")[-1]
        for r in run["results"]:
            v=r["variant"]; d=v["optimizer.lr"]*256;w=v["model.d_model"];l=v["model.num_layers"]
            if any(abs(d-q)<1e-10 for q in [.01,.05,.1,.3,.768]):
                grids[(ds,round(d,8),l,w)]=r
            else:
                ref=round(d/math.sqrt(64/w),8)
                norms[(ds,ref,l,w)]=r
    result["cells"]=[dict(dataset=ds,delta=d,depth=l,width=w,mean=r["mean"],std=r["std"],failed=r["failed_seeds"]) for (ds,d,l,w),r in sorted(grids.items())]
    result["width_contrasts"]=[]
    for ds in ["bg_313ea4","bg_05e970"]:
        for d in [.01,.05,.1,.3,.768]:
            for l in [1,4]:
                for a,b in [(32,64),(64,128),(32,128)]:
                    result["width_contrasts"].append(dict(dataset=ds,delta=d,depth=l,widths=[a,b],**contrast(grids[(ds,d,l,a)],grids[(ds,d,l,b)])))
    result["normalization"]=[]
    for (ds,ref,l,w),r in sorted(norms.items()):
        result["normalization"].append(dict(dataset=ds,reference_delta=ref,actual_delta=r["variant"]["optimizer.lr"]*256,depth=l,width=w,mean=r["mean"],versus_same_width_unscaled=contrast(grids[(ds,ref,l,w)],r),versus_width64=contrast(grids[(ds,ref,l,64)],r)))
    result["depth_crossings"]=[]
    for ds in ["bg_313ea4","bg_05e970"]:
        for l in [1,2,3,4]:
            result["depth_crossings"].append(dict(dataset=ds,delta=.01,comparison="32/L%d minus 64/L1"%l,**contrast(grids[(ds,.01,1,64)],grids[(ds,.01,l,32)])))
            result["depth_crossings"].append(dict(dataset=ds,delta=.1,comparison="128/L%d minus 64/L4"%l,**contrast(grids[(ds,.1,4,64)],grids[(ds,.1,l,128)])))
    result["execution"]=dict(unique_grid_cells=len(grids),normalization_cells=len(norms),finite_seed_evaluations=sum(r["n_seeds"] for r in list(grids.values())+list(norms.values())),failed_seeds=sum(r["failed_seeds"] for r in list(grids.values())+list(norms.values())),cache_observations=sum(r["cached"] for run in all_runs for r in run["results"]))
    (ROOT/"analysis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result["execution"]))
    print("BASE AUDIT",json.dumps(result["audits"]["base_only"]["counts"]))
    print("DEPTH",json.dumps(result["depth_crossings"]))
    print("NORMALIZATION",json.dumps(result["normalization"]))
if __name__ == "__main__":
    main()
