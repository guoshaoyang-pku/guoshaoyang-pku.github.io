import json, hashlib, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
R = ROOT / "research"
rng = np.random.default_rng(913)
index = rng.integers(0, 10, (10000, 10))

def vector(cell, metric="final_test_ce"):
    rows = sorted(cell["seed_results"], key=lambda x: x["seed"])
    assert len(rows) == 10 and not any(x["failed"] for x in rows)
    return np.array([x[metric] for x in rows])

def contrast(a, b):
    out = {}
    for metric in ["final_test_ce", "final_test_accuracy"]:
        diff = vector(a, metric) - vector(b, metric)
        out[metric] = {"mean": float(diff.mean()), "sd": float(diff.std(ddof=1)),
                       "ci95": np.quantile(diff[index].mean(axis=1), [.025, .975]).tolist(),
                       "negative_seeds": int((diff < 0).sum())}
    return out

def key(c):
    v = c["variant"]
    return (v["model.activation"], "".join(str(int(x)) for x in v["model.layer_norm"]),
            v["model.residual"], v["optimizer.lr"], v["budget.batch_size"])

def lookup(cells, a, m, residual=True, lr=.0001, batch=32):
    return next(x for x in cells if key(x) == (a,m,residual,lr,batch))

out = {"cells": {}, "contrasts": {}, "predictions": [], "audit": {}}
all_cells = []
for label in ["S1","S2","A1","A2"]:
    raw = json.loads((R / ("main_" + label + ".json")).read_text())
    cells = raw["results"]
    all_cells.extend(cells)
    out["cells"][label] = [{"key":key(c), "ce":c["mean"], "ce_sd":c["std"],
        "accuracy":float(vector(c,"final_test_accuracy").mean()),
        "accuracy_sd":float(vector(c,"final_test_accuracy").std(ddof=1)),
        "cached":c["cached"]} for c in cells]
    f = lambda a,m,r=True: lookup(cells,a,m,r)
    comparisons = {
        "act000": (f("silu","000"),f("leaky_relu","000"), .03 if label[0]=="S" else .02),
        "act010": (f("silu","010"),f("leaky_relu","010"), -.035 if label[0]=="S" else .005),
        "bundle": (f("silu","010"),f("leaky_relu","011"), -.02 if label[0]=="S" else .01),
        "count_silu": (f("silu","011"),f("silu","010"), .004 if label[0]=="S" else -.01),
        "count_leaky": (f("leaky_relu","011"),f("leaky_relu","010"), -.012 if label[0]=="S" else -.01),
        "place100": (f("silu","100"),f("silu","010"), .01),
        "place001": (f("silu","001"),f("silu","010"), .02 if label[0]=="S" else .005),
        "residual_silu": (f("silu","010",False),f("silu","010",True), .01),
        "residual_leaky": (f("leaky_relu","010",False),f("leaky_relu","010",True), None)
    }
    out["contrasts"][label] = {}
    for name,(a,b,pred) in comparisons.items():
        val=contrast(a,b);out["contrasts"][label][name]=val
        if pred is not None:
            delta=val["final_test_ce"]["mean"]
            out["predictions"].append({"dataset":label,"test":name,"forecast":pred,
                "observed":delta,"sign_correct": bool(pred*delta>0)})
    print(label, "SiLU CE", "/".join("%.6f"%c["mean"] for c in cells[:5]),
          "Leaky CE", "/".join("%.6f"%c["mean"] for c in cells[5:10]))
    print(label, "SiLU acc", "/".join("%.6f"%vector(c,"final_test_accuracy").mean() for c in cells[:5]),
          "Leaky acc", "/".join("%.6f"%vector(c,"final_test_accuracy").mean() for c in cells[5:10]))
    for name in ["bundle","act010","count_silu","residual_silu"]:
        print(label,name,json.dumps(out["contrasts"][label][name]))
for label in ["S1","A1"]:
    raw=json.loads((R/("neighbors_"+label+".json")).read_text())
    cells=raw["results"];all_cells.extend(cells)
    out["cells"]["neighbors_"+label]=[{"key":key(c),"ce":c["mean"],"ce_sd":c["std"],
         "accuracy":float(vector(c,"final_test_accuracy").mean()),
         "accuracy_sd":float(vector(c,"final_test_accuracy").std(ddof=1)),"cached":c["cached"]} for c in cells]
    for lr,batch,name,pred in [(.0003,32,"rate",-.015 if label=="S1" else -.005),
                              (.0001,16,"batch",-.025 if label=="S1" else .005)]:
        a=lookup(cells,"silu","010",lr=lr,batch=batch)
        b=lookup(cells,"leaky_relu","011",lr=lr,batch=batch)
        val=contrast(a,b)
        out["contrasts"][label][name+"_bundle"]=val
        out["predictions"].append({"dataset":label,"test":name+"_bundle","forecast":pred,
             "observed":val["final_test_ce"]["mean"],
             "sign_correct":bool(pred*val["final_test_ce"]["mean"]>0)})
        print(label,name,"CE",a["mean"],b["mean"],"ACC",vector(a,"final_test_accuracy").mean(),
              vector(b,"final_test_accuracy").mean(),"contrast",json.dumps(val))
    for lr,batch,name in [(.0003,32,"rate"),(.0001,16,"batch")]:
        for activation in ["silu","leaky_relu"]:
            for mask in ["010","011"]:
                a=lookup(cells,activation,mask,lr=lr,batch=batch)
                b=lookup(cells,activation,mask)
                out["contrasts"][label][name+"_"+activation+mask]=contrast(a,b)
        for mask in ["010","011"]:
            out["contrasts"][label][name+"_activation"+mask]=contrast(
                 lookup(cells,"silu",mask,lr=lr,batch=batch),
                 lookup(cells,"leaky_relu",mask,lr=lr,batch=batch))
# Original-set provenance is retained; repeated hashes are not replicates.
inherited=json.loads((R/"inherited.json").read_text())
out["inherited"]=[]
for dataset in ["stabcls_0db74c","stabcls_17011f","stabcls_2173ed","stabcls_28e027"]:
    for depth,mask in [(3,"000"),(3,"100"),(3,"111")]:
        eligible=[c for c in inherited if c["dataset_id"]==dataset and c["depth"]==depth and
                  "".join(str(int(x)) for x in c["layer_norm"])==mask]
        for set_id in sorted(set(c["set_id"] for c in eligible)):
            group=[c for c in eligible if c["set_id"]==set_id]
            acts={c["activation"]:c for c in group}
            if set(acts)=={"silu","leaky_relu"}:
                a,b=acts["silu"],acts["leaky_relu"]
                row={"dataset":dataset,"mask":mask,"set_id":set_id,
                     "ce":[a["mean"],b["mean"]],"ce_sd":[a["std"],b["std"]],
                     "accuracy":[float(vector(a,"final_test_accuracy").mean()),float(vector(b,"final_test_accuracy").mean())],
                     "contrast":contrast(a,b)}
                out["inherited"].append(row);break
    for batch,T in [(64,256),(16,1024)]:
        eligible=[c for c in inherited if c["dataset_id"]==dataset and c["depth"] in [2,4] and
                  c["budget"]["batch_size"]==batch and c["budget"]["training_steps"]==T]
        for set_id in sorted(set(c["set_id"] for c in eligible)):
            group=[c for c in eligible if c["set_id"]==set_id];depths={c["depth"]:c for c in group}
            if set(depths)=={2,4}:
                a,b=depths[2],depths[4]
                out["inherited"].append({"dataset":dataset,"batch":batch,"steps":T,"set_id":set_id,
                    "ce":[a["mean"],b["mean"]],"accuracy":[float(vector(a,"final_test_accuracy").mean()),float(vector(b,"final_test_accuracy").mean())],
                    "contrast_d4_d2":contrast(b,a)});break
paths={}
missing={}
for c in all_cells+inherited:
    for info in c.get("measurement_files",{}).values():
        target=ROOT/info["repo_path"]
        if not target.exists():
            missing[str(target.relative_to(ROOT))]=info
            continue
        actual=hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual==info["sha256"],str(target)
        paths[str(target.relative_to(ROOT))]=actual
out["audit"]={"requested_cells":len(all_cells),"cached_requests":sum(c["cached"] for c in all_cells),
    "unique_requested_conditions":len(set(c["measurement_files"]["results/summary.json"]["sha256"] for c in all_cells)),
    "fresh_requests":sum(not c["cached"] for c in all_cells),"failed_seeds":sum(c["failed_seeds"] for c in all_cells),
    "excluded":sum(c["excluded"] for c in all_cells),"verified_unique_files":len(paths),
    "prediction_sign_correct":sum(p["sign_correct"] for p in out["predictions"]),
    "prediction_count":len(out["predictions"]),"hashes":paths,"inherited_missing_external_files":missing}
(R/"analysis_results.json").write_text(json.dumps(out,indent=2))
print("AUDIT",json.dumps({k:v for k,v in out["audit"].items() if k not in ["hashes","inherited_missing_external_files"]}))
print("FAILURES",json.dumps([p for p in out["predictions"] if not p["sign_correct"]]))
print("INHERITED",json.dumps(out["inherited"]))
