import json, math, hashlib, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
T9 = 2.2621571627409915
def seeds(r):
    assert r["n_seeds"] == 10 and r["failed_seeds"] == 0 and not r["excluded"]
    assert [s["seed"] for s in r["seed_results"]] == list(range(10))
    v = np.array([s["final_test_ce"] for s in r["seed_results"]])
    assert np.isfinite(v).all()
    assert abs(v.mean() - r["mean"]) < 1e-10
    return v

def contrast(a, b):
    d = a - b
    margin = float(d.mean())
    half = T9 * float(d.std(ddof=1)) / math.sqrt(10)
    return {"margin": margin, "ratio": float(a.mean()/b.mean()),
            "ci95": [margin-half, margin+half], "a_wins": int((d < 0).sum()),
            "differences": d.tolist()}

records = [json.loads((ROOT / f"exp{i}.json").read_text()) for i in range(1,5)]
scope = [json.loads((ROOT / f"scope{i}.json").read_text()) for i in range(1,3)]
out = {"main": {}, "scope": {}, "counts": {}, "artifacts": []}
names = ["R", "G1", "G3", "W3", "W0"]
for rec in records:
    vals = [seeds(r) for r in rec["results"]]
    out["main"][rec["dataset"]] = {
        "means": dict(zip(names,[float(v.mean()) for v in vals])),
        "sd": dict(zip(names,[float(v.std(ddof=1)) for v in vals])),
        "R_minus": {names[i]:contrast(vals[0],vals[i]) for i in range(1,5)},
        "R_seed_vector": vals[0].tolist()}
for i, rec in enumerate(scope):
    v = [seeds(r) for r in rec["results"]]
    anchor = seeds(records[i]["results"][0])
    # Order: R low-decay h2; R high/low h4; G3/W3 h4; R/G3/W3 T256 b32.
    out["scope"][rec["dataset"]] = {
        "means": dict(zip(["RlowH2","RhighH4","RlowH4","G3H4","W3H4","RT256","G3T256","W3T256"],[float(x.mean()) for x in v])),
        "decay_low_minus_high_h2": contrast(v[0],anchor),
        "decay_low_minus_high_h4": contrast(v[2],v[1]),
        "heads4_minus_heads2_high": contrast(v[1],anchor),
        "heads4_minus_heads2_low": contrast(v[2],v[0]),
        "heads_decay_interaction": contrast(v[2]-v[1],v[0]-anchor),
        "low_R_minus_G3":contrast(v[0],seeds(records[i]["results"][2])),
        "low_R_minus_W3":contrast(v[0],seeds(records[i]["results"][3])),
        "h4_R_minus_G3":contrast(v[1],v[3]),
        "h4_R_minus_W3":contrast(v[1],v[4]),
        "T256_R_minus_G3":contrast(v[5],v[6]),
        "T256_R_minus_W3":contrast(v[5],v[7]),
        "T256_minus_T512_R":contrast(v[5],anchor),
        "R_T256_seed_vector":v[5].tolist()}
unique = set()
all_results = [r for rec in records+scope for r in rec["results"]]
for r in all_results:
    seeds(r)
    spec_file = r["measurement_files"]["candidate_spec.json"]
    source = Path(spec_file["path"]).parent
    spec = json.loads((ROOT.parent/spec_file["repo_path"]).read_text())
    key = (spec["dataset_id"],spec["candidate_id"])
    assert key not in unique
    unique.add(key)
    dst = ROOT / "recipes" / spec["dataset_id"] / spec["candidate_id"]
    dst.mkdir(parents=True,exist_ok=True)
    for name in ["model.py","optimizer.py","loss.py","train.py","candidate_spec.json"]:
        if not (dst/name).exists():
            (dst/name).write_bytes((source/name).read_bytes())
    model = (dst/"model.py").read_text()
    assert 'activation="gelu"' in model or "activation='gelu'" in model
    assert "norm_first=True" not in model
    assert "dropout=0.0" in model
    for name,f in r["measurement_files"].items():
        p = ROOT.parent/f["repo_path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == f["sha256"], str(p)
        if name.endswith("curves.npz"):
            z=np.load(p)
            assert np.allclose(z["curves"][:,-1],seeds(r),atol=1e-10)
            assert z["curves"].shape == (10,spec["budget"]["training_steps"])
            assert int(z["samples"][-1]) == spec["budget"]["total_samples_seen"]
            out["artifacts"].append({"file":str(p.relative_to(ROOT.parent)),"keys":list(z.keys()),"shapes":{k:list(z[k].shape) for k in z.keys()}})
out["counts"] = {"distinct_cells":len(unique),"seed_outcomes":10*len(unique),"datasets":len(records),
                  "cached_cells":sum(bool(r["cached"]) for r in all_results),"failures":sum(r["failed_seeds"] for r in all_results),
                  "excluded_cells":sum(bool(r["excluded"]) for r in all_results)}
manifest=[]
for p in sorted((ROOT.parent/"measurements").iterdir()):
    manifest.append({"path":str(p.relative_to(ROOT.parent)),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"bytes":p.stat().st_size})
out["counts"]["measurement_files"]=len(manifest)
(ROOT/"manifest.json").write_text(json.dumps(manifest,indent=2))
(ROOT/"analysis.json").write_text(json.dumps(out,indent=2))
print(json.dumps(out["counts"],indent=2))
