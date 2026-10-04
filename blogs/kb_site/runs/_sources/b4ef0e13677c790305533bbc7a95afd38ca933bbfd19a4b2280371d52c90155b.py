import json
import numpy as np
from pathlib import Path

def law_summary(d):
    z = np.random.default_rng(d["table_seed"]).standard_normal((d["vocab_size"], d["vocab_size"])) * d["alpha"]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    pi = np.full(d["vocab_size"], 1 / d["vocab_size"])
    for _ in range(256):
        pi = (pi[:, None] * p).sum(axis=0)
        pi /= pi.sum()
    surprisal = -np.log(p)
    h = float(pi @ (p * surprisal).sum(axis=1))
    return {"entropy": h, "concentration": float(pi @ (p*p).sum(axis=1)),
            "surprisal_variance": float(pi @ (p*surprisal**2).sum(axis=1) - h*h)}

def paired_summary(results):
    a,b = results
    aa = {s["seed"]:s["final_test_ce"] for s in a["seed_results"] if not s["failed"]}
    bb = {s["seed"]:s["final_test_ce"] for s in b["seed_results"] if not s["failed"]}
    diffs = np.array([aa[s]-bb[s] for s in sorted(aa.keys() & bb.keys())])
    se = float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
    return {"D":float(diffs.mean()), "paired_sd":float(diffs.std(ddof=1)),
            "descriptive_t95":[float(diffs.mean()-2.262157*se),float(diffs.mean()+2.262157*se)],
            "rms_seed_wins":int((diffs<0).sum()),"adamw_seed_wins":int((diffs>0).sum()),
            "paired_seed_differences":diffs.tolist()}

if __name__ == "__main__":
    ds = json.loads(Path("datasets.json").read_text())
    law = {k:law_summary(v) for k,v in ds.items()}
    Path("law_summaries.json").write_text(json.dumps(law,indent=2))
    raw = json.loads(Path("primary_results.json").read_text())
    stats = {k:paired_summary(v["results"]) for k,v in raw.items()}
    Path("primary_analysis.json").write_text(json.dumps(stats,indent=2))
    print(json.dumps({"law":law,"paired":stats},indent=2))

    extra = json.loads(Path("transfer_and_budget_results.json").read_text())
    extra_stats = {k:paired_summary(v["results"]) for k,v in extra.items()}
    Path("transfer_and_budget_analysis.json").write_text(json.dumps(extra_stats,indent=2))
    import hashlib
    checks = []
    for d in list(raw.values()) + list(extra.values()):
        for r in d["results"]:
            assert r["n_seeds"] == 10 and r["failed_seeds"] == 0 and not r["excluded"]
            for name,f in r["measurement_files"].items():
                assert hashlib.sha256(Path(f["repo_path"]).read_bytes()).hexdigest() == f["sha256"]
            a = np.load(r["measurement_files"]["results/curves.npz"]["repo_path"])
            assert np.array_equal(a["curves"][:,-1], np.array([s["final_test_ce"] for s in r["seed_results"]]))
            checks.append({"dataset":d["dataset"],"steps":r["variant"]["budget.training_steps"],"endpoint_match":True})
    Path("curve_endpoint_checks.json").write_text(json.dumps(checks,indent=2))
    for k in ["262209","389f74"]:
        for j in [0,1]:
            a = np.load(extra[k+"_T256"]["results"][j]["measurement_files"]["results/curves.npz"]["repo_path"])["curves"]
            b = np.load(raw["bg_"+k]["results"][j]["measurement_files"]["results/curves.npz"]["repo_path"])["curves"][:,:256]
            assert np.array_equal(a,b)
    print("Verified 60 file hashes, 20 endpoints, and four budget-prefix pairs.")
