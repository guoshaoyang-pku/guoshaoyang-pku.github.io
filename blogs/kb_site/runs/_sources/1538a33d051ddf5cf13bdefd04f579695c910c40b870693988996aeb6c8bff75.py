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
        pi = pi @ p
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
