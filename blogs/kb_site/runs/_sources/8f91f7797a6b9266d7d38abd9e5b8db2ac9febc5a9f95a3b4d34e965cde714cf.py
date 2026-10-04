import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
T9 = 2.2621571627409915

def contrast(a, b):
    sa = {x["seed"]: x["final_test_ce"] for x in a["seed_results"] if not x["failed"]}
    sb = {x["seed"]: x["final_test_ce"] for x in b["seed_results"] if not x["failed"]}
    ids = sorted(sa.keys() & sb.keys())
    d = np.array([sa[i] - sb[i] for i in ids])
    mean = float(d.mean())
    half = T9 * float(d.std(ddof=1)) / np.sqrt(len(d))
    conservative = 2.8 * (a["std"] / np.sqrt(a["n_seeds"]) + b["std"] / np.sqrt(b["n_seeds"]))
    return {"mean": mean, "paired_ci95": [mean-half, mean+half],
            "conservative_ci": [mean-conservative, mean+conservative],
            "n": len(ids), "negative": int((d < 0).sum()), "positive": int((d > 0).sum()),
            "seed_differences": d.tolist()}

def analyze():
    rows = []
    verified = {}
    for group, count in [("primary", 4), ("control", 2)]:
        for i in range(1, count+1):
            d = json.loads((ROOT / f"{group}_{i}.json").read_text())
            rs = d["results"]
            for r in rs:
                assert r["n_seeds"] == 10 and r["failed_seeds"] == 0 and not r["excluded"]
                assert r["source_provenance"]["status"] == "verified"
                assert abs(np.mean([s["final_test_ce"] for s in r["seed_results"]])-r["mean"]) < 1e-12
                for name, meta in r["measurement_files"].items():
                    path = ROOT.parent / meta["repo_path"]
                    raw = path.read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
                    verified[meta["repo_path"]] = meta["sha256"]
                spec = json.loads((ROOT.parent/r["measurement_files"]["candidate_spec.json"]["repo_path"]).read_text())
                assert spec["budget"] == {"training_steps":256,"batch_size":64,"total_samples_seen":16384}
                assert spec["optimizer"] == {"type":"AdamW","lr":.001 if group=="primary" else .0001,
                                              "weight_decay":.001,"betas":[.9,.999]}
                assert spec["loss"]["loss_id"] == "cross_entropy"
                assert spec["model"]["d_model"] == 64
                assert spec["model"]["vocab_size"] == 24 and spec["model"]["context_length"] == 12
                model = (ROOT.parent/r["measurement_files"]["executed/model.py"]["repo_path"]).read_text()
                assert "dropout=0.0" in model
                if spec["model"]["type"] == "transformer_lm":
                    assert spec["model"]["num_heads"] == 2 and spec["model"]["d_ff"] == 256
                    assert "nn.Embedding(12, 64)" in model and 'activation="gelu"' in model
                    assert "norm_first" not in model
                else:
                    assert spec["model"]["num_layers"] == 2 and not spec["model"]["layer_residual"]
            # Only the num_layers literal changes in the executed TF model source.
            tfs = rs[-2:]
            m1 = (ROOT.parent/tfs[0]["measurement_files"]["executed/model.py"]["repo_path"]).read_text()
            m2 = (ROOT.parent/tfs[1]["measurement_files"]["executed/model.py"]["repo_path"]).read_text()
            assert m1.replace("num_layers=1", "num_layers=2") == m2
            row = {"group":group,"dataset":d["dataset"],"metadata":d["dataset_params"],
                   "means":[r["mean"] for r in rs],"stds":[r["std"] for r in rs],
                   "H":contrast(rs[-1], rs[-2]),"failures":sum(r["failed_seeds"] for r in rs),
                   "cached":sum(bool(r["cached"]) for r in rs)}
            if group == "primary":
                row["G"] = contrast(rs[0],rs[1])
            rows.append(row)
    for row in rows[4:]:
        orig = next(x for x in rows[:4] if x["dataset"] == row["dataset"])
        # Difference of depth effects using identical seed labels.
        delta = np.array(row["H"]["seed_differences"]) - np.array(orig["H"]["seed_differences"])
        half = T9*delta.std(ddof=1)/np.sqrt(10)
        row["rate_by_depth"] = {"mean":float(delta.mean()),"paired_ci95":[float(delta.mean()-half),float(delta.mean()+half)]}
    result = {"rows": rows, "verified_artifacts": len(verified), "hashes": verified,
              "scheduled_seed_slots":160,"finite_seeds":160,"failed_seeds":0}
    (ROOT/"results.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!="hashes"},indent=2))

if __name__ == "__main__":
    analyze()
