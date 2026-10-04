import json
import numpy as np
from pathlib import Path

results = json.loads(Path("experiment_results.json").read_text())
output = {}
for key, panel in results.items():
    if any(r["excluded"] for r in panel["results"]):
        output[key] = {"status": "excluded", "failed_seeds": [r["failed_seeds"] for r in panel["results"]]}
        continue
    i, j = (0, 1) if key.startswith("r8") else (1, 2)
    a, b = panel["results"][i], panel["results"][j]
    av = np.array([r["final_test_mse"] for r in a["seed_results"]])
    bv = np.array([r["final_test_mse"] for r in b["seed_results"]])
    delta = av - bv
    rng = np.random.default_rng(20261005)
    boot = delta[rng.integers(0, 10, (20000, 10))].mean(axis=1)
    output[key] = {
        "difference": float(av.mean() - bv.mean()),
        "ratio": float(av.mean() / bv.mean()),
        "difference_seed_sd": float(delta.std(ddof=1)),
        "paired_seed_first_wins": int((delta < 0).sum()),
        "bootstrap95": np.quantile(boot, [.025, .975]).tolist(),
    }
Path("recomputed_analysis.json").write_text(json.dumps(output, indent=2))
print(json.dumps(output, indent=2))
