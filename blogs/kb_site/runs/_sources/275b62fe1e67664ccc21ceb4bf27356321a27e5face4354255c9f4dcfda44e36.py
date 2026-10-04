import json
from pathlib import Path

root = Path(__file__).resolve().parent
rows = json.loads((root / "polynomial_rows.json").read_text())
groups = {}
for row in rows:
    if not row["set_id"].startswith("exp:"):
        continue
    key = (row["set_id"], row["dataset_id"], row["budget"]["training_steps"])
    groups.setdefault(key, []).append(row)
results = []
for key, pair in sorted(groups.items()):
    assert len(pair) == 2
    low, high = sorted(pair, key=lambda r: r["optimizer"]["lr"])
    for field in ("dataset", "budget", "model", "loss", "init"):
        assert low[field] == high[field], field
    lo_opt = {k: v for k, v in low["optimizer"].items() if k != "lr"}
    hi_opt = {k: v for k, v in high["optimizer"].items() if k != "lr"}
    assert lo_opt == hi_opt
    n = 10
    mean, sd = high["mean"], high["std"]
    results.append({
        "set_id": key[0], "dataset_id": key[1], "steps": key[2],
        "loss": low["loss"], "low_id": low["candidate_id"],
        "high_id": high["candidate_id"],
        "low_mean": low["mean"], "low_sd": low["std"],
        "high_mean": mean, "high_sd": sd,
        "ratio": mean / low["mean"], "margin": mean - low["mean"],
        "high_cv": sd / mean,
        "conditional_max_lower_bound_ddof0": mean + sd**2 / mean,
        "conditional_max_lower_bound_ddof1": mean + (n-1)/n * sd**2 / mean,
    })
print(json.dumps(results, indent=2))
(root / "computed.json").write_text(json.dumps(results, indent=2))
