import collections
import itertools
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text())

def displacement(row):
    opt, steps = row["optimizer"], row["budget"]["training_steps"]
    rate = opt["lr"]
    if opt["type"] == "Adagrad":
        return 2 * rate * math.sqrt(steps)
    if opt["type"] == "SGD":
        momentum = opt.get("momentum", 0)
        scale = 0.003 if row["family"] == "bigram_lm" else (
            0.03 if "regression" in row["family"] and momentum else 0.01)
        return rate * steps * scale / (1 - momentum)
    return rate * steps

def pairs(rows, rate_only):
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get("excluded") or row.get("failed_seeds"):
            continue
        if row["mean"] is None or not math.isfinite(row["mean"]):
            continue
        fields = [row["set_id"], row["model"], row["loss"], row["budget"]]
        if rate_only:
            opt = {k: v for k, v in row["optimizer"].items() if k != "lr"}
            fields.append(opt)
        groups[json.dumps(fields, sort_keys=True)].append(row)
    output = []
    for group in groups.values():
        for a, b in itertools.combinations(group, 2):
            value = (lambda r: r["optimizer"]["lr"]) if rate_only else displacement
            if value(a) > value(b):
                a, b = b, a
            if value(a) == value(b):
                continue
            if not rate_only and displacement(b) < 3 * displacement(a):
                continue
            output.append({
                "family": a["family"], "set": a["set_id"],
                "low_id": a["candidate_id"], "high_id": b["candidate_id"],
                "steps": a["budget"]["training_steps"],
                "batch": a["budget"]["batch_size"],
                "low_value": value(a), "high_value": value(b),
                "low_loss": a["mean"], "high_loss": b["mean"],
                "high_wins": b["mean"] < a["mean"],
                "ratio": b["mean"] / a["mean"] if a["mean"] else None,
                "model": a["model"],
                "low_optimizer": a["optimizer"], "high_optimizer": b["optimizer"],
                "loss": a["loss"]})
    return output

def summarize(items):
    result = {}
    for family in sorted({p["family"] for p in items}):
        ps = [p for p in items if p["family"] == family]
        ratios = [p["ratio"] for p in ps if p["ratio"] is not None]
        result[family] = {
            "high_wins": sum(p["high_wins"] for p in ps),
            "pairs": len(ps), "sets": len({p["set"] for p in ps}),
            "median_ratio": median(ratios) if ratios else None,
            "strata": [
                {"steps": steps, "batch": batch,
                 "high_wins": sum(p["high_wins"] for p in ps
                                  if (p["steps"], p["batch"]) == (steps, batch)),
                 "pairs": sum((p["steps"], p["batch"]) == (steps, batch) for p in ps)}
                for steps, batch in sorted({(p["steps"], p["batch"]) for p in ps})]}
    return result

def main():
    frozen = read("lab_frozen_projection.json")
    experiments = read("own_experiment_rows.json")
    exact = pairs(frozen, True)
    proxy = pairs(frozen, False)
    controlled = pairs(experiments, True)
    output = {
        "initial_candidates": len(frozen),
        "frozen_rate_only": summarize(exact),
        "frozen_proxy": summarize(proxy),
        "experiment_rate_only": summarize(controlled),
        "nonfinite": [{k: r.get(k) for k in
                      ("set_id", "candidate_id", "dataset_id", "model",
                       "optimizer", "budget", "measurement_status")}
                      for r in experiments if r["measurement_status"] != "finite"]}
    for name, data in [("summary.json", output),
                       ("rate_pairs_recomputed.json", exact),
                       ("proxy_pairs_recomputed.json", proxy),
                       ("controlled_rate_pairs.json", controlled)]:
        (ROOT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    assert len(exact) == 3
    assert sum(p["high_wins"] for p in exact) == 3
    print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
