import gzip, json, itertools, math
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
lab = json.load(gzip.open(ROOT / "bigram_bias/lab_frozen.json.gz", "rt"))
audit = []
rows = [r for r in lab if r["family"] == "bigram_lm" and r["model_type"] == "transformer_lm" and r["optimizer"]["type"] in ["Adam", "AdamW"]]
for sid, group in itertools.groupby(sorted(rows, key=lambda r: r["set_id"]), key=lambda r: r["set_id"]):
    for a, b in itertools.combinations(list(group), 2):
        a, b = sorted([a, b], key=lambda r: r["optimizer"]["lr"] * r["budget"]["training_steps"])
        da, db = [r["optimizer"]["lr"] * r["budget"]["training_steps"] for r in [a, b]]
        if .03 <= da <= .1 and db >= .5:
            audit.append({"set_id": sid, "low": a, "high": b, "same_model": a["model"] == b["model"], "same_budget": a["budget"] == b["budget"], "same_loss": a["loss"] == b["loss"], "high_minus_low": b["mean"] - a["mean"], "high_over_low": b["mean"] / a["mean"]})
out = {"audit": audit, "experiments": []}
for file in sorted((ROOT / "experiments").glob("*.json")):
    raw = json.loads(file.read_text())
    ds = raw["result"]["dataset"]
    if ds not in ["bigram_lm/bg_206d10", "bigram_lm/bg_2e7010"]:
        continue
    cells = raw["result"]["results"]
    def key(cell):
        c = cell["candidate"]
        return (c["model"]["num_heads"], c["optimizer"]["lr"], c["optimizer"]["betas"][1], c["optimizer"]["weight_decay"])
    by_key = {key(c): c for c in cells}
    def contrast(a, b):
        x, y = by_key[a], by_key[b]
        return {"a": a, "b": b, "b_minus_a": y["mean"] - x["mean"], "b_over_a": y["mean"] / x["mean"]}
    contrasts = []
    for h in [2, 4]:
        for beta, wd in [(.999, .0001), (.95, .0001), (.95, 0)]:
            contrasts.append({"type": "rate", **contrast((h, .00003, beta, wd), (h, .003, beta, wd))})
    for lr in [.00003, .003]:
        for beta, wd in [(.999, .0001), (.95, .0001), (.95, 0)]:
            contrasts.append({"type": "heads", **contrast((2, lr, beta, wd), (4, lr, beta, wd))})
        for h in [2, 4]:
            contrasts.append({"type": "beta2", **contrast((h, lr, .999, .0001), (h, lr, .95, .0001))})
            contrasts.append({"type": "decay", **contrast((h, lr, .95, 0), (h, lr, .95, .0001))})
    ce = contrast((2, .00003, .999, .0001), (4, .003, .95, 0))
    assert len(cells) == 12 and all(math.isfinite(c["mean"]) for c in cells)
    assert all(c["failed_seeds"] == 0 for c in cells) and raw["seeds"] == 10
    out["experiments"].append({"source_file": str(file.relative_to(ROOT)), "dataset": ds, "seeds": raw["seeds"], "C_E": ce, "cells": cells, "contrasts": contrasts})
(ROOT / "bigram_bias/analysis.json").write_text(json.dumps(out, indent=2))


# Per-seed summaries are additional measurement sources, not new training runs.
paired = []
for e in out["experiments"]:
    seed_cells = {}
    for specfile in (ROOT / "bigram_bias/assets" / e["dataset"].split("/")[-1]).glob("*/candidate_spec.json"):
        spec = json.loads(specfile.read_text())
        summary = json.loads((specfile.parent / "results/summary.json").read_text())
        k = (spec["model"]["num_heads"], spec["optimizer"]["lr"], spec["optimizer"]["betas"][1], spec["optimizer"]["weight_decay"])
        values = {s["seed"]: s["final_test_ce"] for s in summary["seed_results"] if not s["failed"]}
        assert len(values) == 10 and summary["failed_seeds"] == 0
        seed_cells[k] = values
    for contrast in [{"type": "C_E", **e["C_E"]}] + e["contrasts"]:
        a, b = tuple(contrast["a"]), tuple(contrast["b"])
        diff = [seed_cells[b][s] - seed_cells[a][s] for s in range(10)]
        mean = sum(diff) / 10
        sd = (sum((v - mean)**2 for v in diff) / 9)**.5
        half = 2.2621571627409915 * sd / 10**.5
        paired.append({"dataset": e["dataset"], **contrast, "paired_sd": sd, "pointwise_t9_95": [mean-half, mean+half], "b_worse_seed_count": sum(v > 0 for v in diff)})
out["paired_contrasts"] = paired
(ROOT / "bigram_bias/analysis.json").write_text(json.dumps(out, indent=2))
for r in paired:
    if r["type"] == "C_E":
        print("PAIRED_C_E", r)
