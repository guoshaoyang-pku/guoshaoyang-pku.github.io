"""Recompute matched CE, error, seed contrasts, and source integrity."""
import json, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = json.loads((ROOT / "rows.json").read_text())
summary = []
contrasts = []
rng = np.random.default_rng(20261005)
groups = {}
for r in rows:
    groups.setdefault(r["condition"], []).append(r)
    for meta in r["measurement_files"].values():
        p = ROOT.parent / meta["repo_path"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == meta["sha256"]
    v = r["variant"]
    seeds = r["seed_results"]
    assert [s["seed"] for s in seeds] == list(range(10))
    assert all(not s["failed"] for s in seeds)
    losses = np.array([s["final_test_ce"] for s in seeds])
    assert np.isclose(losses.mean(), r["mean"], rtol=1e-10, atol=1e-14)
    summary.append({"condition": r["condition"], "optimizer": v["optimizer.type"],
                    "wd": v["optimizer.weight_decay"], "CE": r["mean"], "SD": r["std"],
                    "mean_error": float(np.mean([1-s["final_test_accuracy"] for s in seeds])),
                    "error_seeds": sum(s["final_test_accuracy"] < 1 for s in seeds),
                    "zero_CE_seeds": int(np.sum(losses == 0)), "cached": r["cached"]})
for condition, group in groups.items():
    by = {(r["variant"]["optimizer.type"], r["variant"]["optimizer.weight_decay"]): r for r in group}
    for wd in [1e-5, 1e-4, .001]:
        a = by[("Adam", wd)]
        for reference in [("Adam", 0), ("AdamW", wd)]:
            b = by[reference]
            av = np.array([s["final_test_ce"] for s in a["seed_results"]])
            bv = np.array([s["final_test_ce"] for s in b["seed_results"]])
            diff = av-bv
            boot = diff[rng.integers(0, 10, (20000, 10))].mean(axis=1)
            contrasts.append({"condition": condition, "Adam_wd": wd, "reference": reference,
                              "mean_difference": float(diff.mean()),
                              "paired_difference_SD": float(diff.std(ddof=1)),
                              "bootstrap_95_percentile": np.quantile(boot, [.025,.975]).tolist(),
                              "Adam_higher_seed_count": int(np.sum(diff > 0)),
                              "ratio_of_means": a["mean"]/b["mean"] if b["mean"] > 0 else None,
                              "floor_sensitive_reference": b["mean"] < 1e-7})
result = {"summary": summary, "contrasts": contrasts,
          "conditions": len(groups), "cells": len(rows),
          "scheduled_seed_cells": len(rows)*10,
          "cached_cells": sum(r["cached"] for r in rows),
          "failed_seed_cells": sum(r["failed_seeds"] for r in rows)}
(ROOT / "analysis.json").write_text(json.dumps(result, indent=2))
for condition in groups:
    print(condition)
    for s in summary:
        if s["condition"] == condition:
            print(s["optimizer"], s["wd"], "%.9g" % s["CE"], "SD %.3g" % s["SD"],
                  "err %.6g" % s["mean_error"], "error seeds", s["error_seeds"])
    for c in contrasts:
        if c["condition"] == condition and c["Adam_wd"] == .001:
            print("paired", c)
print("totals", {k:v for k,v in result.items() if k not in ["summary","contrasts"]})
