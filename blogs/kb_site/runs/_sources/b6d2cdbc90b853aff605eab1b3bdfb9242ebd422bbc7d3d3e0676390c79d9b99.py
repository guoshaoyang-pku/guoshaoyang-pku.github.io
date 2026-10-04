import csv
import json
import math
from pathlib import Path

ROOT = Path("function_audit")
lab = json.loads((ROOT / "post_experiment_lab.json").read_text())
initial = json.loads((ROOT / "lab.json").read_text())
plans = json.loads((ROOT / "plans.json").read_text())
sets = {"sym_0c2620": ("exp:34aa6df219ff", "exp:9188999e597c"),
        "mvar_000e80": ("exp:46dafe43cf05", "exp:b1ebf9ecda47"),
        "mvar_5a829e": ("exp:630c92c3db03", "exp:72201b637132")}
phase = []
effects = []
for plan in plans:
    did = plan["dataset"]
    old_set, new_set = sets[did]
    base = next(r for r in initial if r["set_id"] + "/" + r["candidate_id"] == plan["base"])
    rows = [r for r in lab if r["dataset_id"] == did and r["set_id"] in (old_set, new_set)]
    fresh = [r for r in rows if r["set_id"] == new_set]
    assert len(fresh) == len(plan["variants"])
    for r in rows:
        assert r["model"] == base["model"]
        assert r["loss"] == base["loss"]
        assert r.get("init", "default") == base.get("init", "default")
        assert {k: v for k, v in r["optimizer"].items() if k != "lr"} == {k: v for k, v in base["optimizer"].items() if k != "lr"}
        assert r["budget"]["batch_size"] == 16
        assert r["budget"]["total_samples_seen"] == r["budget"]["training_steps"] * 16
    for t in (256, 1024, 2048):
        selected = []
        for lr in (.0001, .0003, .001, .003):
            candidates = [r for r in rows if r["optimizer"]["lr"] == lr and r["budget"]["training_steps"] == t]
            assert len(candidates) == 1
            r = candidates[0]
            selected.append(r)
            phase.append(dict(dataset=did, steps=t, lr=lr, delta=lr*t,
                              mean=r["mean"], sd=r["std"], set_id=r["set_id"],
                              candidate_id=r["candidate_id"], failed_seeds=r.get("failed_seeds"),
                              initial_mse=None, train_mse=None, function_step=None,
                              layer_motion=None, gradient_rms=None, late_variance=None,
                              paired_seeds_verified=False))
        for i, a in enumerate(selected):
            for b in selected[i+1:]:
                if a["set_id"] != b["set_id"]:
                    continue
                effects.append(dict(dataset=did, steps=t, set_id=a["set_id"],
                                    low_lr=a["optimizer"]["lr"], high_lr=b["optimizer"]["lr"],
                                    high_minus_low=b["mean"]-a["mean"],
                                    high_over_low=b["mean"]/a["mean"],
                                    high_wins=b["mean"] < a["mean"],
                                    sd_low=a["std"], sd_high=b["std"],
                                    paired_ci=None))
assert len(phase) == 36
assert all(math.isfinite(r["mean"]) and math.isfinite(r["sd"]) for r in phase)
for name, rows in (("phase", phase), ("effects", effects)):
    (ROOT / (name + ".json")).write_text(json.dumps(rows, indent=2))
    with (ROOT / (name + ".csv")).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
for did, (_, new_set) in sets.items():
    es = [r for r in effects if r["dataset"] == did and r["set_id"] == new_set]
    print(did, "fresh higher-rate wins", sum(r["high_wins"] for r in es), "/", len(es))
for r in effects:
    if r["steps"] == 1024 and (r["low_lr"], r["high_lr"]) in ((.0003, .003), (.0001, .001)):
        print("contrast", r)
print("verified 27 new cells, 36 phase cells, controls and same-set effects")
