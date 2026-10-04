import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DELTAS = [.01, .02, .0256, .04, .05, .064]
rows = []
for path in sorted((ROOT / "experiments").glob("*.json")):
    exp = json.loads(path.read_text())
    for index, result in enumerate(exp["result"].get("results", [])):
        candidate = result["candidate"]
        model, opt, budget = candidate["model"], candidate["optimizer"], candidate["budget"]
        failed = result.get("failed_seeds")
        mean = result.get("mean")
        rows.append(dict(source=path.name, index=index, dataset=exp["input"]["dataset"],
                         alpha=exp["dataset_params"]["alpha"], model=model["type"],
                         width=model["d_model"], depth=model["num_layers"],
                         T=budget["training_steps"], delta=opt["lr"] * budget["training_steps"],
                         mean=mean, std=result.get("std"), failed=failed,
                         n=None if failed is None else exp["seeds"] - failed,
                         error=result.get("error"), candidate=candidate))

# Duplicate protocol cells are repeated evaluations, not independent replication.
unique = {}
for row in rows:
    key = (row["dataset"], json.dumps(row["candidate"], sort_keys=True))
    if key not in unique or (unique[key]["error"] and not row["error"]) or (unique[key]["n"] is None and row["n"] is not None):
        unique[key] = row

def get(alpha, model, width, depth, T, delta):
    found = [r for r in unique.values() if r["alpha"] == alpha and r["model"] == model
             and r["width"] == width and r["depth"] == depth and r["T"] == T
             and not r["error"] and math.isclose(r["delta"], delta, rel_tol=1e-9)]
    assert len(found) == 1, (alpha, model, width, depth, T, delta, len(found))
    return found[0]

def compare(g, t):
    ratio = g["mean"] / t["mean"]
    # Bonferroni combination of two marginal Student-t intervals: >=95% pointwise
    # coverage under normal iid model-seed losses, without paired covariance.
    # 2.8 exceeds t_9(.9875) and t_8(.9875); n<9 is intentionally excluded.
    interval = None
    if min(g["n"], t["n"]) >= 9:
        eg = 2.8 * g["std"] / math.sqrt(g["n"])
        et = 2.8 * t["std"] / math.sqrt(t["n"])
        interval = [(g["mean"] - eg) / (t["mean"] + et),
                    (g["mean"] + eg) / (t["mean"] - et)]
    return dict(g=g, t=t, ratio=ratio, interval=interval)

grid = []
for alpha in [.8, 1.4]:
    for T in [256, 1024]:
        for delta in DELTAS:
            pair = compare(get(alpha, "gru_lm", 64, 2, T, delta),
                           get(alpha, "transformer_lm", 64, 2, T, delta))
            grid.append(pair)
            print("GRID", alpha, T, delta, "CE", round(pair["g"]["mean"], 6),
                  round(pair["t"]["mean"], 6), "n", pair["g"]["n"], pair["t"]["n"],
                  "R", round(pair["ratio"], 6), "CI", [round(x,6) for x in pair["interval"]])
shapes = []
for alpha, delta in [(.8, .0256), (.8, .256), (1.4, .256)]:
    for width, depth in [(32,2),(64,2),(128,2),(64,1)]:
        pair = compare(get(alpha, "gru_lm", width, depth, 256, delta),
                       get(alpha, "transformer_lm", width, depth, 256, delta))
        shapes.append(pair)
        print("SHAPE", alpha, delta, width, depth, "CE", round(pair["g"]["mean"], 6),
              round(pair["t"]["mean"], 6), "n", pair["g"]["n"], pair["t"]["n"],
              "R", round(pair["ratio"],6), "CI", pair["interval"])
counts = dict(artifact_files=len(list((ROOT/"experiments").glob("*.json"))),
              requested_cells=len(rows), seeded_cells=sum(r["failed"] is not None for r in rows),
              seed_metrics=sum(r["n"] or 0 for r in rows),
              failed_seeds=sum(r["failed"] or 0 for r in rows),
              finite_cells=sum(r["mean"] is not None and math.isfinite(r["mean"]) for r in rows),
              unique_cells=len(unique),
              unique_finite_cells=sum(r["mean"] is not None and math.isfinite(r["mean"]) for r in unique.values()),
              datasets=len({r["dataset"] for r in rows}))
print("COUNTS", counts)
print("GRID WINS", {a:sum(p["ratio"]<1 for p in grid if p["g"]["alpha"]==a) for a in [.8,1.4]})
(ROOT/"audit"/"analysis_results.json").write_text(json.dumps(dict(counts=counts, rows=rows, grid=grid, shapes=shapes), indent=2))

# Reproduce observational counts from the ORIGINAL frozen lab only.
import itertools
lab = json.loads((ROOT/"audit"/"lab.json").read_text())
groups = {}
for row in lab:
    if row["family"] == "bigram_lm" and not row.get("excluded") and math.isfinite(row["mean"]):
        groups.setdefault(row["set_id"], []).append(row)
matched = []
for sid, candidates in groups.items():
    for g, t in itertools.product([x for x in candidates if x["model_type"]=="gru_lm"],
                                  [x for x in candidates if x["model_type"]=="transformer_lm"]):
        if g["optimizer"] != t["optimizer"] or g["budget"] != t["budget"] or g["loss"] != t["loss"]:
            continue
        opt, T = g["optimizer"], g["budget"]["training_steps"]
        delta = (opt["lr"] * T if opt["type"] in ["Adam", "AdamW", "RMSprop"] else
                 2 * opt["lr"] * math.sqrt(T) if opt["type"] == "Adagrad" else
                 opt["lr"] * T * .003 / (1 - opt.get("momentum",0)))
        matched.append(dict(set=sid, delta=delta, gw=g["d_model"], tw=t["d_model"],
                            ratio=g["mean"]/t["mean"], g=g["candidate_id"], t=t["candidate_id"]))
for label, condition in [("gap",lambda x:.02<=x["delta"]<.05),
                         ("equal_or_larger_transformer",lambda x:.05<=x["delta"]<.1 and x["gw"]<=x["tw"]),
                         ("strong",lambda x:.1<=x["delta"]<.5)]:
    pairs = [x for x in matched if condition(x)]
    print("FROZEN",label,len(pairs),"sets",len({x["set"] for x in pairs}),"GRUwins",sum(x["ratio"]<1 for x in pairs))
(ROOT/"audit"/"recomputed_frozen_pairs.json").write_text(json.dumps(matched,indent=2))
