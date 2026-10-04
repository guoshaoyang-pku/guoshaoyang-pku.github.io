import glob, json, os, hashlib
import numpy as np

def paired(a, b, metric="final_test_ce"):
    sa = {x["seed"]: x for x in a["seed_results"] if not x["failed"]}
    sb = {x["seed"]: x for x in b["seed_results"] if not x["failed"]}
    seeds = sorted(set(sa) & set(sb))
    delta = np.array([sa[s][metric] - sb[s][metric] for s in seeds])
    rng = np.random.default_rng(913)
    boots = delta[rng.integers(0, len(seeds), (10000, len(seeds)))].mean(1)
    return dict(seeds=seeds, mean_difference=float(delta.mean()),
                bootstrap95=np.quantile(boots, [.025, .975]).tolist(),
                wins=int((delta < 0).sum()),
                seed_differences=delta.tolist())

def signature(v):
    return json.dumps(v, sort_keys=True)

files = sorted(set(glob.glob("research/factorial_*.json") +
                   glob.glob("research/progress_*.json") +
                   glob.glob("research/controls_*.json") +
                   glob.glob("research/batch_*.json")))
table, comparisons, provenance, unique = [], [], {}, {}
for f in files:
    obj = json.load(open(f))
    results = obj["results"]
    for r in results:
        if r.get("error") or r.get("excluded"):
            raise ValueError((f, r))
        v = r["variant"]
        spec = r["measurement_files"]["candidate_spec.json"]["sha256"]
        unique[spec] = r
        row = dict(file=f, dataset=obj["dataset"], variant=v, mean=r["mean"],
                   SD=r["std"], cached=r["cached"], seeds=r["seed_results"],
                   test_accuracy=float(np.mean([s["final_test_accuracy"] for s in r["seed_results"]])),
                   measurement_files=r["measurement_files"])
        table.append(row)
        for kind, info in r["measurement_files"].items():
            path = info["repo_path"] if os.path.exists(info["repo_path"]) else info["path"]
            digest = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if digest != info["sha256"]:
                raise ValueError(("hash mismatch", path))
            provenance[digest] = dict(kind=kind, source=info)
    # Comparisons never cross a returned experiment set.
    for i, a in enumerate(results):
        for b in results[i+1:]:
            va, vb = a["variant"], b["variant"]
            changes = [k for k in sorted(set(va) | set(vb)) if va.get(k) != vb.get(k)]
            depth_only = set(changes) <= {"model.depth", "model.layer_norm"} and "model.depth" in changes
            factor_only = len(changes) == 1
            batch_progress = set(changes) <= {"budget.training_steps", "budget.batch_size", "optimizer.lr"}
            if depth_only or factor_only or batch_progress:
                comparisons.append(dict(file=f, dataset=obj["dataset"], a=va, b=vb,
                                        changes=changes, paired_ce=paired(a, b),
                                        paired_accuracy=paired(a, b, "final_test_accuracy")))
out = dict(cells=table, comparisons=comparisons, provenance=provenance,
           unique_conditions=len(unique), reported_cells=len(table),
           failed_seeds=sum(r.get("failed_seeds",0) for r in unique.values()),
           uncertainty="Paired-seed percentile bootstrap, 10000 draws, rng913. Descriptive conditional on these fixed datasets; not dataset-population inference, not multiplicity corrected.",
           unmeasured=["train CE", "per-point margins", "wrong-point confidence", "temperature intervention", "distractor-removal intervention"])
json.dump(out, open("research/analysis_results.json", "w"), indent=2)
print("Verified", len(provenance), "measurement hashes;", len(unique), "unique conditions;", len(table), "reported cells;", len(comparisons), "within-set comparisons.")
