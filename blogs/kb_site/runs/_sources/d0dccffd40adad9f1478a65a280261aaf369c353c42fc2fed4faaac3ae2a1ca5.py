import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
records = json.loads((ROOT / "transfer_records.json").read_text())
out = {"datasets": {}, "interval_method": "paired t9, two-sided95%, unadjusted", "tcrit": 2.2621571627409915}
for dataset in sorted({r["dataset_id"] for r in records}):
    rs = {{"Adagrad": "A", "Adam": "E", "AdamW": "D"}[r["optimizer"]["type"]]: r for r in records if r["dataset_id"] == dataset}
    values = {}
    result = {"recipes": {}, "contrasts": {}}
    for name, r in rs.items():
        seeds = sorted(r["seed_results"], key=lambda s: s["seed"])
        assert [s["seed"] for s in seeds] == list(range(10))
        assert not any(s["failed"] for s in seeds)
        y = np.array([s["final_test_ce"] for s in seeds])
        values[name] = y
        curve_path = ROOT / r["measurement_files"]["results/curves.npz"]["repo_path"]
        if not curve_path.exists():
            curve_path = Path(r["measurement_files"]["results/curves.npz"]["path"])
        with np.load(curve_path) as f:
            c = f["curves"].copy()
        assert c.shape == (10, 1024)
        assert np.allclose(c[:, -1], y)
        mean_curve = c.mean(axis=0)
        result["recipes"][name] = {
            "mean": float(y.mean()), "sample_sd": float(y.std(ddof=1)),
            "accuracy_mean": float(np.mean([s["final_test_accuracy"] for s in seeds])),
            "ce_at_steps_256_512_768_1024": mean_curve[[255,511,767,1023]].tolist(),
            "minimum_mean_curve_ce": float(mean_curve.min()),
            "minimum_mean_curve_step": int(mean_curve.argmin()+1),
            "late_increase_seeds_512_to_1024": int((c[:,-1] > c[:,511]).sum()),
            "cached": r["cached"], "failures": r["failed_seeds"],
            "candidate_id": r["candidate_id"]
        }
    for a,b in [("A","E"),("A","D"),("E","D")]:
        d = values[a] - values[b]
        half = out["tcrit"] * d.std(ddof=1) / np.sqrt(10)
        result["contrasts"][a+"-"+b] = {
            "mean_difference": float(d.mean()),
            "paired95": [float(d.mean()-half),float(d.mean()+half)],
            "first_wins": int((d<0).sum())
        }
    out["datasets"][dataset] = result
(ROOT / "transfer_analysis.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
