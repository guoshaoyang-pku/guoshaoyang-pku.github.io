import hashlib
import json
import re
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(913)
history = json.loads((ROOT / "research/focus_history.json").read_text())
source = {}
for record in history:
    source[record["question_id"]] = {
        label: re.findall(r"```python\n(.*?)```", block, re.S)
        for label, block in re.findall(
            r"### Choice ([ABC])(.*?)(?=### Choice|## Your answer)",
            record["question"], re.S)
    }
assert source["q_9e25f3"] == source["q_9be108"]
results = {}
manifest = {}
for name in ["S1", "S2", "P1", "P2"]:
    obj = json.loads((ROOT / f"research/{name}.json").read_text())
    rows = obj["results"]
    stats = {}
    for label, row in zip("ABC", rows):
        seeds = row["seed_results"]
        assert [s["seed"] for s in seeds] == list(range(10))
        assert not row["excluded"] and row["failed_seeds"] == 0
        ce = np.array([s["final_test_ce"] for s in seeds])
        acc = np.array([s["final_test_accuracy"] for s in seeds])
        assert np.isclose(ce.mean(), row["mean"])
        stats[label] = dict(ce_mean=float(ce.mean()), ce_sd=float(ce.std(ddof=0)),
                            accuracy_mean=float(acc.mean()), accuracy_sd=float(acc.std(ddof=0)),
                            cached=row["cached"], ce_seed_values=ce.tolist(),
                            accuracy_seed_values=acc.tolist())
        for meta in row["measurement_files"].values():
            path = ROOT / meta["repo_path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == meta["sha256"], path
            manifest[meta["repo_path"]] = digest
        spec_meta = row["measurement_files"]["candidate_spec.json"]
        spec = json.loads((ROOT / spec_meta["repo_path"]).read_text())
        assert spec["budget"] == dict(training_steps=2048, batch_size=16, total_samples_seen=32768)
        assert spec["model"] == dict(type="mlp",depth=2,width=192,residual=False,
                    activation="leaky_relu",layer_norm=[True,False],input_dim=16,
                    output_dim=2,leaky_relu_slope=.01)
        files = ROOT / "research/executed" / name / label
        for filename, reference in zip(["model.py","optimizer.py","loss.py"],
                                       source["q_9e25f3"][label]):
            generated = (files / filename).read_text()
            assert reference.strip() in generated, (name,label,filename)
        optimizer = (files / "optimizer.py").read_text()
        assert "momentum" not in optimizer
    contrasts = {}
    for left, right in [("A","B"),("A","C"),("B","C")]:
        diff = np.array(stats[left]["ce_seed_values"]) - stats[right]["ce_seed_values"]
        draws = diff[rng.integers(0,10,size=(10000,10))].mean(axis=1)
        adiff = np.array(stats[left]["accuracy_seed_values"]) - stats[right]["accuracy_seed_values"]
        contrasts[left+"-"+right] = dict(mean=float(diff.mean()),
            ci95=np.quantile(draws,[.025,.975]).tolist(), min=float(diff.min()),max=float(diff.max()),
            left_lower_seeds=int((diff<0).sum()), accuracy_margin=float(adiff.mean()),
            accuracy_left_higher_seeds=int((adiff>0).sum()))
    results[name] = dict(dataset=obj["dataset"],dataset_params=obj["dataset_params"],
                         stats=stats,contrasts=contrasts,
                         order=sorted(stats,key=lambda label:stats[label]["ce_mean"]))
output = dict(results=results,verified_measurement_files=len(manifest),
              total_seed_runs=120,failed_seed_runs=0,
              all_fresh=all(not s["cached"] for o in results.values() for s in o["stats"].values()))
(ROOT / "research/analysis_results.json").write_text(json.dumps(output,indent=2))
(ROOT / "research/measurement_manifest.json").write_text(json.dumps(manifest,indent=2))
print(json.dumps(output,indent=2))
