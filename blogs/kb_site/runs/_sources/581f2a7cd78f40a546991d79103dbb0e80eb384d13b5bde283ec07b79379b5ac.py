import glob, hashlib, json, math, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def analyze():
    output = []
    for path in sorted(glob.glob(os.path.join(ROOT, "experiments", "*.json"))):
        e = json.load(open(path))
        if not e.get("why", "").startswith("Preregistered"):
            continue
        rows = e["result"]["results"]
        assert len(rows) == 3
        cells, seeds = {}, {}
        for label, r, spec in zip("ABC", rows, e["input"]["candidates"]):
            assert spec["model"] == {"type":"transformer_lm","vocab_size":32,"context_length":12,"d_model":128,"num_layers":4,"num_heads":2,"d_ff":256}
            assert spec["budget"] == {"training_steps":256,"batch_size":64,"total_samples_seen":16384}
            assert spec["loss"] == {"loss_id":"cross_entropy"}
            seeds[label] = {s["seed"]:s["final_test_ce"] for s in r["seed_results"] if not s["failed"] and math.isfinite(s["final_test_ce"])}
            cells[label] = {k:r[k] for k in ["mean","std","failed_seeds","n_seeds","cached","excluded"]}
            assert len(seeds[label]) == 10 and r["failed_seeds"] == 0
            assert abs(np.mean(list(seeds[label].values())) - r["mean"]) < 1e-10
            for info in r["measurement_files"].values():
                raw = open(os.path.join(ROOT, info["repo_path"]), "rb").read()
                assert hashlib.sha256(raw).hexdigest() == info["sha256"]
        contrasts = {}
        for x,y in [("B","C"),("A","C"),("A","B")]:
            paired = sorted(set(seeds[x]) & set(seeds[y]))
            diff = np.array([seeds[x][s]-seeds[y][s] for s in paired])
            half = 2.2621571627409915 * diff.std(ddof=1) / math.sqrt(len(diff))
            contrasts[x+"-"+y] = {"margin":float(diff.mean()),"ratio_of_means":cells[x]["mean"]/cells[y]["mean"],"paired_ci95":[float(diff.mean()-half),float(diff.mean()+half)],"x_lower_seed_count":int((diff<0).sum()),"n_paired":len(diff)}
        output.append({"dataset":e["result"]["dataset"],"set_id":"exp:"+os.path.basename(path)[:-5],"dataset_params":e["dataset_params"],"cells":cells,"contrasts":contrasts})
    with open(os.path.join(ROOT,"recipe_audit","analysis.json"),"w") as f:
        json.dump(output,f,indent=2)
    print(json.dumps(output,indent=2))
    return output

if __name__ == "__main__":
    analyze()
