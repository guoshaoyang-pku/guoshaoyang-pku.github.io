import ast
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
p = json.loads((ROOT / "corrected_experiments.json").read_text())
stats = {}
verified = {}
for key, panel in p.items():
    rows = dict(zip(panel["labels"], panel["response"]["results"]))
    seeds = {}
    for label, row in rows.items():
        assert row["failed_seeds"] == 0 and not row["excluded"]
        assert [s["seed"] for s in row["seed_results"]] == list(range(10))
        seeds[label] = np.array([s["final_test_mse"] for s in row["seed_results"]])
        assert np.isfinite(seeds[label]).all()
        assert abs(seeds[label].mean() - row["mean"]) < 1e-12
        for name, source in row["measurement_files"].items():
            path = REPO / source["repo_path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"]
            verified[str(path)] = source["sha256"]
        source = (REPO / row["measurement_files"]["executed/model.py"]["repo_path"]).read_text()
        if row["variant"]["model.activation"] == "leaky_relu":
            assert "nn.LeakyReLU(0.01)" in source and "nn.LeakyReLU(0.1)" not in source
        curve = np.load(REPO / row["measurement_files"]["results/curves.npz"]["repo_path"])
        T = row["variant"]["budget.training_steps"]
        assert curve["curves"].shape == (10, T)
        assert np.allclose(curve["curves"][:, -1], seeds[label], rtol=0, atol=1e-12)
        opt = (REPO / row["measurement_files"]["executed/optimizer.py"]["repo_path"]).read_text()
        assert "lr=0.001" in opt and "momentum=0.9" in opt and "weight_decay=0" in opt
    pairs = []
    if key.startswith("7"):
        pairs = [("B","E"),("C","E"),("C","m1PL"),("D","E"),("D","B"),("D","C")]
        cell = lambda d,r,a: "C" if (d,r,a)==(3,"R","G") else f"m{d}{r}{a}"
        for d in [1,3]:
            for a in ["G","L"]: pairs.append((cell(d,"R",a),cell(d,"P",a)))
            for r in ["P","R"]: pairs.append((cell(d,r,"L"),cell(d,r,"G")))
        for r in ["P","R"]:
            for a in ["G","L"]: pairs.append((cell(3,r,a),cell(1,r,a)))
    else:
        pairs = [("C","A"),("C","C_plain"),("C_leaky","C"),("A","A_silu"),("deep64_leaky","A"),("deep64_silu","A_silu")]
    contrasts = {}
    for a,b in pairs:
        diff = seeds[a]-seeds[b]
        half = 2.2621571627409915 * diff.std(ddof=1) / math.sqrt(10)
        contrasts[a+"-"+b] = {"difference":float(diff.mean()),"ci95":[float(diff.mean()-half),float(diff.mean()+half)],"wins":int((diff<0).sum()),"n":10}
    stats[key] = {"means":{a:r["mean"] for a,r in rows.items()},"full_order":"<".join(sorted("ABCDE",key=lambda a:rows[a]["mean"])),"contrasts":contrasts,"cached":sum(r["cached"] for r in rows.values()),"failures":sum(r["failed_seeds"] for r in rows.values())}
(ROOT / "statistics.json").write_text(json.dumps(stats,indent=2))
(ROOT / "verified_hashes.json").write_text(json.dumps(verified,indent=2))
print("Verified",len(verified),"unique archived artifact hashes; 44 cells / 440 valid endpoint seeds")
for key,x in stats.items():
    print(key, x["full_order"], "cached", x["cached"])
    for pair,c in x["contrasts"].items():
        print(pair, "%.9f"%c["difference"], ["%.9f"%v for v in c["ci95"]], str(c["wins"])+"/10")

# Historical and executed class bodies must match, excluding documentation.
import re
class StripDocs(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

def class_ast(source):
    tree = StripDocs().visit(ast.parse(source))
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    return ast.dump(tree, include_attributes=False)

audit = {}
for q, key in [("q_7cbcd0", "7X"), ("q_4c7ce8", "4X")]:
    record = json.loads((ROOT / (q + ".json")).read_text())
    rows = dict(zip(p[key]["labels"], p[key]["response"]["results"]))
    audit[q] = {}
    for label, section in re.findall(r"### Choice ([A-E])(.*?)(?=### Choice|## Your answer|$)", record["question"], re.S):
        historical = re.search(r"\*\*Model code\*\*.*?```python\n(.*?)```", section, re.S).group(1)
        executed = (REPO / rows[label]["measurement_files"]["executed/model.py"]["repo_path"]).read_text()
        equal = class_ast(historical) == class_ast(executed)
        assert equal, (q, label)
        audit[q][label] = {"class_ast_equal":equal,"historical_model_sha256":hashlib.sha256(historical.encode()).hexdigest(),"executed_model_sha256":rows[label]["measurement_files"]["executed/model.py"]["sha256"]}
(ROOT / "exact_recipe_audit.json").write_text(json.dumps(audit,indent=2))
print("All ten focal class ASTs match historical code.")
