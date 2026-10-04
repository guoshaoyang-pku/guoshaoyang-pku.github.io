import json, re, itertools, collections
from pathlib import Path
history = json.loads(Path("frozen_inputs.json").read_text())["history"]
pairs = []
groups = collections.defaultdict(list)
for r in history:
    if r["family"] not in ("multivariate_regression", "synthetic_tabular_classification"):
        continue
    q = r["question"]
    choices = []
    for p in re.split(r"### Choice ", q)[1:]:
        lines = re.findall(r"^- (.*)", p, re.M)
        opt = [s for s in lines if s.startswith(("Optimizer:", "Learning rate:", "Weight decay:", "Betas:", "Momentum:"))]
        model = [s for s in lines if s not in opt and not s.startswith("Loss:")]
        rate = next((s.split(": ", 1)[1] for s in opt if s.startswith("Learning rate:")), None)
        optimizer = next((s for s in opt if s.startswith("Optimizer:")), None)
        if rate:
            choices.append((p[0], model, opt, float(rate), optimizer))
    for a,b in itertools.combinations(choices,2):
        if a[1] != b[1] or a[4] != b[4] or a[3] == b[3]:
            continue
        lo,hi = sorted((a,b),key=lambda c:c[3])
        answer = r["answer"]
        if "<" in answer:
            high_wins = answer.index(hi[0]) < answer.index(lo[0])
        elif answer in (lo[0],hi[0]):
            high_wins = answer == hi[0]
        else:
            continue
        pairs.append(dict(qid=r["question_id"],epoch=r["epoch"],family=r["family"],source=r["source"],
                          lo=lo[0],hi=hi[0],high_wins=high_wins,
                          config_equal_except_lr=[s for s in lo[2] if not s.startswith("Learning rate:")]==[s for s in hi[2] if not s.startswith("Learning rate:")]))
    budget = re.findall(r"^- (?:training_steps|batch_size|total_samples_seen): .*",q,re.M)
    full = sorted((p[0],re.findall(r"^- .*",p,re.M)) for p in re.split(r"### Choice ",q)[1:])
    groups[json.dumps([r["family"],budget,full],sort_keys=True)].append(r)
Path("history_rate_audit.json").write_text(json.dumps(pairs,indent=2))
reversals = [[{k:r.get(k) for k in ("question_id","source","epoch","family","answer")} for r in g] for g in groups.values() if len({r["answer"] for r in g}) > 1]
Path("matched_history_reversals.json").write_text(json.dumps(reversals,indent=2))
print("Ordinal matched groups are outcome-selected, not population reversal frequencies.")
for fam in ("multivariate_regression","synthetic_tabular_classification"):
    p = [p for p in pairs if p["family"]==fam]
    strict = [x for x in p if x["config_equal_except_lr"]]
    print(fam,len(p),sum(x["high_wins"] for x in p),len({x["qid"] for x in p}),len(strict),sum(x["high_wins"] for x in strict))
