import json, csv, pathlib
ROOT = pathlib.Path(__file__).resolve().parent
data = json.loads((ROOT / "experiments/narrow_adam_raw.json").read_text())
index = {}
for block in data:
    for row in block["results"]:
        v = row["variant"]
        key = (block["dataset"], v["model.width"], v["model.depth"], v["model.layer_norm"][0], v["model.activation"], v["optimizer.lr"], v["budget.training_steps"])
        if row.get("error") is not None:
            raise ValueError(row)
        if key in index and index[key] != row:
            raise ValueError("Repeated specification has inconsistent results")
        index[key] = row
rows = []
for key, deep in sorted(index.items()):
    ds,w,d,ln,a,lr,t = key
    if d != 4:
        continue
    for other_d,other_ln,label in [(1,False,"depth4/shallow0"),(1,True,"depth4/shallowLN"),(4,not ln,"LN-toggle-at-depth4")]:
        comparator = index.get((ds,w,other_d,other_ln,a,lr,t))
        if comparator:
            rows.append(dict(dataset=ds,width=w,deep_ln=ln,activation=a,lr=lr,steps=t,nominal_delta=lr*t,contrast=label,mean=deep["mean"],std=deep["std"],comparator_mean=comparator["mean"],comparator_std=comparator["std"],ratio=deep["mean"]/comparator["mean"]))
with (ROOT / "experiments/narrow_adam_ratios.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"{sum(len(b['results']) for b in data)} returned summaries; {len(index)} unique specifications; {len(rows)} ratios; zero reported errors")
