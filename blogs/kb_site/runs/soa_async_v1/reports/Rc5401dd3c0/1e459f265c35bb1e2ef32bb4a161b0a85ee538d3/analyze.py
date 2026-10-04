import json, itertools, collections
from pathlib import Path

rows = json.loads(Path("lab_snapshot.json").read_text())
def signature(r):
    opt = dict(r["optimizer"])
    opt.pop("lr", None)
    return json.dumps([r["model"], r.get("init", "default"), r["loss"], opt, r["budget"]], sort_keys=True)
pairs = []
groups = collections.defaultdict(list)
for r in rows:
    if not r.get("excluded") and not r.get("failed_seeds"):
        groups[r["set_id"]].append(r)
for sid, group in groups.items():
    for a, b in itertools.combinations(group, 2):
        if signature(a) != signature(b) or a["optimizer"]["lr"] == b["optimizer"]["lr"]:
            continue
        lo, hi = sorted([a, b], key=lambda r: r["optimizer"]["lr"])
        pairs.append(dict(set_id=sid, dataset=lo["dataset_id"], loss=lo["loss"], budget=lo["budget"],
                          low_id=lo["candidate_id"], high_id=hi["candidate_id"],
                          low_mean=lo["mean"], low_sd=lo["std"], high_mean=hi["mean"], high_sd=hi["std"],
                          ratio=hi["mean"]/lo["mean"], margin=hi["mean"]-lo["mean"],
                          source=lo.get("source", "original_lab")))
Path("computed_pairs.json").write_text(json.dumps(pairs, indent=2))
for p in pairs:
    print(json.dumps(p, sort_keys=True))
