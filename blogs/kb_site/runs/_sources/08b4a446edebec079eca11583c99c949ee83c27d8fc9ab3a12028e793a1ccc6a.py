import json, itertools, collections, math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/"lab.json").read_text())
for source in ["base","all"]:
    by=collections.defaultdict(list)
    for r in rows:
        if r["family"]=="bigram_lm" and r["model_type"]=="transformer_lm" and r["loss"].get("loss_id")=="cross_entropy" and math.isfinite(r["mean"]) and not r.get("excluded") and (source=="all" or r.get("source")!="experiment"):
            by[r["set_id"]].append(r)
    for key in ["num_layers","d_ff","num_heads"]:
        out=collections.Counter();details=[]
        for sid,rs in by.items():
            for a,b in itertools.combinations(rs,2):
                if a["model"].get(key)==b["model"].get(key):
                    continue
                if any(a[k]!=b[k] for k in ["budget","loss","optimizer"]):
                    continue
                ma,mb=dict(a["model"]),dict(b["model"]);ma.pop(key,None);mb.pop(key,None)
                if ma!=mb:
                    continue
                lo,hi=sorted([a,b],key=lambda r:r["model"][key]);o=lo["optimizer"];t=lo["budget"]["training_steps"];d=o["lr"]*t
                if o["type"]=="SGD":
                    d=d*.003/(1-o.get("momentum",0))
                elif o["type"]=="Adagrad":
                    d=2*o["lr"]*math.sqrt(t)
                band="<.02" if d<.02 else ".02-.1" if d<.1 else ".1-.5" if d<.5 else ">=.5";group=o["type"]+"/"+band
                out[group+"/pairs"]+=1;out[group+"/larger_wins"]+=hi["mean"]<lo["mean"]
                details.append(dict(set_id=sid,candidate_ids=[lo["candidate_id"],hi["candidate_id"]],delta=d,larger_wins=hi["mean"]<lo["mean"]))
        (ROOT/("audit_"+source+"_"+key+".json")).write_text(json.dumps(dict(counts=dict(out),pairs=details),indent=2))
        print(source,key,dict(out))
