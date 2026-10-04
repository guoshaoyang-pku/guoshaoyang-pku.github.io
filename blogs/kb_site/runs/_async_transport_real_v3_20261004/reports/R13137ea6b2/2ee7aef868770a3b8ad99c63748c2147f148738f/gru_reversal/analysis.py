import json, math, collections, re
from pathlib import Path

ROOT = Path(__file__).parent
rows = json.loads((ROOT / "experiment_observations.json").read_text())
history = json.loads((ROOT / "history_bigram_e0002.json").read_text())
frozen = json.loads((ROOT / "frozen_lab_bigram.json").read_text())
DATASETS = ["bg_1d9caf", "bg_0bf6d7", "bg_167595"]
RATES = [3e-5, 1e-4, 3e-4, .001, .003]

def select(ds, kind="gru_lm", rate=.003, T=1024, B=32, width=64,
           depth=2, opt="AdamW", wd=.001, b2=.95, vocab=None):
    found = []
    for r in rows:
        m, o, b = r["model"], r["optimizer"], r["budget"]
        v = r["dataset"]["vocab_size"] if vocab is None else vocab
        if (r["dataset_id"] == ds and m["type"] == kind
            and m["vocab_size"] == v and m["d_model"] == width
            and m["num_layers"] == depth and o["type"] == opt
            and o["lr"] == rate and o.get("weight_decay", 0) == wd
            and b["training_steps"] == T and b["batch_size"] == B
            and (opt not in ["Adam", "AdamW"] or o["betas"][1] == b2)
            and r["loss"]["loss_id"] == "cross_entropy"):
            found.append(r)
    assert found, (ds, kind, rate, T, B, width, depth, opt, wd, b2, vocab)
    first = found[0]
    assert all(float(x["mean"]) == float(first["mean"]) for x in found)
    return first

def rival(ds, opt, T=1024):
    return select(ds, kind="transformer_lm", rate=.003 if opt=="SGD" else 3e-5,
                  T=T, width=64 if opt=="SGD" else 32,
                  depth=1 if opt=="SGD" else 3, opt=opt,
                  wd=.001 if opt=="SGD" else .0001)

def canonical(r):
    m, o = r["model"], r["optimizer"]
    mk = ["type", "vocab_size", "context_length", "d_model", "num_layers"]
    mk += ["layer_residual"] if m["type"]=="gru_lm" else ["num_heads", "d_ff"]
    ok = ["type", "lr", "weight_decay"]
    ok += ["betas"] if o["type"] in ["Adam", "AdamW"] else (["momentum"] if o["type"]=="SGD" else [])
    return json.dumps([r["dataset_id"], {k:m.get(k) for k in mk},
                       {k:o.get(k,0) for k in ok},r["loss"],r["budget"]],sort_keys=True)

def contrast(a, b):
    aa = [r for r in rows if canonical(r)==canonical(a)]
    bb = [r for r in rows if canonical(r)==canonical(b)]
    common = sorted({r["set_id"] for r in aa} & {r["set_id"] for r in bb})
    assert common, "Never pool different materialized sets"
    a = next(r for r in aa if r["set_id"]==common[0])
    b = next(r for r in bb if r["set_id"]==common[0])
    assert a["dataset"] == b["dataset"]
    effect = float(a["mean"]) - float(b["mean"])
    sa, sb = float(a["std"]), float(b["std"])
    # Worst covariance bound; sqrt(9) also allows population-SD reporting.
    half = 2.262 * (sa + sb) / 3
    return effect, half

print("Frozen bigram candidates/sets/budgets:",
      len(frozen), len({r["set_id"] for r in frozen}),
      dict(collections.Counter(r["budget"]["training_steps"] for r in frozen)))
unique = {}
for r in rows:
    key = canonical(r)
    unique[key] = r
matched = [r for r in unique.values() if r["model"]["vocab_size"] == r["dataset"]["vocab_size"]]
print("Unique matched / oversized-output configurations:", len(matched), len(unique)-len(matched))
print("Materialized experiment sets:", sorted({r["set_id"] for r in rows}))
wins = collections.Counter()
for ds in DATASETS:
    for T in [256, 1024]:
        print("RATE_GRID", ds, T, [(lr, float(select(ds, rate=lr,T=T)["mean"])) for lr in RATES])
        high = select(ds,T=T)
        for opt in ["SGD", "Adagrad"]:
            rr = rival(ds,opt,T)
            d,h = contrast(high,rr)
            wins[T] += d < 0
            print("WEAK_RIVAL", ds,T,opt,"high/rival",high["mean"],rr["mean"],
                  "effect",round(d,6),"conditional_approx_envelope",round(h,6))
    print("SAME_MODEL_1024",ds,contrast(select(ds),select(ds,rate=.0003)))
print("High-rate weak-rival wins by budget:",dict(wins),"denominator=6 each")
ds=DATASETS[0]
base=select(ds)
for kwargs in [dict(wd=0),dict(opt="Adam"),dict(opt="Adam",wd=0),
               dict(wd=.01),dict(b2=.999),dict(width=32),
               dict(depth=1),dict(width=32,depth=1)]:
    r=select(ds,**kwargs)
    print("ABLATION",kwargs,"mean/std",r["mean"],r["std"],
          "vs_SGD",contrast(r,rival(ds,"SGD")),
          "vs_Adagrad",contrast(r,rival(ds,"Adagrad")))
for T,B in [(256,32),(256,128),(1024,8),(1024,32)]:
    print("BUDGET",T,B,"windows",T*B,
          [(lr,select(ds,rate=lr,T=T,B=B)["mean"]) for lr in [.0003,.003]])
for T1,B1,T2,B2 in [(1024,32,256,128),(1024,8,256,32),(1024,32,1024,8)]:
    print("BUDGET_EFFECT",T1,B1,T2,B2,contrast(select(ds,T=T1,B=B1),select(ds,T=T2,B=B2)))
for lr in [.0003,.003]:
    r=select(ds,rate=lr,T=2048)
    print("T2048",lr,r["mean"],r["std"],"SGD",rival(ds,"SGD",2048)["mean"],
          "Adagrad",rival(ds,"Adagrad",2048)["mean"])
print("Nonfinite matched configurations:",
      [(r["dataset_id"],r["model"],r["optimizer"],r["budget"])
       for r in matched if not math.isfinite(float(r["mean"]))])
for qid in ["q_7f2c15","q_f7bc1b","q_7dbb0e"]:
    r=next(x for x in history if x["question_id"]==qid)
    print("HISTORY",qid,r["epoch"],r["source"],r["answer"])
    for choice in r["question"].split("### Choice ")[1:]:
        print(choice[0], re.findall(r"- (?:d_model|num_layers|Optimizer|Learning rate|Weight decay|Betas).*",choice))
