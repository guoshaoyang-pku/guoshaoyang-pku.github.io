import json
import os,sys
sys.path.insert(0,os.getcwd())
from audit import audit
h=json.load(open("history_sources.json"))
l=json.load(open("lab_sources.json"))
for mode in ["core","linear"]:
    a=audit(h,l,mode)
    saved=json.load(open("audit_"+mode+".json"))
    assert a==saved
    narrow=[x for x in a["history_depth"] if x["q"]!="q_879733" and x["deep"]["depth"]<=3]
    assert len(narrow)==4 and sum(x["deeper_win"] for x in narrow)==2
    assert len(a["lab_depth"])==6 and sum(x["deeper_win"] for x in a["lab_depth"])==2
    assert not a["lab_zero"]
r=json.load(open("controlled_results.json"))
assert len(r)==4 and len({x["set_id"] for x in r})==2
for x in r:
    assert x["budget"]==dict(training_steps=256,batch_size=64,total_samples_seen=16384)
    assert x["optimizer"]==dict(type="AdamW",lr=.0003,betas=[.9,.999],weight_decay=1e-5)
    assert x["init"]=="default" and x["model"]["init"]=="default"
    assert x["model"]["activation"]=="leaky_relu" and x["model"]["leaky_relu_slope"]==.01
    assert x["model"]["residual"] and x["loss"]["loss_id"]=="cross_entropy"
    assert (x["depth"],x["width"],x["layer_norm"]) in [(3,64,[True]*3),(1,128,[True])]
for did in {x["dataset_id"] for x in r}:
    A=next(x for x in r if x["dataset_id"]==did and x["depth"]==3)
    B=next(x for x in r if x["dataset_id"]==did and x["depth"]==1)
    assert A["mean"]<B["mean"]
print("Verified frozen pair audits and four persisted recipe results; no retraining.")
