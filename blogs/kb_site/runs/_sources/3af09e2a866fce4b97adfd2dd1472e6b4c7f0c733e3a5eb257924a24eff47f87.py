import json, math
from pathlib import Path
import numpy as np
rows=[]
for p in Path("experiments").glob("*.json"):
 x=json.loads(p.read_text()); result=x.get("result",{})
 if "results" not in result: continue
 for r, spec in zip(result["results"],x["input"]["candidates"]):
  v={section+"."+key:value for section in ["model","optimizer","loss","budget"] for key,value in spec[section].items()}
  if v.get("model.width")!=32 or v.get("model.depth")!=3: continue
  t=v["budget.training_steps"]; opt=v["optimizer.type"]; lr=v["optimizer.lr"]; mom=v.get("optimizer.momentum",0)
  family=result["dataset"].split("/")[0]
  ratio=1 if opt=="AdamW" else (2*lr/(.0001*math.sqrt(t)) if opt=="Adagrad" else lr/(.0001*(3 if family=="multivariate_regression" else 10)) if mom==.9 else lr*.01/.0001)
  rr=dict(dataset=result["dataset"],T=t,opt=opt,momentum=mom if opt=="SGD" else None,lr=lr,ratio=round(ratio,6),mean=r["mean"],std=r["std"],failed=r.get("failed_seeds"),cached=r.get("cached"),excluded=r.get("excluded"),seeds=r.get("seed_results",[]),files=r.get("measurement_files",{}),source=str(p))
  rows.append(rr)
unique={}
for r in rows:
 key=(r["dataset"],r["T"],r["opt"],r["momentum"],r["ratio"])
 unique.setdefault(key,r)
rows=list(unique.values()); contrasts=[]
for r in rows:
 if r["opt"]=="AdamW": continue
 a=unique.get((r["dataset"],r["T"],"AdamW",None,1))
 if not a: continue
 z=dict(dataset=r["dataset"],T=r["T"],opt=r["opt"],momentum=r["momentum"],ratio=r["ratio"],failed_a=a["failed"],failed_b=r["failed"])
 if a["failed"]==0 and r["failed"]==0:
  def vals(x): return {s["seed"]:s.get("final_test_mse",s.get("final_test_ce")) for s in x["seeds"]}
  va,vb=vals(a),vals(r); ids=sorted(va.keys()&vb.keys()); d=np.array([vb[i]-va[i] for i in ids]); se=d.std(ddof=1)/np.sqrt(len(d)); z.update(n=len(d),margin=float(d.mean()),ci95=[float(d.mean()-2.262157*se),float(d.mean()+2.262157*se)],wins=int((d<0).sum()))
 else: z.update(n=0,margin=None,ci95=None,wins=None)
 contrasts.append(z)
Path("study/results.json").write_text(json.dumps(rows,indent=2));Path("study/contrasts.json").write_text(json.dumps(contrasts,indent=2))
print("CELLS",len(rows),"DATASETS",len({r["dataset"] for r in rows}),"FAILURES",sum(r["failed"] or 0 for r in rows),"ALLFAIL",sum(r["failed"]==10 for r in rows),"PARTIAL",sum(0<(r["failed"] or 0)<10 for r in rows))
for r in rows:
 if r["ratio"]==1: print("PARITY",r["dataset"],r["T"],r["opt"],r["momentum"],r["mean"],r["failed"])
for c in contrasts:
 if c["ratio"]==1: print("CI",c)
