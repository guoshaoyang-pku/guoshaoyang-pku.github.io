import json, pathlib, numpy as np
rows=[]
for p in sorted(pathlib.Path("results").glob("spiralcls_*.json")):
 s=json.loads(p.read_text()); rs=s["results"]
 if len(rs)!=8: continue
 keys=[(d,r,t) for d in [2,4] for r in [False,True] for t in [False,True]]
 cells=dict(zip(keys,rs))
 for kind in ["skip","depth","placement"]:
  for key,b in cells.items():
   d,r,t=key
   if (kind=="skip" and r) or (kind=="depth" and d==4) or (kind=="placement" and t):continue
   other=(d,True,t) if kind=="skip" else (4,r,t) if kind=="depth" else (d,r,True)
   a=cells[other]
   av={x["seed"]:x["final_test_ce"] for x in a["seed_results"] if not x["failed"]};bv={x["seed"]:x["final_test_ce"] for x in b["seed_results"] if not x["failed"]}
   delta=np.array([av[i]-bv[i] for i in sorted(av.keys()&bv.keys())]);se=delta.std(ddof=1)/np.sqrt(len(delta));mean=delta.mean()
   rows.append(dict(dataset=s["dataset"],kind=kind,key=list(key),a=a["mean"],b=b["mean"],difference=float(mean),ratio=a["mean"]/b["mean"],ci95=[float(mean-2.262*se),float(mean+2.262*se)],wins=int((delta<0).sum()),n=len(delta)))
pathlib.Path("contrasts.json").write_text(json.dumps(rows,indent=2))
for r in rows: print(r["dataset"].split("/")[-1],r["kind"],r["key"],"delta %.6g ratio %.4g CI [%.5g, %.5g] wins %d/%d"%(r["difference"],r["ratio"],*r["ci95"],r["wins"],r["n"]))
