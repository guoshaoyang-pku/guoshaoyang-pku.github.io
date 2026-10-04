import json, os, hashlib, itertools
import numpy as np
rows = {}
manifest = {}
os.makedirs("contradiction/recipes", exist_ok=True)
for filename in sorted(os.listdir("experiments")):
    d = json.load(open("experiments/" + filename))
    for r in d["result"]["results"]:
        c = r["candidate"]
        for artifact in r.get("measurement_files", {}).values():
            p = artifact["repo_path"]
            digest = hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert digest == artifact["sha256"], p
            manifest[p] = digest
        if c["budget"]["training_steps"] == 0:
            continue
        seed = sorted(r["seed_results"], key=lambda x: x["seed"])
        assert [x["seed"] for x in seed] == list(range(10))
        assert not any(x["failed"] for x in seed)
        y = np.array([x["final_test_ce"] for x in seed])
        assert np.isclose(y.mean(), r["mean"], atol=1e-12)
        curve = np.load(r["measurement_files"]["results/curves.npz"]["repo_path"])["curves"]
        assert np.allclose(curve[:, -1], y, atol=1e-7)
        key = (d["result"]["dataset"].split("/")[-1], c["model"]["d_ff"], c["optimizer"]["type"], c["optimizer"]["lr"])
        if key in rows:
            assert np.array_equal(rows[key]["values"], y), key
        rows[key] = dict(mean=float(y.mean()), sd=float(y.std()), values=y.tolist(), cached=r["cached"], source=filename, candidate=c)
        specpath = r["measurement_files"]["candidate_spec.json"]["path"]
        root = os.path.dirname(specpath)
        recipe = os.path.join("contradiction/recipes", hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12])
        os.makedirs(recipe, exist_ok=True)
        for name in ["model.py","optimizer.py","loss.py","train.py"]:
            with open(recipe+"/"+name,"w") as f:
                f.write(open(root+"/"+name).read())
def contrast(a,b):
    x=np.array(a)-np.array(b)
    half=2.2621571627409915*x.std(ddof=1)/np.sqrt(10)
    return dict(diff=float(x.mean()),ci=[float(x.mean()-half),float(x.mean()+half)],wins=int((x<0).sum()),seed_diffs=x.tolist())
effects=[]
for ds in sorted({k[0] for k in rows}):
    for ff in [64,256]:
        a=rows.get((ds,ff,"RMSprop",.003));b=rows.get((ds,ff,"Adagrad",.0001))
        if a and b:
            effects.append(dict(dataset=ds,ff=ff,comparison="RMS.003-Adagrad.0001",ratio=a["mean"]/b["mean"],**contrast(a["values"],b["values"])))
    for opt, rates in [("RMSprop",[.0001,.0003,.001,.003]),("Adagrad",[.00003,.0001,.0003,.001,.003])]:
        for lo,hi in itertools.combinations(rates,2):
            a=rows.get((ds,64,opt,hi));b=rows.get((ds,64,opt,lo))
            if a and b:
                effects.append(dict(dataset=ds,ff=64,comparison=opt+":"+str(hi)+"-"+str(lo),**contrast(a["values"],b["values"])))
    if (ds,256,"RMSprop",.003) in rows:
        r256=np.array(rows[(ds,256,"RMSprop",.003)]["values"]);a256=np.array(rows[(ds,256,"Adagrad",.0001)]["values"])
        r64=np.array(rows[(ds,64,"RMSprop",.003)]["values"]);a64=np.array(rows[(ds,64,"Adagrad",.0001)]["values"])
        effects.append(dict(dataset=ds,comparison="FF ranking difference-in-differences",**contrast(r256-a256,r64-a64)))
out=dict(cells=[dict(key=list(k),**v) for k,v in sorted(rows.items())],effects=effects,manifest=manifest)
with open("contradiction/audit.json","w") as f: json.dump(out,f,indent=2)
print("AUDIT",len(rows),"cells",len(manifest),"hashed artifacts")
for e in effects:
    if e["comparison"] in ["RMS.003-Adagrad.0001","FF ranking difference-in-differences","RMSprop:0.001-0.0001","RMSprop:0.003-0.001","Adagrad:0.001-0.0001","Adagrad:0.003-0.001"]:print(json.dumps(e))
