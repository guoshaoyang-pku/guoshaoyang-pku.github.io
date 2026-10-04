import json,gzip,glob,os,hashlib
import numpy as np
root="gru_optimizer"
rows=[]
for path in sorted(glob.glob("experiments/*.json")):
 e=json.load(open(path)); set_id="exp:"+os.path.basename(path)[:-5]
 for t,off in [(1024,0),(256,3)]:
  specs=e["input"]["candidates"][off:off+3]; rs=e["result"]["results"][off:off+3]
  assert len(rs)==3
  for s in specs:
   assert s["model"]=={"type":"gru_lm","vocab_size":48,"context_length":24,"d_model":64,"num_layers":1,"layer_residual":False}
   assert s["budget"]=={"training_steps":t,"batch_size":16}
   assert s["loss"]["loss_id"]=="cross_entropy"
  vals=[np.array([z["final_test_ce"] for z in r["seed_results"]]) for r in rs]
  for r in rs:
   assert r["n_seeds"]==10 and r["failed_seeds"]==0 and not r["excluded"]
   for m in r["measurement_files"].values():
    assert hashlib.sha256(open(m["repo_path"],"rb").read()).hexdigest()==m["sha256"]
  d=vals[2]-vals[1];margin=float(d.mean());se=float(d.std(ddof=1)/np.sqrt(10))
  conservative=2.8*(rs[1]["std"]+rs[2]["std"])/np.sqrt(10)
  cs=rs[2]["measurement_files"]["results/curves.npz"]["repo_path"];curves=np.load(cs)["curves"]
  row={"dataset":e["input"]["dataset"],"set_id":set_id,"alpha":e["dataset_params"]["alpha"],"T":t,
       "means":[r["mean"] for r in rs],"stds":[r["std"] for r in rs],"C_minus_B":margin,
       "C_over_B":rs[2]["mean"]/rs[1]["mean"],"paired_t9_95":[margin-2.262157*se,margin+2.262157*se],
       "conservative_interval":[margin-conservative,margin+conservative],
       "C_seed_wins":int((d<0).sum()),"A_minus_B":float((vals[0]-vals[1]).mean()),
       "A_over_B":rs[0]["mean"]/rs[1]["mean"],"A_minus_C":float((vals[0]-vals[2]).mean()),
       "A_over_C":rs[0]["mean"]/rs[2]["mean"],"C_mean_curve_min":float(curves.mean(0).min()),
       "C_min_step":int(curves.mean(0).argmin()+1),"scheduled_seeds":30,"failed_seeds":0,
       "cached":[r["cached"] for r in rs]}
  rows.append(row)
json.dump(rows,open(root+"/analysis.json","w"),indent=2)
old=json.load(gzip.open(root+"/history.json.gz","rt"));new=json.load(gzip.open(root+"/checkpoint_history.json.gz","rt"))
ids={x["question_id"] for x in old}
focal=json.load(open(root+"/q_90b402.json"))["question"]
choices=focal[focal.index("### Choice A"):focal.index("## Your answer")]
exact=[x["question_id"] for x in new if x["question_id"] not in ids and choices in x.get("question","")]
print("new exact recipe questions",exact)
for r in sorted(rows,key=lambda r:(r["alpha"],r["dataset"],-r["T"])): print(json.dumps(r))
