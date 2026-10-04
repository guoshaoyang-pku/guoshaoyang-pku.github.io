"""Reproduce paired statistics from content-addressed measurements."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT = Path.cwd()
PAIRS = {
"sym_1582b1": [
["74955d588961b7a7c016de461a453746137ae6a76fbd448473d70e3f313ae963","40b0c2427c8f60e9c3afc4516b64f6bf795d426dc5463f3473d86c8505be80aa"],
["022263f040f9d12f0403112728897d85628c1547ffd0a4a2b34f8ceff9dc2f9d","280c60fdd5f83b8f9a7c6f732e33b652ef9dad9a343f96cb558b401287b18e42"]],
"sym_994c08": [
["5f84a8091e2ee07c2a8407b94586f622d21e82916ec7be1d716b25f49638f11a","16264fa181e23221a8f9b6fa2c6da48a899a4353e7335d763210fba462968639"],
["4ffbe25e46828714c6d1ed1575fe86a3ccc495e3eea829bf4255b82dc4c1f284","e28b6b22e6a3bac99873d8d5964e5478ef312e3bfc175bae36edd2ff743342ff"]]}
def verified_path(digest,suffix):
    p=ROOT/"measurements"/(digest+suffix)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    return p
def analyze():
    output={}
    for dataset,(summaries,curves) in PAIRS.items():
        ss=[json.loads(verified_path(h,".json").read_text()) for h in summaries]
        rows=[sorted(s["seed_results"],key=lambda r:r["seed"]) for s in ss]
        assert [[r["seed"] for r in rs] for rs in rows]==[list(range(10))]*2
        v=np.array([[r["final_test_mse"] for r in rs] for rs in rows])
        assert np.isfinite(v).all() and not any(r["failed"] for rs in rows for r in rs)
        low,high=v; d=low-high; se=d.std(ddof=1)/np.sqrt(10)
        idx=np.random.default_rng(6259).integers(0,10,(100000,10))
        bs=v[:,idx].mean(axis=2)
        cs=[np.load(verified_path(h,".npz")) for h in curves]
        for c,final in zip(cs,v):
            assert c["curves"].shape==(10,512)
            assert np.allclose(c["curves"][:,-1],final)
            assert np.array_equal(c["samples"],np.arange(1,513)*32)
        output[dataset]={
            "means":v.mean(axis=1).tolist(),"sample_sd":v.std(axis=1,ddof=1).tolist(),
            "runner_sd":[s["std_test_mse"] for s in ss],
            "finite_counts":np.isfinite(v).sum(axis=1).tolist(),"failed_counts":[s["failed_seeds"] for s in ss],
            "ratio_of_means":float(low.mean()/high.mean()),"delta_mean":float(d.mean()),
            "paired_t95_df9":[float(d.mean()-2.2621571628540993*se),float(d.mean()+2.2621571628540993*se)],
            "paired_bootstrap95":np.quantile(bs[0]-bs[1],[.025,.975]).tolist(),
            "ratio_bootstrap95":np.quantile(bs[0]/bs[1],[.025,.975]).tolist(),
            "low_wins":int((d<0).sum()),"high_wins":int((d>0).sum()),"ties":int((d==0).sum()),
            "seed_values_low_high":v.T.tolist(),
            "test_curve_means_steps1_128_256_512":[c["curves"][:,[0,127,255,511]].mean(axis=0).tolist() for c in cs]}
    (ROOT/"artifacts"/"analysis.json").write_text(json.dumps(output,indent=2)+"\n")
    return output
if __name__=="__main__":
    print(json.dumps(analyze(),indent=2))
