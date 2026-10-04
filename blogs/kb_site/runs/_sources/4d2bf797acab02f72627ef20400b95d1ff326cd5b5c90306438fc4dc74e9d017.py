"""Expression-only domain descriptors; not optimizer motion."""
import json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
base=json.loads((P/'descriptors.json').read_text());out=[]
for N in [65,129,257]:
    x0,x1=np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N),indexing='ij')
    env=dict(x0=x0,x1=x1,pi=np.pi,sin=np.sin,cos=np.cos,tanh=np.tanh,abs=np.abs)
    for r in base:
        y=eval(r['expression'],{'__builtins__':{}},env)
        g=np.gradient(y,1/(N-1));H=[np.gradient(a,1/(N-1)) for a in g]
        out.append(dict(id=r['id'],N=N,mean=float(y.mean()),sd=float(y.std()),rms=float(np.sqrt(np.mean(y*y))),curvature=float(np.sqrt(sum(np.mean(a*a) for row in H for a in row))/y.std()),grad=float(np.sqrt(sum(np.mean(a*a) for a in g))/y.std())))
(P/'descriptor_convergence.json').write_text(json.dumps(out,indent=2))
print([(r['id'],r['N'],round(r['curvature'],3)) for r in out])
