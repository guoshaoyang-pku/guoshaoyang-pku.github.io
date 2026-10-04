import glob,json,numpy as np

def feature(delta,width,depth,p=.5,q=.5):
 x=np.log(delta/.1)+p*np.log(width/64)+q*np.log(depth)
 return np.array([1,np.log(width/64),np.log(depth),x,x*x,x*x*x])

def inherited():
 return [json.load(open(p)) for p in sorted(glob.glob('joint_law/inherited_*.json'))]

def benchmark(rows,alpha,delta,width,depth):
 rr=sorted([r for r in rows if r['dataset']['alpha']==alpha and r['d_model']==width and r['num_layers']==depth],key=lambda r:r['optimizer']['lr'])
 return float(np.interp(np.log(delta),[np.log(r['optimizer']['lr']*256) for r in rr],[r['mean'] for r in rr]))

def fit(rows,p,q):
 out={}
 for a in [.8,1.4]:
  rr=[r for r in rows if r['dataset']['alpha']==a]
  X=np.array([feature(r['optimizer']['lr']*256,r['d_model'],r['num_layers'],p,q) for r in rr]);y=np.array([r['mean'] for r in rr]);b=np.linalg.lstsq(X,y,rcond=None)[0]
  out[str(a)]={'coef':b.tolist(),'rmse':float(np.sqrt(np.mean((X@b-y)**2))),'max_error':float(np.max(abs(X@b-y)))}
 return out

if __name__=='__main__':
 for name,p,q in [('nominal',0,0),('width',.5,0),('joint',.5,.5)]:print(name,json.dumps(fit(inherited(),p,q)))
