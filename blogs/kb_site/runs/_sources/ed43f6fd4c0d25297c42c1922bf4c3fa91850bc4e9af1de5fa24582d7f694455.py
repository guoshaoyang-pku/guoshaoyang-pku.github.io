import ast, json, numpy as np
FUN = {'sin':np.sin,'cos':np.cos,'tanh':np.tanh,'abs':np.abs}
def parse(s):
    root=ast.parse(s,mode='eval').body
    for n in ast.walk(root):
        if isinstance(n,(ast.Expression,ast.Load,ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow,ast.UAdd,ast.USub)): continue
        if isinstance(n,ast.Constant) and type(n.value) in (int,float): continue
        if isinstance(n,ast.Name) and n.id in list(FUN)+['pi']+['x'+str(i) for i in range(8)]: continue
        if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub,ast.Mult,ast.Div,ast.Pow)): continue
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,(ast.UAdd,ast.USub)): continue
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in FUN and len(n.args)==1 and not n.keywords: continue
        raise ValueError('unsupported expression node '+ast.dump(n))
    return root
def value(n,x):
    if isinstance(n,ast.Constant): return n.value
    if isinstance(n,ast.Name): return np.pi if n.id=='pi' else x[:,int(n.id[1:])]
    if isinstance(n,ast.Call): return FUN[n.func.id](value(n.args[0],x))
    if isinstance(n,ast.UnaryOp): return (-1 if isinstance(n.op,ast.USub) else 1)*value(n.operand,x)
    a,b=value(n.left,x),value(n.right,x)
    if isinstance(n.op,ast.Add): return a+b
    if isinstance(n.op,ast.Sub): return a-b
    if isinstance(n.op,ast.Mult): return a*b
    if isinstance(n.op,ast.Div): return a/b
    if isinstance(n.op,ast.Pow): return a**b
    raise ValueError('unsupported')
def additive_constant(n):
    if isinstance(n,ast.BinOp) and isinstance(n.op,(ast.Add,ast.Sub)):
        return additive_constant(n.left)+(1 if isinstance(n.op,ast.Add) else -1)*additive_constant(n.right)
    if isinstance(n,ast.UnaryOp): return (-1 if isinstance(n.op,ast.USub) else 1)*additive_constant(n.operand)
    if any(isinstance(a,ast.Name) and a.id.startswith('x') for a in ast.walk(n)): return 0.
    return float(value(n,np.zeros((1,8))))
def features(expr):
    n=parse(expr); rng=np.random.default_rng(1141); chunks=[]; sy=sy2=0.; cov=np.zeros(8)
    for j in range(16):
        x=rng.uniform(size=(32768,8)); y=value(n,x)
        if not np.all(np.isfinite(y)): raise ValueError('nonfinite')
        chunks.append(float(np.mean(y))); sy+=float(np.sum(y)); sy2+=float(np.sum(y*y)); cov+=np.sum((x-.5)*y[:,None],axis=0)
    N=16*32768; mean=sy/N; var=sy2/N-mean**2
    return dict(mean=mean,mean_mc_se=float(np.std(chunks,ddof=1)/4),variance=var,linear_fraction=float(12*np.sum((cov/N)**2)/var),constant=additive_constant(n),n_mc=N)
def fit(z,y):
    u=sorted(set(z)); ts=[u[0]-1]+[(a+b)/2 for a,b in zip(u,u[1:])]+[u[-1]+1]
    candidates=[(int(np.sum((np.array(z)>t if direction==1 else np.array(z)<=t)!=y)),direction,t) for t in ts for direction in [1,-1]]
    err,d,t=min(candidates,key=lambda a:(a[0],-a[1],a[2])); return dict(errors=err,direction=d,threshold=t)
def predict(model,z): return bool(z>model['threshold']) if model['direction']==1 else bool(z<=model['threshold'])
if __name__=='__main__':
    ds=json.load(open('visible_targets.json')); rows=json.load(open('inherited.json')); ids=sorted(set(r['dataset_id'] for r in rows)); fresh=['mvar_14c786','mvar_15c0c5','mvar_16d428','mvar_1b1e06']
    f={i:features(ds[i]['expression']) for i in ids+fresh}; z=[abs(f[i]['mean']) for i in ids]; y=[]; results=[]
    for i in ids:
        pair={r['optimizer']['type']:r for r in rows if r['dataset_id']==i}; delta=pair['SGD']['mean']-pair['Adam']['mean']; y.append(delta<0)
        a={s['seed']:s['final_test_mse'] for s in pair['SGD']['seed_results']}; b={s['seed']:s['final_test_mse'] for s in pair['Adam']['seed_results']}; d=np.array([a[s]-b[s] for s in sorted(a)]); ci=2.262157*np.std(d,ddof=1)/np.sqrt(10)
        results.append(dict(dataset=i,delta=delta,ci=[float(d.mean()-ci),float(d.mean()+ci)],features=f[i]))
    model=fit(z,np.array(y)); loo=[]
    for j,i in enumerate(ids):
        keep=[k for k in range(9) if k!=j]; m=fit([z[k] for k in keep],np.array([y[k] for k in keep])); loo.append(dict(dataset=i,pred_sgd=predict(m,z[j]),actual_sgd=y[j],model=m))
    out=dict(model=model,results=results,loo=loo,fresh=[dict(dataset=i,features=f[i],pred_sgd=predict(model,abs(f[i]['mean']))) for i in fresh])
    json.dump(out,open('preregistration.json','w'),indent=2); print(json.dumps(out,indent=2))
