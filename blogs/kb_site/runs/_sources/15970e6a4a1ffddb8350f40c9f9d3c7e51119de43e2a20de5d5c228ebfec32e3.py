import os,json,gzip,hashlib,re,collections,ast
import numpy as np
ROOT='sign_audit'
with gzip.open(ROOT+'/frozen_history.json.gz','rt') as f: history=json.load(f)
with gzip.open(ROOT+'/frozen_lab.json.gz','rt') as f: baseline=json.load(f)
def parse(r):
 q=r['question']; out={}
 for a,s in re.findall(r'### Choice ([A-Z])\n(.*?)(?=### Choice [A-Z]\n|## Your answer|\Z)',q,re.S):
  def get(p):
   m=re.search(p,s); return m.group(1) if m else None
  out[a]={k:get(p) for k,p in {
   'depth':r'Depth: (\d+)','width':r'Width: (\d+)',
   'residual':r'Residual connections: (True|False)','activation':r'Activation: (\w+)',
   'ln':r'Layer norm per layer: (\[[^\n]+\])','dim':r'Input dimension: (\d+)',
   'optimizer':r'Optimizer: (\w+)','lr':r'Learning rate: ([\d.e-]+)',
   'wd':r'Weight decay: ([\d.e-]+)','betas':r'Betas: (\[[^\n]+\])'}.items()}
  out[a]['default']='Initialization: PyTorch Linear defaults' in s
  out[a]['section_hash']=hashlib.sha256(s.strip().encode()).hexdigest()
  codes=re.findall(r'```python\n(.*?)\n```',s,re.S)
  for c in codes: ast.parse(c)
  out[a]['code_hashes']=[hashlib.sha256(c.encode()).hexdigest() for c in codes]
 return out
xor=[r for r in history if r.get('family')=='xor_classification']
p={r['question_id']:parse(r) for r in xor}
assert len(p)==131 and all(len(c) in (3,5) for c in p.values())
f=p['q_57db51']; allkeys=['depth','width','residual','activation','ln','dim','optimizer','lr','wd','betas','default']
audit={}
for keys in [allkeys,allkeys[:5],['depth','residual']]:
 def sig(v): return tuple(v[k] for k in keys)
 target=sorted(sig(v) for v in f.values())
 ids=[id for id,c in p.items() if len(c)==5 and sorted(sig(v) for v in c.values())==target]
 audit[','.join(keys)]=ids
assert all(ids==['q_57db51'] for ids in audit.values())
with open(ROOT+'/archive_audit.json','w') as z:json.dump({'parsed':p,'match_filters':audit},z,indent=2)
def eligible(r):
 return (r.get('family')=='xor_classification' and r['dataset']['input_dim']==16
  and r['budget']['training_steps']==512 and r['budget']['batch_size']==64
  and r['optimizer']=={'type':'Adam','lr':.0003,'weight_decay':.0001,'betas':[.9,.95]}
  and r['loss'].get('loss_id')=='cross_entropy')
old=[r for r in baseline if eligible(r)]
print('Baseline exact protocol candidates',len(old),'sets',len(set(r['set_id'] for r in old)))
for r in old: print(r['set_id'],r['model'])
with open(ROOT+'/baseline_eligible.json','w') as z:json.dump(old,z,indent=2)
with open(ROOT+'/results.json') as z: rows=json.load(z)
def stats(a,b):
 d=np.array([x['final_test_ce'] for x in a['seed_results']])-np.array([x['final_test_ce'] for x in b['seed_results']])
 half=2.2621571627409915*d.std(ddof=1)/np.sqrt(len(d))
 return {'difference':float(d.mean()),'ci95':[float(d.mean()-half),float(d.mean()+half)],'negative_seeds':int((d<0).sum()),'differences':d.tolist()}
comparisons={}; verification=[]
for dataset in sorted(set(r['dataset_id'] for r in rows)):
 rs=[r for r in rows if r['dataset_id']==dataset]
 by={(r['depth'],''.join(str(int(x)) for x in r['layer_norm']),r['activation']):r for r in rs}
 out={}
 for depth,mask in [(4,'1011'),(4,'1101'),(5,'10101'),(5,'11001')]:
  out['Leaky-ReLU '+mask]=stats(by[depth,mask,'leaky_relu'],by[depth,mask,'relu'])
 for shallow,deep in [('1011','10101'),('1101','11001')]:
  for act in ['relu','leaky_relu']:
   out['d5-d4 '+shallow+' '+act]=stats(by[5,deep,act],by[4,shallow,act])
 E=by[4,'1011','leaky_relu'];A=by[4,'1101','relu'];C=by[5,'11001','leaky_relu']
 for name,a,b in [('E-A',E,A),('C-E',C,E),('C-A',C,A),('LN1101-1011 ReLU',A,by[4,'1011','relu']),('LN1101-1011 Leaky',by[4,'1101','leaky_relu'],E)]:
  out[name]=stats(a,b)
 comparisons[dataset]=out
 print(dataset)
 print('CELLS',[(k,round(r['mean'],6),round(r['std'],6)) for k,r in by.items()])
 for name,v in out.items():print(name,round(v['difference'],6),[round(x,6) for x in v['ci95']],v['negative_seeds'])
for r in rows:
 assert r['n_seeds']==10 and r['failed_seeds']==0 and not r['excluded']
 assert abs(np.mean([s['final_test_ce'] for s in r['seed_results']])-r['mean'])<1e-12
 for name,m in r['measurement_files'].items():
  path=m['repo_path'] if os.path.exists(m['repo_path']) else m['path']
  raw=open(path,'rb').read()
  assert hashlib.sha256(raw).hexdigest()==m['sha256']
  verification.append({'dataset':r['dataset_id'],'candidate':r['candidate_id'],'name':name,'sha256':m['sha256']})
with open(ROOT+'/statistics.json','w') as z:json.dump(comparisons,z,indent=2)
with open(ROOT+'/verification.json','w') as z:json.dump(verification,z,indent=2)
print('verified artifacts',len(verification),'configurations',len(rows),'finite seeds',sum(r['n_seeds'] for r in rows))

def normalized_classes(src):
 tree=ast.parse(src)
 classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
 for cls in classes:
  for fn in cls.body:
   if isinstance(fn,ast.FunctionDef) and fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):fn.body.pop(0)
  if cls.body and isinstance(cls.body[0],ast.Expr) and isinstance(cls.body[0].value,ast.Constant) and isinstance(cls.body[0].value.value,str):cls.body.pop(0)
 return [ast.dump(n,include_attributes=False) for n in classes]
q=next(r['question'] for r in history if r['question_id']=='q_57db51')
sections=dict(re.findall(r'### Choice ([A-Z])\n(.*?)(?=### Choice [A-Z]\n|## Your answer|\Z)',q,re.S))
recipe_checks=[]
for r in rows:
 files=r['measurement_files']
 spec=json.load(open(files['candidate_spec.json']['repo_path']))
 assert spec['model']==r['model'] and spec['optimizer']==r['optimizer'] and spec['budget']==r['budget']
 curves=np.load(files['results/curves.npz']['repo_path'])
 ys=np.array([s['final_test_ce'] for s in r['seed_results']])
 assert curves['curves'].shape==(10,512)
 assert np.array_equal(curves['curves'][:,-1],ys)
 assert np.array_equal(curves['samples'],np.arange(1,513)*64)
 assert abs(ys.std(ddof=0)-r['std'])<1e-12
 target=None
 if (r['depth'],r['layer_norm'],r['activation'])==(4,[True,False,True,True],'leaky_relu'):target='E'
 if (r['depth'],r['layer_norm'],r['activation'])==(4,[True,True,False,True],'relu'):target='A'
 if (r['depth'],r['layer_norm'],r['activation'])==(5,[True,True,False,False,True],'leaky_relu'):target='C'
 if target:
  code=re.findall(r'```python\n(.*?)\n```',sections[target],re.S)[0]
  assert normalized_classes(code)==normalized_classes(open(files['executed/model.py']['repo_path']).read())
 recipe_checks.append({'dataset':r['dataset_id'],'candidate':r['candidate_id'],'focal_class_ast_match':target,'first_CE':float(curves['curves'][:,0].mean()),'last_CE':float(curves['curves'][:,-1].mean())})
with open(ROOT+'/recipe_checks.json','w') as z:json.dump(recipe_checks,z,indent=2)
print('All16 specs,128 file hashes,160 curve endpoints verified; E/A/C class ASTs equal original raw prompt.')
