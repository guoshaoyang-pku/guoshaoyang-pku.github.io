import glob,hashlib,json,sys
import numpy as np
from fit import fit,feature,inherited

rows=inherited()
assert len(rows)==84
for row in rows:
 assert np.mean([s['final_test_ce'] for s in row['seed_results']])==row['mean']
for name,p,q in [('nominal',0,0),('width',.5,0),('joint',.5,.5)]:
 calculated=fit(rows,p,q)
 frozen=json.load(open('joint_law/fit.json'))[name]
 for alpha in ['0.8','1.4']:
  assert np.allclose(calculated[alpha]['coef'],frozen[alpha]['coef'],rtol=0,atol=1e-12)
for pred in json.load(open('joint_law/predictions.json')):
 b=np.array(json.load(open('joint_law/fit.json'))['joint'][str(pred['alpha'])]['coef'])
 for mode in ['unchanged','joint']:
  for width in [64,128]:
   d=pred['Delta'] if mode=='unchanged' else pred['Delta']*np.sqrt(64/width)/np.sqrt(pred['depth'])
   assert abs(float(feature(d,width,pred['depth'])@b)-pred['means'][mode+str(width)])<1e-12
for digest,item in json.load(open('joint_law/manifest.json')).items():
 assert hashlib.sha256(open(item['repo_path'],'rb').read()).hexdigest()==digest
for item in json.load(open('joint_law/recipe_manifest.json')).values():
 assert hashlib.sha256(open(item['repo_path'],'rb').read()).hexdigest()==item['sha256']
for path in glob.glob('joint_law/fresh_*.json'):
 record=json.load(open(path))
 for row in record['results']:
  v=row['variant']
  assert v['model.num_heads']==4 and v['model.d_ff']==128
  assert v['optimizer.type']=='AdamW' and v['optimizer.betas']==[.9,.999] and v['optimizer.weight_decay']==0
  assert v['budget.training_steps']==256 and v['budget.batch_size']==16 and v['loss.loss_id']=='cross_entropy'
  assert row['n_seeds']==10 and row['failed_seeds']==0 and not row['excluded']
print('PASS:84 inherited means, frozen fits/predictions,494 measurement hashes,recipe hashes and all fresh fixed controls')
