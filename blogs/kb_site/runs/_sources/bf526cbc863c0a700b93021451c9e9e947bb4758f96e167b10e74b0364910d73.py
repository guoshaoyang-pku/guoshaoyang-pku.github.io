import glob,json,hashlib,os
import numpy as np
from fit import feature, inherited, benchmark

def seeds(r):
 return np.array([s['final_test_ce'] for s in sorted(r['seed_results'],key=lambda x:x['seed']) if not s['failed']])

def contrast(a,b):
 x=seeds(a)-seeds(b)
 m=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(len(x)))
 return {'effect':m,'ratio':float(a['mean']/b['mean']),'ci':[m-2.262157162854*se,m+2.262157162854*se],'wins':int((x<0).sum())}

def main():
 pred=json.load(open('joint_law/predictions.json'));models=json.load(open('joint_law/fit.json'))
 cells=[];allraw=[]; scores={};rs=[]
 for path in sorted(glob.glob('joint_law/fresh_*.json')):
  raw=json.load(open(path)); rr=raw['results'];allraw.extend(rr)
  ds=raw['dataset'].split('/')[-1];a=raw['dataset_params']['alpha']
  D=max(r['variant']['optimizer.lr'] for r in rr)*256
  for h in [1,4]:
   def get(w,mode):
    rate=D/256 if mode=='unchanged' else D/256*np.sqrt(64/w)/np.sqrt(h)
    return next(r for r in rr if r['variant']['model.d_model']==w and r['variant']['model.num_layers']==h and np.isclose(r['variant']['optimizer.lr'],rate,rtol=1e-12,atol=0))
   u64,u128,j64,j128=[get(w,mode) for mode,w in [('unchanged',64),('unchanged',128),('joint',64),('joint',128)]]
   pr=next(p for p in pred if p['alpha']==a and np.isclose(p['Delta'],D) and p['depth']==h)
   c={'dataset':ds,'alpha':a,'Delta':D,'depth':h,'means':[r['mean'] for r in [u64,u128,j64,j128]],'std':[r['std'] for r in [u64,u128,j64,j128]],'width_unchanged':contrast(u128,u64),'width_joint':contrast(j128,j64),'rescue128':contrast(j128,u128)}
   bm=pr['benchmark_means']
   for model,pp in [('joint',pr),('threshold',{'width_unchanged':bm['unchanged128']-bm['unchanged64'],'width_joint':bm['joint128']-bm['joint64'],'rescue128':bm['joint128']-bm['unchanged128']})]:
    for key in ['width_unchanged','width_joint','rescue128']:
     actual=c[key]['effect'];point=pp[key]
     rs.append({'dataset':ds,'Delta':D,'depth':h,'model':model,'contrast':key,'point':point,'actual':actual,'error':actual-point,'sign_ok':bool(np.sign(actual)==np.sign(point)),'range_ok':bool(abs(actual-point)<=.04)})
   cells.append(c)
   for model,p,q in [('nominal',0,0),('width',.5,0),('joint',.5,.5),('threshold',None,None)]:
    for i,(mode,w) in enumerate([('unchanged',64),('unchanged',128),('joint',64),('joint',128)]):
     dd=D if mode=='unchanged' else D*np.sqrt(64/w)/np.sqrt(h)
     point=bm[mode+str(w)] if model=='threshold' else float(feature(dd,w,h,p,q)@np.array(models[model][str(a)]['coef']))
     scores.setdefault(model,[]).append(c['means'][i]-point)
 for model in ['joint','threshold']:
  out=[r for r in rs if r['model']==model]
  print('SCORE',model,'signs',sum(r['sign_ok'] for r in out),'ranges',sum(r['range_ok'] for r in out),'n',len(out),'MAE',np.mean([abs(r['error']) for r in out]))
  for key in ['width_unchanged','width_joint','rescue128']:
   sub=[r for r in out if r['contrast']==key]
   print(key,sum(r['sign_ok'] for r in sub),sum(r['range_ok'] for r in sub),len(sub),np.mean([abs(r['error']) for r in sub]))
 for model,err in scores.items():print('ENDPOINT',model,'MAE',np.mean(abs(np.array(err))),'RMSE',np.sqrt(np.mean(np.array(err)**2)),'within.1',sum(abs(e)<=.1 for e in err),'n',len(err))
 print('RAW',len(allraw),'cached',sum(r['cached'] for r in allraw),'failed',sum(r['failed_seeds'] for r in allraw),'excluded',sum(r['excluded'] for r in allraw))
 print('WIDTHFIRST',sum(c['width_unchanged']['effect']<0 for c in cells),len(cells))
 depth=[]
 for ds in sorted(set(c['dataset'] for c in cells)):
  for D in [.05,.1,.3]:
   one=next(c for c in cells if c['dataset']==ds and np.isclose(c['Delta'],D) and c['depth']==1)
   four=next(c for c in cells if c['dataset']==ds and np.isclose(c['Delta'],D) and c['depth']==4)
   depth.extend([four['means'][i]-one['means'][i] for i in [0,1]])
 print('DEPTHFIRST',sum(d<0 for d in depth),len(depth))
 json.dump(cells,open('joint_law/contrasts.json','w'))
 json.dump(rs,open('joint_law/scores.json','w'))
 print('|Table|Delta|depth|unchanged CE64 /128|joint CE64 /128|W unchanged: effect [95%CI];ratio|W joint: effect [95%CI];ratio|')
 print('|---|---|---|---|---|---|---|')
 for c in cells:
  def fmt(key):
   r=c[key];return '%+.5f [%+.5f,%+.5f];%.5f'%(r['effect'],*r['ci'],r['ratio'])
  print('|%s|%.2g|%d|%.5f /%.5f|%.5f /%.5f|%s|%s|'%(c['dataset'],c['Delta'],c['depth'],*c['means'],fmt('width_unchanged'),fmt('width_joint')))
 print('RESCUE')
 for c in cells:
  r=c['rescue128'];print(c['dataset'],'%.2g'%c['Delta'],c['depth'],'%+.5f [%+.5f,%+.5f] ratio %.5f'%(r['effect'],*r['ci'],r['ratio']))
 return cells
if __name__=='__main__':main()
