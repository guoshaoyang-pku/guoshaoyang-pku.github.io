import json,gzip,glob,hashlib,math,collections,csv
import numpy as np
ROOT='slow_sgd/'
def load_response(path):
    x=json.load(open(path))
    return json.loads(x['content'][0]['text'])
lab=json.load(gzip.open(ROOT+'lab_frozen.json.gz','rt'))
raw=json.load(open(ROOT+'backtest_False.json'))
clean=[];seen=set()
for p in raw:
    g,t=p['g'],p['t']
    if any(k in t['model'] for k in ['head_scale','position_embedding']): continue
    key=(g['dataset_id'],json.dumps(g['model'],sort_keys=True),json.dumps(t['model'],sort_keys=True),json.dumps(g['optimizer'],sort_keys=True),json.dumps(g['budget'],sort_keys=True))
    if key not in seen:clean.append(p);seen.add(key)
main=[]
for p in clean:
    g,t=p['g'],p['t']
    if g['dataset_id'] in ['bg_1d9caf','bg_209ff2','bg_0bf6d7','bg_388f11','bg_0f0001','bg_30d1db'] and g['num_layers']==1 and t['d_model']==32 and t['num_layers']==4 and t['d_ff']==128:
        main.append(p)
# Reproduce the full old 36-cell grid, including exposure1.536 outside the proposed slow band.
old=[];seenold=set()
for g in lab:
    if g['dataset_id'] not in ['bg_1d9caf','bg_209ff2','bg_0bf6d7','bg_388f11','bg_0f0001','bg_30d1db'] or g['model_type']!='gru_lm' or g['d_model']!=64 or g['num_layers']!=1:continue
    if g['optimizer'].get('type')!='SGD' or g['optimizer'].get('weight_decay',0)!=0 or g['optimizer'].get('momentum',0)!=0 or g['optimizer']['lr'] not in [.0001,.001,.003] or g['budget']['training_steps'] not in [256,512] or g['budget']['batch_size']!=16 or g['loss']['loss_id']!='cross_entropy':continue
    for t in lab:
        if t['set_id']!=g['set_id'] or t['model_type']!='transformer_lm' or t['d_model']!=32 or t['num_layers']!=4 or t.get('num_heads')!=2 or t.get('d_ff')!=128 or t['optimizer']!=g['optimizer'] or t['budget']!=g['budget'] or t['loss']!=g['loss'] or 'head_scale' in t['model']:continue
        key=(g['dataset_id'],g['optimizer']['lr'],g['budget']['training_steps'])
        if key not in seenold:old.append((g,t));seenold.add(key)
results={'backtest':{'n':len(clean),'wins':sum(p['adv']>0 for p in clean),'sets':len(set(p['g']['set_id'] for p in clean)),'datasets':len(set(p['g']['dataset_id'] for p in clean)),'min':min(p['adv'] for p in clean),'max':max(p['adv'] for p in clean),'gru2':sum(p['g']['num_layers']==2 for p in clean),'shapes':dict(collections.Counter(str((p['g']['num_layers'],p['t']['d_model'],p['t']['num_layers'],p['t'].get('num_heads'),p['t'].get('d_ff'))) for p in clean))},'old36':{'n':len(old),'median_difference':float(np.median([g['mean']-t['mean'] for g,t in old])),'median_ratio':float(np.median([g['mean']/t['mean'] for g,t in old]))}}
configs={};errors=[];hashes={};sources={}
for path in glob.glob(ROOT+'*.json'):
    if not any(s in path for s in ['zero12','zero24','grid_','correct24','adam.json']):continue
    response=load_response(path)
    if 'results' not in response:continue
    d=response['dataset'].split('/')[-1]
    for r in response['results']:
        v=r['variant'];shape=('G' if v['model.type']=='gru_lm' else 'TF')+str(v['model.num_layers'])
        T=v['budget.training_steps'];delta=round(v['optimizer.lr']*T,8);opt=v['optimizer.type']
        if r.get('mean') is None:
            errors.append({'file':path,'variant':v,'error':r.get('error'),'n':r.get('n_seeds'),'failed_seeds':r.get('failed_seeds')});continue
        configs[(d,opt,T,delta,shape)]=r
        for label,m in r.get('measurement_files',{}).items():
            fp=m['repo_path'];actual=hashlib.sha256(open(fp,'rb').read()).hexdigest()
            assert actual==m['sha256'],fp
            hashes[fp]=actual
            if label.startswith('executed/'):sources[fp]=open(fp).read()
def vals(r):return np.array([s['final_test_ce'] for s in sorted(r['seed_results'],key=lambda s:s['seed']) if not s.get('failed')])
def interval(a):
    m=float(np.mean(a));w=2.2621571628*float(np.std(a,ddof=1))/math.sqrt(len(a))
    return [m-w,m+w]
baselines={}
for (d,opt,T,delta,shape),r in configs.items():
    if opt=='SGD' and delta==0:baselines[(d,shape)]=r
cells=[];contrasts=[]
for d in ['bg_04502b','bg_0bf6d7']:
    for T in [512,2048]:
        for delta in [.2,.6,.8]:
            rs={s:configs.get((d,'SGD',T,delta,s)) for s in ['G1','G2','TF1','TF4']}
            if not all(rs.values()):continue
            cell={'dataset':d,'T':T,'delta':delta,'final_test_ce':{s:r['mean'] for s,r in rs.items()},'final_train_ce':None,'test_progress':{s:baselines[(d,s)]['mean']-r['mean'] for s,r in rs.items()}}
            cells.append(cell)
            for g in ['G1','G2']:
                for t in ['TF1','TF4']:
                    margin=vals(rs[t])-vals(rs[g]);initial=vals(baselines[(d,t)])-vals(baselines[(d,g)])
                    progextra=initial-margin
                    conservative=2.8*(rs[t]['std']+rs[g]['std'])/math.sqrt(10)
                    contrasts.append({'dataset':d,'T':T,'delta':delta,'pair':g+'/'+t,'margin':float(np.mean(margin)),'ci95':interval(margin),'conservative_ci':[float(np.mean(margin))-conservative,float(np.mean(margin))+conservative],'n':len(margin),'seeds':list(range(10)),'failures':rs[t]['failed_seeds']+rs[g]['failed_seeds'],'initial_margin':float(np.mean(initial)),'tf_extra_progress':float(np.mean(progextra)),'progress_ci95':interval(progextra),'retention_fraction':float(np.mean(margin)/np.mean(initial))})
results.update({'baselines':{d:{s:r['mean'] for (dd,s),r in baselines.items() if dd==d} for d in ['bg_04502b','bg_0bf6d7']},'cells':cells,'contrasts':contrasts,'errors':errors,'hashes':hashes,'configurations':len(configs),'finite_seed_evaluations':sum(len(vals(r)) for r in configs.values()),'seed_failures':sum(r.get('failed_seeds',0) for r in configs.values()),'source_status':collections.Counter(r.get('source_provenance',{}).get('status') for r in configs.values())})
results['duration']=[]
for d in ['bg_04502b','bg_0bf6d7']:
 for delta in [.2,.6,.8]:
  for g in ['G1','G2']:
   for t in ['TF1','TF4']:
    keys=[(d,'SGD',T,delta,s) for T in [512,2048] for s in [g,t]]
    if all(k in configs for k in keys):
     a,b,c,e=[vals(configs[k]) for k in keys]
     results['duration'].append({'dataset':d,'delta':delta,'pair':g+'/'+t,'margin_shift':float(np.mean((e-c)-(b-a))),'ci95':interval((e-c)-(b-a))})
with open(ROOT+'results.json','w') as f:json.dump(results,f,indent=2)
with open(ROOT+'contrasts.csv','w') as f:
 w=csv.DictWriter(f,fieldnames=list(contrasts[0]) if contrasts else ['dataset']);w.writeheader();w.writerows(contrasts)
print('AUDIT',results['backtest'],results['old36'])
print('BASELINES',results['baselines'])
print('CONFIG',results['configurations'],results['finite_seed_evaluations'],results['seed_failures'],'errors',len(errors),'sources',results['source_status'])
for c in cells:
 cs=[a for a in contrasts if a['dataset']==c['dataset'] and a['T']==c['T'] and a['delta']==c['delta']]
 print('CELL',c['dataset'],c['T'],c['delta'],'CE',','.join('%.6f'%c['final_test_ce'][s] for s in ['G1','G2','TF1','TF4']),'progress',','.join('%.6f'%c['test_progress'][s] for s in ['G1','G2','TF1','TF4']))
 print('MARGINS',';'.join(a['pair']+': %.6f [%.6f,%.6f]'%(a['margin'],*a['ci95']) for a in cs))
if contrasts:
 print('WIN',sum(c['margin']>0 for c in contrasts),len(contrasts),'CIWIN',sum(c['ci95'][0]>0 for c in contrasts),'CILOSS',sum(c['ci95'][1]<0 for c in contrasts),'TFPROGRESS',sum(c['tf_extra_progress']>0 for c in contrasts))
 print('DURATION_MAX',max(abs(c['margin_shift']) for c in results['duration']))
for key,r in configs.items():
 if key[1]=='Adam':print('ADAM',key,r['mean'],r['std'])
