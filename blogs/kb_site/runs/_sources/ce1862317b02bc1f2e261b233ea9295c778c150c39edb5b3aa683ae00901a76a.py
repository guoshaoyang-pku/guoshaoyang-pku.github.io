import json,glob,hashlib,os
import numpy as np
def pinned(v):
    p=v.get('repo_path')
    if not p:
        p='measurements/'+v['sha256']+os.path.splitext(v.get('name','x.json'))[1]
    assert hashlib.sha256(open(p,'rb').read()).hexdigest()==v['sha256'],p
    return p
cells={}
for p in glob.glob('gate_audit/fresh_*.json'):
    c=json.load(open(p));cells[(c['dataset_id'],c['optimizer']['weight_decay'])]=c
for p in glob.glob('gate_audit/raw_*_baseline.json'):
    raw=json.load(open(p));c=raw['results'][0];c['dataset_id']=raw['dataset'].split('/')[-1];c['optimizer']={'weight_decay':0}
    cells[(c['dataset_id'],0)]=c
manifest=json.load(open('measurements/a023b521b0324eac61dfda19461631b3803081449433c20875f5265f68beaa08.json'))
files={}
for n,v in manifest['measurements'].items():
    files[n]={**v,'repo_path':'measurements/'+v['sha256']+os.path.splitext(n)[1]}
for n,v in manifest['sources'].items():
    files['executed/'+n]={**v,'repo_path':'measurements/'+v['sha256']+'.py'}
files['results/execution_manifest.json']={'sha256':'a023b521b0324eac61dfda19461631b3803081449433c20875f5265f68beaa08','repo_path':'measurements/a023b521b0324eac61dfda19461631b3803081449433c20875f5265f68beaa08.json'}
summary=json.load(open(pinned(files['results/summary.json'])))
cells[('spiralcls_01a657',0)]={'dataset_id':'spiralcls_01a657','optimizer':{'weight_decay':0},'mean':summary['mean_test_ce'],'seed_results':summary['seed_results'],'failed_seeds':summary['failed_seeds'],'cached':False,'measurement_files':files,'source_provenance':{'status':'verified'},'process_provenance':{'status':'verified'}}
assert len(cells)==16
out={};pin_count=0
for (d,w),c in cells.items():
    for n,v in c['measurement_files'].items():pinned(v);pin_count+=1
    assert c['source_provenance']['status']=='verified' and c['process_provenance']['status']=='verified'
    spec=json.load(open(pinned(c['measurement_files']['candidate_spec.json'])))
    assert spec['optimizer']=={'type':'Adam','lr':.001,'weight_decay':w,'betas':[.9,.95]}
    assert spec['model']['depth']==3 and spec['model']['width']==48 and spec['model']['activation']=='silu' and spec['model']['layer_norm']==[True]*3 and not spec['model']['residual']
    assert spec['loss']=={'loss_id':'cross_entropy'} and spec['budget']['training_steps']==2048 and spec['budget']['batch_size']==16
    logs=[];prs=[];ces=[];errs=[];head=[];marg=[];stepchecks=[]
    for seed in range(10):
        p=pinned(c['measurement_files']['results/process/seed_%d.npz'%seed]);a=np.load(p)
        j=json.load(open(pinned(c['measurement_files']['results/process/seed_%d.json'%seed])))
        z=a['logits'].astype(float);y=a['targets'];pr=z.argmax(1);m=z[np.arange(len(y)),y]-z[np.arange(len(y)),1-y]
        ce=float(np.logaddexp(0,-m).mean())
        assert np.array_equal(a['predictions'],pr) and np.array_equal(a['errors'],pr!=y)
        assert np.allclose(a['true_class_margins'],m,atol=1e-6)
        assert abs(ce-c['seed_results'][seed]['final_test_ce'])<2e-7
        ces.append(c['seed_results'][seed]['final_test_ce']);errs.append(int((pr!=y).sum()));logs.append(z);prs.append(pr);marg.append(m)
        stepchecks.extend(p['adam_step_check']['passed'] for s in j['steps'] for p in s['parameters'] if p['active'])
        hp=[p for p in j['steps'][-1]['parameters'] if p['role']=='head']
        g=np.sqrt(sum(p['data_gradient_norm']**2 for p in hp));dec=np.sqrt(sum(p['decay_gradient_norm']**2 for p in hp))
        head.append({'data_gradient_norm':g,'decay_gradient_norm':dec,'ratio':float(dec/g) if g else None})
        if seed==0:inputs=j['inputs'];stream=j['minibatch_stream_sha256']
    assert all(stepchecks)
    out.setdefault(d,{})[str(w)]={'ce':ces,'errors':errs,'head_step2048':head,'margin_median_per_seed':[float(np.median(m)) for m in marg],'inputs':inputs,'stream':stream,'files':c['measurement_files'],'failed':c['failed_seeds'],'cached':c['cached']}
    c['_prs']=prs;c['_marg']=marg
contrasts={}
for d in sorted(out):
    b=cells[(d,0)];cc={}
    for wa,wb in [(1e-5,0),(1e-4,0),(.001,0),(1e-4,1e-5)]:
        a=cells[(d,wa)];bb=cells[(d,wb)]
        diff=np.array(out[d][str(wa)]['ce'])-np.array(out[d][str(wb)]['ce'])
        half=2.2621571627409915*float(diff.std(ddof=1))/np.sqrt(10)
        disagreements=[int((a['_prs'][s]!=bb['_prs'][s]).sum()) for s in range(10)]
        cc[str(wa)+'-'+str(wb)]={'mean':float(diff.mean()),'ci95':[float(diff.mean()-half),float(diff.mean()+half)],'positive':int((diff>0).sum()),'negative':int((diff<0).sum()),'disagreements':disagreements}
    for w in [1e-5,1e-4,.001]:
        assert out[d][str(w)]['inputs']==out[d]['0']['inputs']
        assert out[d][str(w)]['stream']==out[d]['0']['stream']
    contrasts[d]=cc
assert len({out[d]['0']['inputs']['train_x']['sha256'] for d in out})==4
result={'cells':out,'contrasts':contrasts,'verified_file_references':pin_count,'all_step_checks':True}
json.dump(result,open('gate_audit/results.json','w'))
for d in sorted(out):
    print(d,'CE',[np.mean(out[d][str(w)]['ce']) for w in [0,1e-5,1e-4,.001]],'errors',[out[d][str(w)]['errors'] for w in [0,1e-5,1e-4,.001]])
    print('contrasts',contrasts[d])
print('verified',pin_count)

# Surviving factorials are measurements, never additional fresh replications.
import collections
prior=[json.load(open(p)) for p in glob.glob('gate_audit/surviving_*.json')]
prior=[c for c in prior if c.get('depth')==3 and c.get('width')==48 and c.get('activation')=='silu' and not c.get('residual') and c.get('n_ln')==3 and c['optimizer'].get('lr')==.001 and c['budget']['training_steps'] in [256,2048]]
old_n=0
for c in prior:
    for name,v in c.get('measurement_files',{}).items():
        path=v.get('repo_path','measurements/'+v['sha256']+os.path.splitext(name)[1])
        assert hashlib.sha256(open(path,'rb').read()).hexdigest()==v['sha256']
        old_n+=1
r64={}
for d in ['spiralcls_00f3c5','spiralcls_134b58']:
    cs={c['optimizer']['weight_decay']:c for c in prior if c['dataset_id']==d and c['optimizer']['type']=='Adam' and c['budget']['training_steps']==2048}
    diff=np.array([s['final_test_ce'] for s in cs[.0001]['seed_results']])-np.array([s['final_test_ce'] for s in cs[1e-5]['seed_results']])
    half=2.262157162741*diff.std(ddof=1)/np.sqrt(10)
    r64[d]={'mean':float(diff.mean()),'ci95':[float(diff.mean()-half),float(diff.mean()+half)],'positive':int((diff>0).sum())}
assert len({out[d]['0']['inputs']['test_x']['sha256'] for d in out})==4
json.dump({'n_cells':len(prior),'verified_hashes':old_n,'status_counts':dict(collections.Counter(c.get('source_provenance',{}).get('status','missing') for c in prior)),'R64_contrasts':r64},open('gate_audit/inherited_source_audit.json','w'))
print('inherited hash references',old_n,'R64',r64)
