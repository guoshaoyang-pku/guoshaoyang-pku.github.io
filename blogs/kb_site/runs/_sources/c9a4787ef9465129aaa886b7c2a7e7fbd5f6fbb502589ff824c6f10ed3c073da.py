import json, pathlib, hashlib, numpy as np, csv
ROOT=pathlib.Path(__file__).resolve().parent
labels={(4,(0,1,1,0)):'B',(4,(0,1,0,1)):'R',(4,(0,1,1,1)):'N',(3,(0,1,1)):'D',(3,(0,1,0)):'U'}
cells=[]
for x in json.load(open(ROOT/'prior_cells.json')):
    cells.append(dict(x,topology='P',opt='A',label=labels[(x['depth'],tuple(x['layer_norm']))]))
for filename in sorted(ROOT.glob('new_*.json'))+sorted(ROOT.glob('residual_*.json')):
    receipt=json.load(open(filename))
    if not isinstance(receipt,dict) or 'results' not in receipt: continue
    dataset=receipt['dataset'].split('/')[-1]
    for r in receipt['results']:
        v=r['variant']; opt='A' if v['optimizer.type']=='AdamW' else ('S0' if v['optimizer.momentum']==0 else 'S9')
        cells.append(dict(r,dataset_id=dataset,topology='R' if v.get('model.residual',False) else 'P',opt=opt,label=labels[(v['model.depth'],tuple(v['model.layer_norm']))]))
archive=ROOT/'artifacts'; archive.mkdir(exist_ok=True)
def artifact_path(meta):
    p=pathlib.Path(meta['path'])
    local=archive/(meta['sha256']+p.suffix)
    return local if local.exists() else p
hashes={}; rows=[]; index={}
for c in cells:
    assert c['failed_seeds']==0 and not c['excluded'] and c['n_seeds']==10
    vals=np.array([s['final_test_ce'] for s in c['seed_results']])
    assert np.all(np.isfinite(vals)) and abs(vals.mean()-c['mean'])<1e-10
    assert np.array_equal([s['seed'] for s in c['seed_results']],np.arange(10))
    assert min(abs(vals.std(ddof=d)-c['std']) for d in [0,1])<1e-10
    assert c['source_provenance']['status']=='verified'
    for name,meta in c['measurement_files'].items():
        p=pathlib.Path(meta['path']); b=artifact_path(meta).read_bytes(); digest=hashlib.sha256(b).hexdigest()
        assert digest==meta['sha256']
        local=archive/(digest+p.suffix)
        if not local.exists(): local.write_bytes(b)
        hashes[digest]={'name':name,'source':str(p),'local':str(local.relative_to(ROOT)),'bytes':len(b)}
    z=np.load(artifact_path(c['measurement_files']['results/curves.npz']))
    assert z['curves'].shape==(10,512) and int(z['batch_size'])==64
    assert np.array_equal(z['samples'],np.arange(1,513)*64)
    assert np.allclose(z['curves'][:,-1],vals,rtol=0,atol=1e-12)
    spec=json.load(open(artifact_path(c['measurement_files']['candidate_spec.json'])))
    assert spec['model']['width']==128 and spec['model']['activation']=='gelu'
    assert spec['budget']['training_steps']==512 and spec['budget']['batch_size']==64
    assert spec['optimizer']['weight_decay']==.0001
    assert spec['loss']['loss_id']=='cross_entropy'
    row={'dataset':c['dataset_id'],'topology':c['topology'],'opt':c['opt'],'label':c['label'],'mean':c['mean'],'sd':c['std'],'post_update1_mean':float(z['curves'][:,0].mean()),'cached':c.get('cached',False)}
    rows.append(row); index[(c['dataset_id'],c['topology'],c['opt'],c['label'])]=vals
def summarize(a):
    mean=float(a.mean()); se=float(a.std(ddof=1)/np.sqrt(len(a)))
    return {'difference':mean,'ci95':[mean-2.2621571627409915*se,mean+2.2621571627409915*se],'negative_seeds':int(np.sum(a<0)),'n':len(a)}
contrasts=[]
for ds in sorted(set(r['dataset'] for r in rows)):
    for topology in ['P','R']:
        for opt in ['A','S0','S9']:
            for a,b in [('N','B'),('R','B'),('D','U'),('B','D'),('R','D'),('N','D')]:
                k1=(ds,topology,opt,a); k2=(ds,topology,opt,b)
                if k1 in index and k2 in index:
                    contrasts.append(dict(dataset=ds,topology=topology,opt=opt,contrast=a+'-'+b,**summarize(index[k1]-index[k2])))
            if topology=='P' and opt!='A':
                for a,b in [('N','B'),('R','B'),('D','U')]:
                    dif=(index[(ds,'P',opt,a)]-index[(ds,'P',opt,b)])-(index[(ds,'P','A',a)]-index[(ds,'P','A',b)])
                    contrasts.append(dict(dataset=ds,topology='P',opt=opt,contrast='optimizer_interaction_'+a+'-'+b,**summarize(dif)))
            if topology=='R' and (ds,'R',opt,'N') in index:
                dif=(index[(ds,'R',opt,'N')]-index[(ds,'R',opt,'B')])-(index[(ds,'P',opt,'N')]-index[(ds,'P',opt,'B')])
                contrasts.append(dict(dataset=ds,topology='RxP',opt=opt,contrast='placement_interaction_N-B',**summarize(dif)))
                for label in ['B','N','D']:
                    contrasts.append(dict(dataset=ds,topology='R-P',opt=opt,contrast=label,**summarize(index[(ds,'R',opt,label)]-index[(ds,'P',opt,label)])))
with open(ROOT/'cells.csv','w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
json.dump(contrasts,open(ROOT/'contrasts.json','w'),indent=2)
json.dump(hashes,open(ROOT/'hash_audit.json','w'),indent=2)
json.dump(rows,open(ROOT/'cells.json','w'),indent=2)
print('cells',len(cells),'unique verified artifacts',len(hashes),'new cells',sum(r['opt']!='A' or r['topology']=='R' for r in rows))
print('FINAL CE TABLE')
for ds in ['spiralcls_016ff9','spiralcls_01793b','spiralcls_05d021','spiralcls_06bcc2']:
    for top in ['P','R']:
        for opt in ['A','S0','S9']:
            selected=[next((r for r in rows if (r['dataset'],r['topology'],r['opt'],r['label'])==(ds,top,opt,l)),None) for l in ['B','R','N','D','U']]
            if selected[0]: print(ds,top,opt,[round(r['mean'],9) if r else None for r in selected])
print('PLACEMENT AND DEPTH CONTRASTS')
for c in contrasts:
    if c['contrast'] in ['N-B','R-B','D-U','B-D','R-D','N-D'] or c['topology']=='RxP': print(json.dumps(c))
print('FIRST UPDATE MEANS')
for r in rows:
    if r['dataset']=='spiralcls_016ff9': print(r['topology'],r['opt'],r['label'],r['post_update1_mean'],r['mean'])
