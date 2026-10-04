import os,json,gzip,glob,math,collections,csv,hashlib
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(HERE)
def read(name):
    with gzip.open(os.path.join(HERE,name+'.json.gz'),'rt') as f:return json.load(f)
def delta(x):
    o=x['optimizer'];t=x['budget']['training_steps'];m=o.get('momentum',0);lr=o['lr']
    if o['type']=='SGD':return lr*t*(.03 if x['family'].endswith('regression') and m else .01)/(1-m)
    return 2*lr*math.sqrt(t) if o['type']=='Adagrad' else lr*t
def summary(ps):
    return dict(wins=sum(a['mean']<b['mean'] for a,b in ps),n=len(ps),sets=len({a['set_id'] for a,b in ps}),ratio=float(np.median([a['mean']/b['mean'] for a,b in ps])) if ps else None)
def old_audit():
    lab=read('lab');groups=collections.defaultdict(list)
    for x in lab:
        if not x['set_id'].startswith('exp:') and not x.get('excluded') and x.get('model_type')=='mlp' and x['loss']['loss_id'] in ['cross_entropy','mse']:groups[x['set_id']].append(x)
    zero=[];norm=[];terminal=collections.defaultdict(list)
    for xs in groups.values():
        for a in xs:
            if a['optimizer']['type']=='SGD':
                for b in xs:
                    if b['optimizer']['type']=='SGD' or delta(b)>=.1 or delta(a)<3*delta(b):continue
                    if a.get('n_ln',0)>=1:norm.append((a,b))
                    elif a['depth']>=3 and a['activation'] in ['silu','gelu']:zero.append((a,b))
            if a['family'] not in ['xor_classification','synthetic_tabular_classification','spiral_classification'] or a['residual'] or a['depth']<2 or a['loss']['loss_id']!='cross_entropy' or not a['layer_norm'][-1]:continue
            for b in xs:
                if b['residual'] or b['depth']<2 or b['loss']['loss_id']!='cross_entropy' or b['layer_norm'][-1] or a['optimizer']['type']!=b['optimizer']['type']:continue
                terminal[('type',a['optimizer']['type'])].append((a,b))
                if a['optimizer']['lr']==b['optimizer']['lr']:terminal[('lr',a['optimizer']['type'])].append((a,b))
                if a['optimizer']==b['optimizer']:terminal[('full',a['optimizer']['type'])].append((a,b))
    result={'zero':summary(zero),'normalized':summary(norm),'zero_residual':{str(r):summary([(a,b) for a,b in zero if a['residual']==r]) for r in [False,True]},'zero_family':{fam:summary([(a,b) for a,b in zero if a['family']==fam]) for fam in sorted({a['family'] for a,b in zero})},'terminal':{'/'.join(k):summary(v) for k,v in terminal.items()}}
    assert result['zero']['n']==30 and result['zero']['wins']==11
    assert result['zero_residual']['False']['wins']==3 and result['zero_residual']['True']['wins']==8
    with open(os.path.join(HERE,'old_pairs.json'),'w') as f:json.dump({'zero':[{'a':a,'b':b} for a,b in zero],'terminal':{'/'.join(k):[{'a':a,'b':b} for a,b in v] for k,v in terminal.items()}},f)
    return result
def opt(c):
    o=c['optimizer'];return 'Adam' if o['type']=='Adam' else 'SGD9' if o.get('momentum',0) else 'SGD0'
def kind(c):
    mask=c['model']['layer_norm']
    if not any(mask):return 'zero'
    if all(mask):return 'all'
    if sum(mask)==1 and mask[-1]:return 'terminal'
    if sum(mask)==1 and mask[0]:return 'early'
    raise ValueError(mask)
def identity(c):return json.dumps(c,sort_keys=True)
def contrast(a,b,tag,source,dataset):
    ca,ra=a;cb,rb=b
    key='final_test_ce' if ra['metric']=='test_ce' else 'final_test_mse'
    sa={s['seed']:s[key] for s in ra.get('seed_results',[]) if not s['failed']}
    sb={s['seed']:s[key] for s in rb.get('seed_results',[]) if not s['failed']}
    seeds=sorted(sa.keys()&sb.keys());diff=np.array([sa[s]-sb[s] for s in seeds])
    ci=None
    if len(diff)==10:
        se=float(diff.std(ddof=1)/np.sqrt(10));ci=[float(diff.mean()-2.262157*se),float(diff.mean()+2.262157*se)]
    return dict(source=source,dataset=dataset,family=dataset.split('/')[0],contrast=tag,depth=ca['model']['depth'],residual=ca['model']['residual'],optimizer=opt(ca),kind=kind(ca),a=ca,b=cb,mean_a=ra['mean'],sd_a=ra['std'],mean_b=rb['mean'],sd_b=rb['std'],ratio=ra['mean']/rb['mean'],margin=ra['mean']-rb['mean'],win=ra['mean']<rb['mean'],seed_wins=int(sum(diff<0)),paired_n=len(seeds),paired_ci=ci,failed_a=ra.get('failed_seeds'),failed_b=rb.get('failed_seeds'))
def new_audit():
    rows=[];configs={};sources=[];returned=0;cached=0;failed=0;executed=0
    for path in sorted(glob.glob(os.path.join(ROOT,'experiments','*.json'))):
        e=json.load(open(path));dataset=e['input']['dataset'];source='exp:'+os.path.basename(path).split('.')[0];pairs=list(zip(e['input']['candidates'],e['result']['results']));sources.append(source)
        for c,r in pairs:
            returned+=1;cached+=int(r.get('cached',False));failed+=r.get('failed_seeds',0);executed+=0 if r.get('cached') else r.get('n_seeds',0)
            configs[(dataset,identity(c))]=r
            assert r.get('n_seeds')==10 and len(r.get('seed_results',[]))==10 and r.get('failed_seeds')==0 and math.isfinite(r['mean'])
            for name,meta in r.get('measurement_files',{}).items():
                if name!='candidate_spec.json':continue
                p=os.path.dirname(meta['path']);dest=os.path.join(HERE,'recipes',os.path.basename(p));os.makedirs(dest,exist_ok=True)
                curve_path=r['measurement_files']['results/curves.npz']['path']
                local_curve=os.path.join(dest,'curves.npz')
                curve_bytes=open(local_curve if os.path.exists(local_curve) else curve_path,'rb').read()
                assert hashlib.sha256(curve_bytes).hexdigest()==r['measurement_files']['results/curves.npz']['sha256']
                with open(os.path.join(dest,'curves.npz'),'wb') as f:f.write(curve_bytes)
                z=np.load(os.path.join(dest,'curves.npz'),allow_pickle=False)
                assert z['curves'].shape==(10,c['budget']['training_steps'])
                assert np.allclose(z['curves'][:,-1].mean(),r['mean'],rtol=1e-7,atol=1e-8)
                for fn in ['model.py','optimizer.py','train.py','loss.py','candidate_spec.json']:
                    local_recipe=os.path.join(dest,fn)
                    content=open(local_recipe if os.path.exists(local_recipe) else os.path.join(p,fn),'rb').read()
                    with open(os.path.join(dest,fn),'wb') as f:f.write(content)
        for a in pairs:
            ca=a[0];ma=ca['model']
            for b in pairs:
                cb=b[0];mb=cb['model']
                if ca['loss']!=cb['loss'] or ca['budget']!=cb['budget']:continue
                if ma['depth']==mb['depth'] and ma['residual']==mb['residual'] and opt(ca)==opt(cb):
                    if (kind(ca),kind(cb)) in [('terminal','early'),('terminal','zero'),('early','zero'),('all','terminal'),('all','zero')]:rows.append(contrast(a,b,kind(ca)+'/'+kind(cb),source,dataset))
                if ma['depth']==mb['depth'] and ma['residual'] and not mb['residual'] and opt(ca)==opt(cb) and kind(ca)==kind(cb):rows.append(contrast(a,b,'residual/plain',source,dataset))
                if ma['depth']>mb['depth'] and ma['residual']==mb['residual'] and opt(ca)==opt(cb) and kind(ca)==kind(cb):rows.append(contrast(a,b,'depth'+str(ma['depth'])+'/'+str(mb['depth']),source,dataset))
                if ma==mb and (opt(ca),opt(cb)) in [('SGD9','Adam'),('SGD0','Adam'),('SGD9','SGD0')]:rows.append(contrast(a,b,opt(ca)+'/'+opt(cb),source,dataset))
    # A cached repeated pair is not a new replication; keep one source per exact contrast.
    dedup={}
    for row in rows:dedup.setdefault((row['dataset'],row['contrast'],identity(row['a']),identity(row['b'])),row)
    rows=list(dedup.values())
    groups=collections.defaultdict(list)
    for x in rows:groups[(x['contrast'],x['family'],x['depth'],x['residual'],x['optimizer'],x['kind'])].append(x)
    table=[]
    for k,xs in sorted(groups.items()):table.append(dict(contrast=k[0],family=k[1],depth=k[2],residual=k[3],optimizer=k[4],kind=k[5],n=len(xs),wins=sum(x['win'] for x in xs),ratios=[x['ratio'] for x in sorted(xs,key=lambda x:x['dataset'])],margins=[x['margin'] for x in sorted(xs,key=lambda x:x['dataset'])],paired_ci=[x['paired_ci'] for x in sorted(xs,key=lambda x:x['dataset'])]))
    with open(os.path.join(HERE,'contrasts.json'),'w') as f:json.dump(rows,f,indent=2)
    with open(os.path.join(HERE,'regime_table.json'),'w') as f:json.dump(table,f,indent=2)
    fields=['contrast','family','depth','residual','optimizer','kind','n','wins','ratios','margins','paired_ci']
    with open(os.path.join(HERE,'regime_table.csv'),'w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    return dict(sources=sources,returned=returned,cached=cached,unique_configs=len(configs),new_seed_evaluations=executed,failed=failed,contrasts=len(rows),table=table)

def interactions():
    rows=[]
    for path in sorted(glob.glob(os.path.join(ROOT,'experiments','*.json'))):
        e=json.load(open(path))
        if not e['why'].startswith('Exploratory paired optimizer-by-placement'):continue
        pairs=list(zip(e['input']['candidates'],e['result']['results']))
        assert len(pairs)==12 and all(r['cached'] for c,r in pairs)
        d={(c['model']['residual'],opt(c),kind(c)):(c,r) for c,r in pairs}
        for res in [False,True]:
            for op in ['Adam','SGD9']:
                ordered=[d[(res,o,k)] for o,k in [(op,'terminal'),(op,'early'),('SGD0','terminal'),('SGD0','early')]]
                key='final_test_ce' if ordered[0][1]['metric']=='test_ce' else 'final_test_mse'
                vals=[np.array([q[key] for q in r['seed_results']]) for c,r in ordered]
                diff=(vals[0]-vals[1])-(vals[2]-vals[3])
                logs=np.log(vals[0]/vals[1])-np.log(vals[2]/vals[3])
                def ci(v):
                    se=float(v.std(ddof=1)/np.sqrt(10));return [float(v.mean()-2.262157*se),float(v.mean()+2.262157*se)]
                rows.append(dict(source='exp:'+os.path.basename(path).split('.')[0],dataset=e['input']['dataset'],residual=res,optimizer=op,ratio=ordered[0][1]['mean']/ordered[1][1]['mean'],sgd0_ratio=ordered[2][1]['mean']/ordered[3][1]['mean'],absolute_interaction=float(diff.mean()),absolute_ci=ci(diff),mean_log_interaction=float(logs.mean()),log_ci=ci(logs)))
    with open(os.path.join(HERE,'interactions.json'),'w') as f:json.dump(rows,f,indent=2)
    return rows
def curve_audit():
    rows={}
    for path in sorted(glob.glob(os.path.join(ROOT,'experiments','*.json'))):
        e=json.load(open(path));dataset=e['input']['dataset']
        for c,r in zip(e['input']['candidates'],e['result']['results']):
            ident=(dataset,identity(c))
            if ident in rows:continue
            folder=os.path.basename(os.path.dirname(r['measurement_files']['candidate_spec.json']['path']))
            z=np.load(os.path.join(HERE,'recipes',folder,'curves.npz'),allow_pickle=False)
            v=z['curves'];assert int(z['samples'][0])==c['budget']['batch_size']
            key='final_test_ce' if r['metric']=='test_ce' else 'final_test_mse'
            assert np.allclose(v[:,-1],[q[key] for q in r['seed_results']],rtol=1e-7,atol=1e-8)
            rows[ident]=dict(dataset=dataset,source='exp:'+os.path.basename(path).split('.')[0],depth=c['model']['depth'],residual=c['model']['residual'],optimizer=opt(c),kind=kind(c),first_update=float(v[:,0].mean()),endpoint=float(v[:,-1].mean()),change=float((v[:,-1]-v[:,0]).mean()),seed_changes=(v[:,-1]-v[:,0]).tolist())
    with open(os.path.join(HERE,'curve_audit.json'),'w') as f:json.dump(list(rows.values()),f,indent=2)
    return len(rows)
def unified_table(result):
    rows=[]
    def oldrow(cohort,contrast,optimizer,family,residual,depth,stat,definition,source,verified=True):
        rows.append(dict(cohort=cohort,contrast=contrast,optimizer=optimizer,family=family,residual=residual,depth=depth,n=stat['n'],wins=stat['wins'],sets=stat.get('sets'),median_ratio=stat.get('ratio'),ratios=None,margins=None,paired_ci=None,definition=definition,source=source,verified=verified))
    oldrow('legacy reported','terminal/nonterminal','SGD','classification',False,'>=2',dict(wins=31,n=31,sets=14,ratio=.69),'Original selection filter underidentified','initial KB K1059',False)
    oldrow('legacy reported','terminal/nonterminal','Adam','classification',False,'>=2',dict(wins=70,n=102,ratio=.92),'Original selection filter underidentified','initial KB K1059',False)
    for key,stat in result['old']['terminal'].items():
        match,op=key.split('/');oldrow('explicit base','terminal/nonterminal',op,'classification',False,'>=2',stat,'Nonexcluded base MLP,plain CE,same set;matched '+match+';no exact architecture match','old_pairs.json')
    oldrow('explicit base','terminal/nonterminal','SGD','classification',False,'>=2',dict(wins=0,n=0),'Full optimizer configuration equality;other base terminal filters','old_pairs.json')
    definition='Nonexcluded base MLP,plain CE/MSE,SGD Delta>=3*adaptive,rivalDelta<.1;SGD deep SiLU/GELU zeroLN;no model/optimizer/tail match'
    oldrow('explicit base','SGD/adaptive','SGD','mixed','mixed','>=3',result['old']['zero'],definition,'old_pairs.json')
    for res,stat in result['old']['zero_residual'].items():oldrow('explicit base','SGD/adaptive','SGD','mixed',res,'>=3',stat,definition,'old_pairs.json')
    for fam,stat in result['old']['zero_family'].items():oldrow('explicit base','SGD/adaptive','SGD',fam,'mixed','>=3',stat,definition,'old_pairs.json')
    oldrow('explicit base','SGD/adaptive','SGD','mixed','mixed','mixed',result['old']['normalized'],'Same calibrated optimizer contrast butSGD>=1LN,no deep/smooth filter','analysis.json')
    for x in result['new']['table']:
        rows.append(dict(cohort='matched controls',contrast=x['contrast']+':'+x['kind'],optimizer=x['optimizer'],family=x['family'],residual=x['residual'],depth=x['depth'],n=x['n'],wins=x['wins'],sets=x['n'],median_ratio=None,ratios=x['ratios'],margins=x['margins'],paired_ci=x['paired_ci'],definition='w24 SiLU default init,wd0,bs16,T256;fixed within-set contrasts;split-specific ratios are not pooled',source='contrasts.json',verified=True))
    with open(os.path.join(HERE,'unified_regime_table.json'),'w') as f:json.dump(rows,f,indent=2)
    with open(os.path.join(HERE,'unified_regime_table.csv'),'w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return len(rows)

if __name__=='__main__':
    result={'old':old_audit(),'new':new_audit()}
    result['interactions']=interactions()
    result['curve_configs']=curve_audit()
    result['unified_rows']=unified_table(result)
    with open(os.path.join(HERE,'analysis.json'),'w') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='table'} if k=='new' else v) for k,v in result.items() if k not in ['interactions']},indent=2))
