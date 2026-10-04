import collections, gzip, json, math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent.parent

def canonical(c):
    c=json.loads(json.dumps(c)); o=c['optimizer']; typ=o['type']
    allowed={'type','lr','weight_decay'} | ({'momentum'} if typ=='SGD' else {'betas'} if typ in ['Adam','AdamW'] else set())
    c['optimizer']={k:v for k,v in o.items() if k in allowed}
    if typ=='SGD':c['optimizer'].setdefault('momentum',0)
    if typ in ['Adam','AdamW']:c['optimizer'].setdefault('betas',[.9,.999])
    c['loss']={'loss_id':c['loss']['loss_id']}
    c['budget']={k:c['budget'][k] for k in ['training_steps','batch_size']}
    return c

def key(c):return json.dumps(c,sort_keys=True)

def interval(g,t):
    ng=g['n'];nt=t['n'];d=g['mean']-t['mean']
    if min(ng,nt)<9 or not math.isfinite(d):return None
    h=2.8*(g['std']/math.sqrt(ng)+t['std']/math.sqrt(nt))
    return [d-h,d+h]

records={}; artifacts=[]
for p in sorted((ROOT/'experiments').glob('*.json')):
    a=json.loads(p.read_text());artifacts.append({'source':'exp:'+p.stem,'why':a.get('why'),'result_count':len(a.get('result',{}).get('results',[]))})
    for r in a.get('result',{}).get('results',[]):
        c=canonical(r['candidate']);ds=a['input']['dataset'].split('/')[-1];signature=(ds,key(c))
        z={'dataset_id':ds,'dataset':a['dataset_params'],'candidate':c,'mean':r['mean'],'std':r['std'],'failed':r.get('failed_seeds'),'n':a['seeds']-r.get('failed_seeds',a['seeds']),'sources':['exp:'+p.stem]}
        if signature in records:
            old=records[signature]
            assert old['mean']==z['mean'] and old['std']==z['std'],signature
            old['sources']+=z['sources']
        else:records[signature]=z
rs=list(records.values());pairs=[]
for g in rs:
    m=g['candidate']['model'];o=g['candidate']['optimizer'];b=g['candidate']['budget']
    if (m['type'],m['d_model'],m['num_layers'])!=('gru_lm',64,1):continue
    for t in rs:
        mt=t['candidate']['model']
        if (mt['type'],mt['d_model'],mt['num_layers'])!=('transformer_lm',32,4):continue
        if any(k in mt for k in ['head_scale','position_embedding']):continue
        if g['dataset_id']!=t['dataset_id']:continue
        if any(g['candidate'][k]!=t['candidate'][k] for k in ['optimizer','budget','loss']):continue
        ci=interval(g,t)
        pairs.append({'dataset_id':g['dataset_id'],'alpha':g['dataset']['alpha'],'optimizer':o,'T':b['training_steps'],'FF':mt['d_ff'],'heads':mt['num_heads'],'D':o['lr']*b['training_steps'],'g':g['mean'],'t':t['mean'],'g_sd':g['std'],'t_sd':t['std'],'n_g':g['n'],'n_t':t['n'],'difference':g['mean']-t['mean'],'ci':ci,'sources':g['sources']+t['sources']})

selected=json.loads((ROOT/'audit/selected.json').read_text())+json.loads((ROOT/'audit/selected_v32.json').read_text());entropies=[]
for x in selected:
    d=x['dataset'];V=d['vocab_size'];rng=np.random.default_rng(d['table_seed']);z=rng.standard_normal((V,V))*d['alpha'];z-=z.max(1,keepdims=True);P=np.exp(z);P/=P.sum(1,keepdims=True)
    pi=np.full(V,1/V)
    for _ in range(256):pi=(pi[:,None]*P).sum(axis=0);pi/=pi.sum()
    assert np.isfinite(pi).all() and abs(pi.sum()-1)<1e-12
    H=-float((pi*np.log(pi)).sum());HC=-float((pi[:,None]*P*np.log(P)).sum())
    rng=np.random.default_rng(d['sequence_seed']+1);counts=np.zeros(V);joint=np.zeros((V,V));oracle=0
    for _ in range(d['test_size']):
        s=int(rng.choice(V,p=pi))
        for _ in range(d['context_length']):
            y=int(rng.choice(V,p=P[s]));counts[y]+=1;joint[s,y]+=1;oracle-=math.log(P[s,y]);s=y
    p=counts/counts.sum();eh=-float((p[p>0]*np.log(p[p>0])).sum());j=joint/joint.sum();rows=j.sum(1);ech=0
    for i in range(V):
        if rows[i]>0:
            q=j[i]/rows[i];ech-=float((j[i,q>0]*np.log(q[q>0])).sum())
    entropies.append({'dataset_id':x['dataset_id'],'alpha':d['alpha'],'H_marginal_law':H,'H_conditional_law':HC,'I_bigram_law':H-HC,'H_test_plugin':eh,'H_conditional_test_plugin':ech,'oracle_test_CE':oracle/counts.sum(),'source':'numpy reconstruction of documented bigram synthesis using allowed dataset parameters; not model logits'})

progress=[]
for r in rs:
    c=r['candidate'];o=c['optimizer'];m=c['model']
    if any(k in m for k in ['head_scale','position_embedding']) or o['lr']==0:continue
    initial=next((z for z in rs if z['dataset_id']==r['dataset_id'] and z['candidate']['model']==m and z['candidate']['optimizer']['type']=='SGD' and z['candidate']['optimizer']['lr']==0),None)
    H=next((e['H_marginal_law'] for e in entropies if e['dataset_id']==r['dataset_id']),None)
    progress.append({'dataset_id':r['dataset_id'],'candidate':c,'CE0':initial['mean'] if initial else None,'CEfinal':r['mean'],'progress':initial['mean']-r['mean'] if initial else None,'marginal_baseline_gain':H-r['mean'] if H is not None else None})

frozen=json.load(gzip.open(ROOT/'audit/lab.json.gz','rt'));sets=collections.defaultdict(list)
for x in frozen:
    if x['family']=='bigram_lm':sets[x['set_id']].append(x)
fp=[]
for sid,xs in sets.items():
 for g in xs:
    if (g['model_type'],g['d_model'],g['num_layers'])!=('gru_lm',64,1):continue
    for t in xs:
        if (t['model_type'],t['d_model'],t['num_layers'])!=('transformer_lm',32,4):continue
        if any(g[k]!=t[k] for k in ['optimizer','budget','loss']):continue
        o=g['optimizer'];D=o['lr']*g['budget']['training_steps']
        if (o['type']=='SGD' and .0001<=o['lr']<=.003) or (o['type'] in ['Adam','AdamW'] and .01<=D<=.064):fp.append({'g':g,'t':t})

summary={'artifacts':artifacts,'distinct_configurations':len(rs),'scheduled_seeds':sum(z['n']+(z['failed'] or 0) for z in rs),'failed_seeds':sum(z['failed'] or 0 for z in rs),'finite_metrics':sum(z['n'] for z in rs),'frozen_pair_count':len(fp),'frozen_set_count':len({p['g']['set_id'] for p in fp}),'entropies':entropies,'pairs':pairs,'progress':progress,'records':rs}
(ROOT/'audit/results.json').write_text(json.dumps(summary,indent=2))
(ROOT/'audit/frozen_pairs.json').write_text(json.dumps(fp,indent=2))
print('COUNTS',{k:summary[k] for k in ['distinct_configurations','scheduled_seeds','failed_seeds','finite_metrics','frozen_pair_count','frozen_set_count']})
for typ in ['SGD','Adam','AdamW']:
 for ff in [64,128]:
    ps=[p for p in pairs if p['optimizer']['type']==typ and p['optimizer']['lr']>0 and p['FF']==ff]
    if ps:print('PAIR_SUMMARY',typ,ff,len(ps),'datasets',len({p['dataset_id'] for p in ps}),'G_lower',sum(p['difference']<0 for p in ps),'G_CI',sum(p['ci'] is not None and p['ci'][1]<0 for p in ps),'TF_CI',sum(p['ci'] is not None and p['ci'][0]>0 for p in ps),'margin_range',[min(p['difference'] for p in ps),max(p['difference'] for p in ps)])
print('ENTROPIES',entropies)
