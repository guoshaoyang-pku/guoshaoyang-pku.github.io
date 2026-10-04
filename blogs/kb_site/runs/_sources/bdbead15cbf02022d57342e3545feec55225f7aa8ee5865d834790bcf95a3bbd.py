import gzip,itertools,json,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def grab(p,s):
    m=re.search(p,s,re.S)
    return m.group(1).strip() if m else None
def history_pairs(h):
    out=[]
    for r in h:
        if r['family']!='univariate_regression':continue
        q=r['question'];parts=re.split(r'### Choice ([A-E])\n',q)
        if len(parts)<3:continue
        t=grab(r'training_steps:\s*(\d+)',parts[0]);b=grab(r'batch_size:\s*(\d+)',parts[0])
        if not t or not b:continue
        cand=[]
        for label,s in zip(parts[1::2],parts[2::2]):
            o=grab(r'- Optimizer:\s*(AdamW|Adam|RMSprop)\b',s)
            lr=grab(r'- Learning rate:\s*([\deE.+-]+)',s)
            if not o or not lr:continue
            model=grab(r'\*\*Model code\*\*\s*\x60\x60\x60python\s*(.*?)\x60\x60\x60',s)
            loss=grab(r'\*\*Loss\*\*.*?\x60\x60\x60python\s*(.*?)\x60\x60\x60',s)
            wd=grab(r'- Weight decay:\s*([\deE.+-]+)',s)
            beta=grab(r'- Betas:\s*(\[[^\]]+\])',s)
            if model is None or loss is None:continue
            # Equal executable model/loss bodies, ignoring formatting only.
            cand.append(dict(label=label,o=o,lr=float(lr),wd=float(wd) if wd else None,beta=beta,model=''.join(model.split()),loss=''.join(loss.split())))
        order=re.findall(r'[A-E]',r['answer'])
        for a,b_ in itertools.combinations(cand,2):
            if a['model']!=b_['model'] or a['loss']!=b_['loss']:continue
            if len(order)==1:
                if order[0] not in [a['label'],b_['label']]:continue
                winner=order[0]
            else:
                if a['label'] not in order or b_['label'] not in order:continue
                winner=min([a['label'],b_['label']],key=order.index)
            if {a['o'],b_['o']} in [{'Adam','RMSprop'},{'AdamW','RMSprop'}]:
                rms=a if a['o']=='RMSprop' else b_;ad=b_ if a['o']=='RMSprop' else a
                out.append(dict(q=r['question_id'],source=r['source'],epoch=r['epoch'],T=int(t),batch=int(b),kind='family',rms={k:v for k,v in rms.items() if k not in ['model','loss']},adam={k:v for k,v in ad.items() if k not in ['model','loss']},rms_win=winner==rms['label']))
            elif a['o']==b_['o']=='RMSprop' and a['wd']==b_['wd'] and a['lr']!=b_['lr']:
                hi=max([a,b_],key=lambda z:z['lr']);lo=min([a,b_],key=lambda z:z['lr'])
                out.append(dict(q=r['question_id'],source=r['source'],epoch=r['epoch'],T=int(t),batch=int(b),kind='rate',high=hi['lr'],low=lo['lr'],high_win=winner==hi['label']))
    return out
def lab_pairs(l):
    out=[]
    u=[r for r in l if r['family']=='univariate_regression' and not r['set_id'].startswith('exp:') and not r.get('excluded',False) and r['loss']=={'loss_id':'mse'}]
    for a,b in itertools.combinations(u,2):
        if a['set_id']!=b['set_id'] or a['budget']!=b['budget'] or a['model']!=b['model']:continue
        if {a['optimizer']['type'],b['optimizer']['type']} not in [{'Adam','RMSprop'},{'AdamW','RMSprop'}]:continue
        rms=a if a['optimizer']['type']=='RMSprop' else b;ad=b if a['optimizer']['type']=='RMSprop' else a
        out.append(dict(set_id=a['set_id'],dataset=a['dataset_id'],expression=a['dataset']['expression'],budget=a['budget'],model=a['model'],rms=rms['optimizer'],adam=ad['optimizer'],mean_rms=rms['mean'],mean_adam=ad['mean'],rms_win=rms['mean']<ad['mean'],ratio=rms['mean']/ad['mean'],failures=[rms.get('failed_seeds'),ad.get('failed_seeds')]))
    return out
def main():
    summary={}
    for name in ['history','checkpoint_history']:
        p=history_pairs(json.load(gzip.open(ROOT/(name+'.json.gz'),'rt')))
        (ROOT/(name+'_pairs.json')).write_text(json.dumps(p,indent=2))
        f=[x for x in p if x['kind']=='family' and x['rms']['lr']==x['adam']['lr']==3e-5]
        summary[name]=dict(low_equal_pairs=len(f),rms_wins=sum(x['rms_win'] for x in f),questions=len(set(x['q'] for x in f)),pairs=f,rms_rate=[x for x in p if x['kind']=='rate'])
    p=lab_pairs(json.load(gzip.open(ROOT/'lab.json.gz','rt')))
    (ROOT/'base_pairs.json').write_text(json.dumps(p,indent=2))
    summary['base']=dict(pairs=len(p),sets=len(set(x['set_id'] for x in p)),rms_wins=sum(x['rms_win'] for x in p),median_ratio=float(np.median([x['ratio'] for x in p])) if p else None,equal_low=[x for x in p if x['rms']['lr']==x['adam']['lr']==3e-5])
    lab=json.load(gzip.open(ROOT/'lab.json.gz','rt'))
    u=[r for r in lab if r['family']=='univariate_regression' and not r['set_id'].startswith('exp:') and not r.get('excluded',False) and r['loss'].get('loss_id')=='mse']
    broad={}
    for lr in [3e-5,1e-4,.001,.003]:
        pairs=[]
        for a,b in itertools.combinations(u,2):
            if a['set_id']!=b['set_id'] or a['budget']!=b['budget'] or a['budget']['training_steps']!=256:continue
            if {a['optimizer']['type'],b['optimizer']['type']} not in [{'Adam','RMSprop'},{'AdamW','RMSprop'}] or a['optimizer']['lr']!=lr or b['optimizer']['lr']!=lr:continue
            rms=a if a['optimizer']['type']=='RMSprop' else b;ad=b if rms is a else a
            pairs.append(dict(set_id=a['set_id'],rms=rms['candidate_id'],adam=ad['candidate_id'],mean_rms=rms['mean'],mean_adam=ad['mean'],rms_win=rms['mean']<ad['mean'],failures=[rms.get('failed_seeds'),ad.get('failed_seeds')]))
        broad[str(lr)]=dict(rms_wins=sum(x['rms_win'] for x in pairs),pairs=len(pairs),sets=len({x['set_id'] for x in pairs}),failure_pairs=sum(any(x['failures']) for x in pairs),measurements=pairs)
    summary['broad_unmatched_base']=broad

    (ROOT/'audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary))
if __name__=='__main__':main()
