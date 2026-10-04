import json, itertools, numpy as np

def analyze(lab, history):
    eligible = [x for x in lab if x['family']=='spiral_classification' and x['budget']['training_steps']==256 and 2<=x['dataset']['spiral_turns']<=3 and x['optimizer']['type']=='RMSprop' and x.get('source')!='experiment']
    pairs=[]
    for a,b in itertools.combinations(eligible,2):
        if a['set_id']!=b['set_id'] or a['optimizer']['lr']!=b['optimizer']['lr']: continue
        if a['activation'] in ['relu','leaky_relu']: a,b=b,a
        if a['activation'] not in ['gelu','silu'] or b['activation'] not in ['relu','leaky_relu']: continue
        pairs.append(dict(set_id=a['set_id'], smooth=a['candidate_id'],rectifier=b['candidate_id'],lr=a['optimizer']['lr'],ratio=a['mean']/b['mean'],smooth_wd=a['optimizer']['weight_decay'],rectifier_wd=b['optimizer']['weight_decay']))
    aggregate={}
    for label,p in [('high',[x for x in pairs if x['lr']>=.001]),('low',[x for x in pairs if x['lr']<=.0003])]:
        aggregate[label]=dict(pairs=len(p),sets=len(set(x['set_id'] for x in p)),wins=sum(x['ratio']<1 for x in p),median_ratio=float(np.median([x['ratio'] for x in p])))
    matched=[x for x in lab if x['set_id'] in ['exp:d0cc7f139934','exp:0e0d8da270c3']]
    strata=[]
    for rate in [.003,.0003]:
        for ln in [False,True]:
            cell=[x for x in matched if x['optimizer']['lr']==rate and x['layer_norm']==[ln,ln]]
            ratios=[a['mean']/b['mean'] for a in cell for b in cell if a['activation'] in ['gelu','silu'] and b['activation'] in ['relu','leaky_relu']]
            strata.append(dict(lr=rate,layer_norm=ln,wins=sum(r<1 for r in ratios),comparisons=len(ratios),ratios=ratios,median_ratio=float(np.median(ratios))))
    decay=[x for x in lab if x['set_id']=='exp:3dbc82fbb876']
    effects=[]
    for rate in [.0001,.001]:
        cells={x['optimizer']['weight_decay']:x for x in decay if x['optimizer']['lr']==rate}
        effects.append(dict(lr=rate,decay_ratio=cells[.0001]['mean']/cells[.00001]['mean']))
    audit=[dict(question_id=x['question_id'],epoch=x['epoch'],source=x.get('source'),cited=x.get('cited'),score=x.get('score'),comment=x.get('comment')) for x in history if 'K1040' in str(x)]
    return dict(aggregate=aggregate,pairs=pairs,matched_strata=strata,decay_effects=effects,audit=audit)

if __name__=='__main__':
    with open('evidence.json') as f: data=json.load(f)
    result=analyze(data['lab'],data['history'])
    with open('results.json','w') as f: json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['pairs','audit']},indent=2))
