import json, glob, collections, math
lab=[x for f in glob.glob('audit/sets/*.json') for x in json.load(open(f))]
sets=collections.defaultdict(list)
for x in lab:
    if x['family']=='synthetic_tabular_classification' and x['model_type']=='mlp' and not x.get('excluded') and not x.get('failed_seeds'):
        sets[x['set_id']].append(x)
pairs=[]
for sid, xs in sets.items():
    for i,x in enumerate(xs):
        for y in xs[i+1:]:
            if x['depth']==y['depth'] or any(x[k]!=y[k] for k in ['optimizer','budget','loss']):
                continue
            a,b=sorted([x,y],key=lambda z:z['depth'])
            o=a['optimizer']; t=a['budget']['training_steps']
            d=2*o['lr']*math.sqrt(t) if o['type']=='Adagrad' else o['lr']*t
            if o['type']=='SGD':
                d=o['lr']*t*.01/(1-o.get('momentum',0))
            pairs.append({'set':sid,'shallow':a['candidate_id'],'deep':b['candidate_id'],'rule':a['dataset']['rule_family'],'E':a['budget']['total_samples_seen']/a['dataset']['train_size'],'delta':d,'win':a['mean']<b['mean'],'difference':a['mean']-b['mean']})
groups=collections.defaultdict(list)
for p in pairs:
    if p['rule'] in ['smooth_additive','sparse_interaction','piecewise_boundary']:
        groups[(p['rule'],p['E'],p['delta']>=.1)].append(p)
for key, ps in sorted(groups.items()):
    print(key, len(ps), sum(p['win'] for p in ps))
print('all pairs',len(pairs))
