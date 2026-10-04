import json, itertools, collections, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text())
def summary(rows):
    return dict(wins=sum(r['ratio'] < 1 for r in rows), pairs=len(rows),
                sets=len({r['a']['set_id'] for r in rows}),
                median_ratio=float(np.median([r['ratio'] for r in rows])) if rows else None)
lab = read('lab.json')
groups = collections.defaultdict(list)
for x in lab:
    if x['family'] == 'spiral_classification' and not x.get('excluded') and x.get('source') != 'experiment':
        groups[x['set_id']].append(x)
pairs, architecture = [], []
for sid, xs in groups.items():
    for a, b in itertools.combinations(xs, 2):
        if a['budget'] != b['budget'] or a['loss'] != b['loss']:
            continue
        if a['optimizer']['lr'] == b['optimizer']['lr'] and {a['optimizer']['type'], b['optimizer']['type']} in ({'Adam','RMSprop'}, {'AdamW','RMSprop'}):
            if a['optimizer']['type'] == 'RMSprop':
                a, b = b, a
            pairs.append(dict(a=a,b=b,ratio=a['mean']/b['mean'],net_equal=a['model']==b['model']))
        for deep, plain in [(a,b),(b,a)]:
            if deep['residual'] and deep['depth']>=3 and not plain['residual'] and plain['depth']<deep['depth'] and deep['n_ln']==plain['n_ln'] and deep['optimizer']['type']==plain['optimizer']['type'] and deep['optimizer']['lr']==plain['optimizer']['lr']:
                architecture.append((deep,plain))
easyhigh = [p for p in pairs if p['a']['dataset']['spiral_turns'] <= 1.5 and p['a']['optimizer']['lr'] >= .001 and p['a']['budget']['training_steps']==256]
assert summary(easyhigh)['wins']==13 and len(easyhigh)==15 and summary(easyhigh)['sets']==9
assert not [p for p in pairs if p['net_equal']]
assert not architecture
print('Base shared-rate all:',summary(pairs))
print('Comment reconstruction:',summary(easyhigh))
for turns in [(1,1.5),(2,2.5,3)]:
    for high in (False,True):
        rows=[p for p in pairs if p['a']['dataset']['spiral_turns'] in turns and (p['a']['optimizer']['lr']>=.001)==high]
        print(turns,high,summary(rows))
history=read('history.json')
for qa,qb in [('q_7e1811','q_95fe03'),('q_611798','q_bca005')]:
    a=next(x for x in history if x['question_id']==qa)
    b=next(x for x in history if x['question_id']==qb)
    assert a['question'].split('## Choices')[1]==b['question'].split('## Choices')[1]
    print('Identical full choices:',qa,qb)
experiments=collections.defaultdict(list)
for row in read('experiment_rows.json'):
    experiments[row['set_id']].append(row)
assert len(experiments)==8 and sum(map(len,experiments.values()))==60
for sid,rows in experiments.items():
    print(sid,rows[0]['dataset_id'])
    if len(rows)==10:
        for lr in (.0003,.003):
            for T in (256,1024):
                match=[x for x in rows if x['optimizer']['lr']==lr and x['budget']['training_steps']==T and x['optimizer']['weight_decay']==.0001 and (x['optimizer']['type']=='RMSprop' or x['optimizer'].get('betas')==[.9,.95])]
                a=next(x for x in match if x['optimizer']['type']=='Adam')
                b=next(x for x in match if x['optimizer']['type']=='RMSprop')
                print(lr,T,a['mean'],a['std'],b['mean'],b['std'],a['mean']/b['mean'])
    else:
        by={(x['depth'],x['residual'],x['width']):x for x in rows}
        deep=by[(5,True,192)]
        for label,key in [('original',(3,False,256)),('matched',(3,False,192)),('depth5_plain',(5,False,192))]:
            print(label,deep['mean']/by[key]['mean'])
        print('depth3_skip',by[(3,True,192)]['mean']/by[(3,False,192)]['mean'])
