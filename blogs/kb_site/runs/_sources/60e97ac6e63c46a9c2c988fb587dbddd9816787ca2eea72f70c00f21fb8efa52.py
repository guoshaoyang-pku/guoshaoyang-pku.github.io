import json, pathlib
root=pathlib.Path(__file__).parent
history={x['question_id']:x for x in json.loads((root/'history.json').read_text())}
pairs=json.loads((root/'history_pairs.json').read_text())
eligible=[x for x in pairs if history[x['q']]['epoch']<4 and x['plain']['opt']!='SGD' and x['plain']['nln']>0]
assert len(eligible)==9 and sum(x['win'] for x in eligible)==7
high=[x for x in eligible if x['delta']>=.5]
assert len(high)==2 and all(x['win'] for x in high)
strict=[x for x in eligible if x['exact']]
assert len(strict)==8 and sum(x['win'] for x in strict)==6
rows=json.loads((root/'experiments.json').read_text())
assert len(rows)==18
names=['D','A','B','E','C','L_TFF','L_FFT','R3_FTF','P2_TT']
for ds in sorted({x['dataset_id'] for x in rows}):
    conditions=[x for x in rows if x['dataset_id']==ds]
    by=dict(zip(names,conditions))
    assert all(x['budget']['training_steps']==2048 and x['budget']['batch_size']==16 for x in conditions)
    assert all(x['optimizer']=={'type':'AdamW','lr':.0003,'weight_decay':1e-5,'betas':[.9,.999]} for x in conditions)
    assert sorted(names[:5],key=lambda n:by[n]['mean'])==names[:5]
    print(ds, 'B-D',by['B']['mean']-by['D']['mean'],'D/B',by['D']['mean']/by['B']['mean'])
print('History 7/9 weak match; 6/8 exact positive-LN adaptive. Failure metadata unavailable.')
