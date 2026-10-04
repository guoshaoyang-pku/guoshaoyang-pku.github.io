import json, pathlib, hashlib, ast, re
import numpy as np
P=pathlib.Path(__file__).resolve().parent
labels=['D','B','N','R','U']
question=json.loads((P/'q_48c5b5.json').read_text())['question']
original={}
for label in ['D','B']:
    sec=re.search(r'### Choice '+label+r'\n(.*?)(?=### Choice|## Your answer)',question,re.S).group(1)
    src=re.findall(r'```python\n(.*?)```',sec,re.S)[0]
    original[label]=[ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,ast.ClassDef)]

output={}
verified=set()
for ds in ['016ff9','01793b','05d021','06bcc2']:
    receipt=json.loads((P/(ds+'.json')).read_text())
    rows=receipt['results']; assert len(rows)==5
    cells={}
    for label,row in zip(labels,rows):
        assert row['error'] is None and row['failed_seeds']==0 and not row['excluded']
        seeds=row['seed_results']; assert [s['seed'] for s in seeds]==list(range(10))
        vals=np.array([s['final_test_ce'] for s in seeds])
        assert np.isclose(vals.mean(),row['mean']) and np.isclose(vals.std(),row['std'])
        for key,meta in row['measurement_files'].items():
            f=P.parent/meta['repo_path']; data=f.read_bytes()
            assert hashlib.sha256(data).hexdigest()==meta['sha256']
            verified.add(meta['repo_path'])
            if key=='executed/model.py':
                tree=ast.parse(data.decode()); assert 'nn.GELU' in data.decode()
                if label in original:
                    assert original[label]==[ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.ClassDef)]
            if key=='results/curves.npz':
                z=np.load(f)
                assert z['curves'].shape==(10,512)
                assert np.allclose(z['curves'][:,-1],vals)
                assert z['samples'][-1]==512*64 and int(z['batch_size'])==64
                cells.setdefault(label,{})['curve_keys']=list(z.keys())
        cells[label].update(mean=row['mean'],std=row['std'],cached=row['cached'],seeds=vals.tolist(),provenance=row['source_provenance'])
    contrasts={}
    for a,b in [('B','D'),('N','B'),('R','B'),('D','U'),('N','D'),('R','D')]:
        v=np.array(cells[a]['seeds'])-np.array(cells[b]['seeds'])
        half=2.2621571628540993*v.std(ddof=1)/np.sqrt(10)
        contrasts[a+'-'+b]={'difference':float(v.mean()),'ci95':[float(v.mean()-half),float(v.mean()+half)],'first_wins':int((v<0).sum()),'ratio':cells[a]['mean']/cells[b]['mean'],'first_losing_seeds':[int(i) for i in np.where(v>0)[0]]}
    output[ds]={'turns':receipt['dataset_params']['spiral_turns'],'cells':cells,'contrasts':contrasts}
(P/'statistics.json').write_text(json.dumps(output,indent=2))
print('Verified unique artifacts:',len(verified))
for ds,x in output.items():
    print(ds,[(k,round(v['mean'],9)) for k,v in x['cells'].items()])
    for k,v in x['contrasts'].items(): print(k,v)
