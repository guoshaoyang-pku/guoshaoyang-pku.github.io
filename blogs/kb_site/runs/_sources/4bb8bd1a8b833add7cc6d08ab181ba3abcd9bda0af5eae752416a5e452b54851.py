import pathlib,json,numpy as np
rows=[]; contrasts=[]
for p in sorted(pathlib.Path('experiments').glob('*.json')):
 e=json.loads(p.read_text()); result=e['result'];print('RESULT_TYPE',type(result).__name__)
 if isinstance(result,str):result=json.loads(result)
 print('RESULT_KEYS',result.keys())
 for r in result['results']:
  v={'model.num_layers':r['candidate']['model']['num_layers'],'model.d_ff':r['candidate']['model']['d_ff']};mf=r['measurement_files']; z=np.load(mf['results/curves.npz']['repo_path']);a=np.array([s['final_test_ce'] for s in r['seed_results']])
  rows.append({'dataset':result['dataset'],'depth':v['model.num_layers'],'ff':v['model.d_ff'],'mean':r['mean'],'std':r['std'],'seeds':r['seed_results'],'failures':r['failed_seeds'],'cached':r['cached'],'measurements':mf,'step1':z['curves'][:,0].tolist(),'final':a.tolist()})
for d in sorted({r['dataset'] for r in rows}):
 grid={(r['depth'],r['ff']):r for r in rows if r['dataset']==d}
 def contrast(name,a,b):
  dif=np.array(a)-np.array(b);mean=float(dif.mean());half=2.262157*float(dif.std(ddof=1))/np.sqrt(10)
  item={'dataset':d,'contrast':name,'mean':mean,'ci95':[mean-half,mean+half],'negative_seeds':int(sum(dif<0)),'differences':dif.tolist()};contrasts.append(item);print(json.dumps(item))
 for f in [128,256]:contrast('depth3-depth2 FF'+str(f),grid[3,f]['final'],grid[2,f]['final'])
 contrast('depth3FF128-depth2FF256',grid[3,128]['final'],grid[2,256]['final'])
 for depth in [2,3]:contrast('FF256-FF128 depth'+str(depth),grid[depth,256]['final'],grid[depth,128]['final'])
 contrast('interaction',(np.array(grid[3,256]['final'])-grid[2,256]['final']).tolist(),(np.array(grid[3,128]['final'])-grid[2,128]['final']).tolist())
 print('GRID',[(k,round(v['mean'],6),round(v['std'],6),round(np.mean(v['step1']),6)) for k,v in grid.items()])
pathlib.Path('research/results.json').write_text(json.dumps(rows,indent=2));pathlib.Path('research/contrasts.json').write_text(json.dumps(contrasts,indent=2))
