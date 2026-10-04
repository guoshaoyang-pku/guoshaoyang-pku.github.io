import json, math, csv, collections
from pathlib import Path
P=Path(__file__).parent
raw=json.loads((P/'raw_experiments.json').read_text())
lab=json.loads((P/'lab_results_snapshot.json').read_text())
rows=[]
for batch in raw:
 data=batch['data']
 for r in data.get('results',[]):
  v=r['variant'];d=v['model.depth'];ln=v['model.layer_norm'];n=sum(ln)
  pos='shallow' if d==1 else 'dense' if n==d else 'early' if ln[0] else 'last' if ln[-1] else 'middle'
  row=dict(batch=batch['key'],dataset=data['dataset'].split('/')[-1],family=data['dataset'].split('/')[0],depth=d,width=v['model.width'],activation=v['model.activation'],residual=v['model.residual'],ln=ln,position=pos,tail=d-1-max(i for i,z in enumerate(ln) if z),optimizer=v['optimizer.type'],lr=v['optimizer.lr'],wd=v['optimizer.weight_decay'],mean=r['mean'],sd=r['std'],error=r['error'],protocol_seeds=10)
  row['finite']=r['error'] is None and all(z is not None and math.isfinite(z) for z in [r['mean'],r['std']])
  matches=[]
  for x in lab:
   if not x['set_id'].startswith('exp:') or x['dataset_id']!=row['dataset']:continue
   if all(x.get(k)==row[k] for k in ['depth','width','activation','residual']) and x['layer_norm']==ln and x['optimizer']['type']==row['optimizer'] and x['optimizer']['lr']==row['lr'] and x['optimizer'].get('weight_decay',0)==row['wd']:
    if x['mean']==row['mean'] and x['std']==row['sd']:matches.append(x['set_id']+'/'+x['candidate_id'])
  row['lab_ids']=matches;rows.append(row)
(P/'all_attempts.json').write_text(json.dumps(rows,indent=2))
primary=[]
for r in rows:
 key=r['batch']
 if key.startswith('repbatch_fixed') or key.startswith('unifull') or key.startswith('batch') or key in ['core_mvar_00d128','unicore_sym_00bebe','replacement']:
  primary.append(r)
# Each condition has one selected batch; core duplicates on the completed univariate target are retained in all_attempts only.
def cellkey(r):return tuple(r[k] for k in ['dataset','optimizer','lr','activation','residual','depth','position'])
assert len({cellkey(r) for r in primary})==len(primary)
idx={cellkey(r):r for r in primary}
for r in primary:
 b=idx.get((r['dataset'],r['optimizer'],r['lr'],r['activation'],r['residual'],1,'shallow'))
 r['ratio']=r['mean']/b['mean'] if b and b['finite'] and r['finite'] and b['mean']>0 else None
 r['ratio_se_upper']=(r['sd']+r['ratio']*b['sd'])/(math.sqrt(10)*b['mean']) if r['ratio'] is not None and r['depth']>1 else None
(P/'controlled_cells.json').write_text(json.dumps(primary,indent=2))
fields=['dataset','family','optimizer','lr','activation','residual','depth','position','tail','mean','sd','finite','ratio','ratio_se_upper','error','batch','lab_ids']
with (P/'controlled_cells.csv').open('w') as f:
 w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(primary)
print('primary cells',len(primary),'finite',sum(r['finite'] for r in primary),'datasets',len({r['dataset'] for r in primary}))
print('ratio available',sum(r['ratio'] is not None and r['depth']>1 for r in primary))
print('nonfinite',[(r['dataset'],r['optimizer'],r['activation'],r['residual'],r['depth'],r['position'],r['error']) for r in primary if not r['finite']])
print('CORE RATIOS plain SiLU Adagrad: d4early/middle/last/dense;d5early/middle/last/dense')
for ds in sorted({r['dataset'] for r in primary}):
 rs=[idx.get((ds,'Adagrad',0.0003,'silu',False,d,p)) for d in [4,5] for p in ['early','middle','last','dense']]
 print(ds,[round(r['ratio'],5) if r and r['ratio'] is not None else None for r in rs])
print('depth5 full factorial ratios middle,last,dense; each dataset separately')
for ds in ['mvar_000e80','mvar_00ccf0','sym_018114']:
 for opt in ['Adagrad','Adam','SGD']:
  for act in ['silu','relu']:
   for res in [False,True]:
    rs=[idx[(ds,opt,0.0003,act,res,5,pos)] for pos in ['early','middle','last','dense']]
    print(ds,opt,act,res,[round(r['ratio'],5) if r['ratio'] is not None else None for r in rs])

cal=[r for r in rows if r['batch'].startswith('parity_')]
for r in cal:
 b=next(x for x in cal if x['batch']==r['batch'] and x['optimizer']==r['optimizer'] and x['depth']==1)
 r['ratio']=r['mean']/b['mean'] if r['finite'] and b['finite'] else None
 r['ratio_se_upper']=(r['sd']+r['ratio']*b['sd'])/(math.sqrt(10)*b['mean']) if r['ratio'] is not None and r['depth']>1 else None
(P/'calibrated_cells.json').write_text(json.dumps(cal,indent=2))
assert len(primary)==360 and all(r['width']==96 for r in primary)
assert all(len(r['ln'])==r['depth'] for r in primary)
assert len(cal)==36 and all(r['finite'] for r in cal)
assert len([r for r in primary if r['dataset']=='sym_018114'])==108
assert all(r['lab_ids'] for r in primary)
(P/'validation.json').write_text(json.dumps({'primary_cells':360,'finite_primary':sum(r['finite'] for r in primary),'primary_datasets':6,'calibrated_cells':36,'finite_calibrated':36,'missing_dataset':'sym_01a8a6','checks':'width96, LN-array length, balanced factorial counts, calibrated finiteness, measurement source IDs verified','seeds':'10 requested per tool protocol; marginal SD only; valid per-seed counts not exposed'}))
print('validation passed')
