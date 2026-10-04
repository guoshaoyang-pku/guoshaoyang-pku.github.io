exec(open('backtest.py').read().split('P=make_pairs')[0])
L=[x for x in load_lab() if not x.get('excluded') and x.get('model_type')=='mlp' and x['loss']['loss_id'] in ['mse','cross_entropy']]
R=[]
for sid,rs in itertools.groupby(sorted(L,key=lambda x:x['set_id']),lambda x:x['set_id']):
 for a,b in itertools.permutations(list(rs),2):
  R.append(dict(a=a,b=b,da=delta(a),db=delta(b),win=a['mean']<b['mean'],ratio=a['mean']/b['mean']))
def matched(p):
 a,b=p['a'],p['b'];return a['optimizer']['type']==b['optimizer']['type'] and a['optimizer']['lr']==b['optimizer']['lr']
def show(label,ps):print(label,summary(ps))
for fam in ['xor_classification','synthetic_tabular_classification']:
 ps=[p for p in R if p['a']['family']==fam and matched(p) and p['a']['n_ln']>p['b']['n_ln']]
 if fam=='xor_classification':
  ps=[p for p in ps if p['a']['dataset']['input_dim']>=8]
  for lo,hi in [(0,.03),(.03,.05),(.05,.08),(.08,.1),(.1,100)]:show('XOR moreLN '+str((lo,hi)),[p for p in ps if lo<=p['da']<hi])
 else:
  for rule in sorted({p['a']['dataset']['rule_family'] for p in ps}):
   for low in [True,False]:show('TAB '+rule+' low='+str(low),[p for p in ps if p['a']['dataset']['rule_family']==rule and (p['da']<.1)==low])
sg=[p for p in R if p['a']['optimizer']['type']=='SGD' and p['b']['optimizer']['type']!='SGD' and p['da']>=3*p['db'] and p['db']<.1]
for label,ps in [('ln>=1',[p for p in sg if p['a']['n_ln']>=1]),('ln0',[p for p in sg if p['a']['n_ln']==0]),('ln0 deep',[p for p in sg if p['a']['n_ln']==0 and p['a']['depth']>=3])]:
 show('SGD '+label,ps)
 if label=='ln0 deep':
  for res in [False,True]:show('SGD deep residual='+str(res),[p for p in ps if p['a']['residual']==res])
open('scope_pairs.json','w').write(json.dumps([p for p in R if (p['a']['family'] in ['xor_classification','synthetic_tabular_classification'] and matched(p) and p['a']['n_ln']>p['b']['n_ln']) or p in sg],indent=2))
