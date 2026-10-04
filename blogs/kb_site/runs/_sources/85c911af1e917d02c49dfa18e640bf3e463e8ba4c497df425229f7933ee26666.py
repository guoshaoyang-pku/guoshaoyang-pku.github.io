import re,json,itertools,math,pathlib
h=json.loads(pathlib.Path('research/history_frozen.json').read_text()); out=[]; parsed=[]
def number(pattern,s):
 m=re.search(pattern,s);return float(m.group(1)) if m else None
for x in h:
 if x.get('epoch',0)>=9 or 'Bigram' not in x.get('question',''):continue
 q=x['question']; gold=x['answer'].strip();rank=gold.split('<')
 if len(rank)<2:continue
 pre=q.split('### Choice ')[0]; globalT=number(r'training_steps:\s*(\d+)',pre)
 cs=[]
 for b in re.split(r'### Choice ',q)[1:]:
  if 'torch.optim.Adagrad' not in b or 'nn.TransformerEncoderLayer' not in b:continue
  T=number(r'training_steps:\s*(\d+)',b) or globalT
  if not T:continue
  c={'letter':b[0],'width':number(r'd_model\s*[=:]\s*(\d+)',b),'depth':number(r'num_layers\s*[=:]\s*(\d+)',b),'ff':number(r'(?:d_ff\s*:|dim_feedforward\s*=)\s*(\d+)',b),'heads':number(r'(?:num_heads\s*:|nhead\s*=)\s*(\d+)',b),'lr':number(r'Learning rate:\s*([\de.\-]+)',b),'wd':number(r'Weight decay:\s*([\de.\-]+)',b),'T':T}
  c['loss']='l1' if 'cross_entropy_l1' in b or 'lambda_l1' in b else 'ce'
  cs.append(c)
 parsed.append({'id':x['question_id'],'candidates':cs,'gold':gold})
 for a,b in itertools.combinations(cs,2):
  if any(a[k]!=b[k] for k in ['width','lr','wd','T','loss']) or a['depth']==b['depth'] or a['width'] is None:continue
  if a['letter'] not in rank or b['letter'] not in rank:continue
  deeper,shallower=(a,b) if a['depth']>b['depth'] else (b,a)
  out.append({'id':x['question_id'],'deep':deeper,'shallow':shallower,'delta':2*a['lr']*math.sqrt(a['T']),'win':rank.index(deeper['letter'])<rank.index(shallower['letter']),'matched':a['ff']==b['ff'] and a['heads']==b['heads']})
pathlib.Path('research/history_pairs.json').write_text(json.dumps(out,indent=2))
pathlib.Path('research/history_parsed.json').write_text(json.dumps(parsed,indent=2))
for low in [True,False]:
 rows=[r for r in out if (r['delta']<.02)==low]
 print('BAND',low,'wins',sum(r['win'] for r in rows),'n',len(rows),'questions',len({r['id'] for r in rows}),'matched',sum(r['matched'] for r in rows))
 for r in rows: print(r['id'],r['deep']['letter'],r['shallow']['letter'],r['win'],r['matched'],r['delta'],r['deep']['width'],r['deep']['T'],r['deep']['lr'])
