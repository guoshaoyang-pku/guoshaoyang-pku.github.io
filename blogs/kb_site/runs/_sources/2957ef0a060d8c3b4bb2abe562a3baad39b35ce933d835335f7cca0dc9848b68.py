import json,re,itertools
from pathlib import Path
root=Path(__file__).resolve().parent
history=[r for p in sorted(root.glob("history_[0-9]*.json")) for r in json.loads(p.read_text())]
strict=[];bundles=[]
for r in history:
 if r["family"]!="bigram_lm":continue
 chunks=re.split(r"### Choice ([A-E])",r["question"]);choices={}
 for i in range(1,len(chunks),2):
  s=chunks[i+1];shape={}
  for key in ["d_model","num_layers","num_heads","d_ff"]:
   m=re.search(r"- "+key+r": (\d+)",s)
   if m:shape[key]=int(m[1])
  op=re.search(r"- Optimizer: (\w+)",s);lr=re.search(r"- Learning rate: ([\deE.+-]+)",s);wd=re.search(r"- Weight decay: ([\deE.+-]+)",s)
  if len(shape)==4 and op and lr and wd:choices[chunks[i]]=(shape,op[1],float(lr[1]),float(wd[1]))
 order=re.findall("[A-E]",r.get("answer",""))
 for a,b in itertools.combinations(choices,2):
  sa,oa,la,wa=choices[a];sb,ob,lb,wb=choices[b]
  if (oa,la,wa)!=(ob,lb,wb) or oa!="RMSprop" or la!=.003:continue
  diff=[key for key in sa if sa[key]!=sb[key]]
  for key,target in [(diff[0],strict)] if len(diff)==1 else []:
   small,big=(a,b) if sa[key]<sb[key] else (b,a)
   if small in order and big in order:target.append({"qid":r["question_id"],"dimension":key,"small":small,"big":big,"small_wins":order.index(small)<order.index(big),"shapes":[sa,sb]})
  if sa["d_model"]!=sb["d_model"]:
   small,big=(a,b) if sa["d_model"]<sb["d_model"] else (b,a)
   if small in order and big in order:bundles.append({"qid":r["question_id"],"small":small,"big":big,"small_wins":order.index(small)<order.index(big),"shapes":[sa,sb]})
for name,rows in [("history_pairs",strict),("history_width_bundles",bundles)]:
 assert rows==json.loads((root/(name+".json")).read_text())
 print(name,len(rows),"verified")
