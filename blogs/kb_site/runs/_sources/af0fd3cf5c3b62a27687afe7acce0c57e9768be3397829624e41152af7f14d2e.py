import json,math,collections
lab=json.load(open('research/xor_lab.json')); audit=json.load(open('research/audit.json'))
lookup={(r['set_id'],r['candidate_id']):r for r in lab}
def finite(o,T):
 if o['type']=='SGD':
  m=o.get('momentum',0)
  return .01*o['lr']*(T/(1-m)-m*(1-m**T)/(1-m)**2)
 if o['type']=='Adagrad':return o['lr']*sum(t**-.5 for t in range(1,T+1))
 return o['lr']*T
print('finite sums T256',sum(t**-.5 for t in range(1,257)),.0003*.01*(256/.1-.9/.1**2))
for source in ['history','lab']:
 changes=[]
 for r in audit[source]:
  if source=='history':o=r['plain']['optimizer'];T=r['plain']['T'];old=r['plain']['delta']
  else:
   p=lookup[(r['set_id'],r['plain'])];o=p['optimizer'];T=p['budget']['training_steps'];old=r['delta']
  if (old<.05)!=(finite(o,T)<.05):changes.append(r)
 print('finite proxy classification changes',source,len(changes))
out=[]
for r in audit['lab']:
 p=lookup[(r['set_id'],r['plain'])];s=lookup[(r['set_id'],r['rival'])]
 r['loss_match']=p['loss']==s['loss'];r['budget_match']=p['budget']==s['budget'];r['ln_mask_match']=p['layer_norm']==s['layer_norm'];out.append(r)
for relation in ['ge','lt']:
 for high in [False,True]:
  sub=[r for r in out if (r['delta']>=.05)==high and r['ln_relation']==relation and r['full_optimizer_match'] and r['loss_match'] and r['budget_match']]
  exact=[r for r in sub if all(r[k] for k in ['width_match','activation_match','depth_match','last_match'])]
  twins=[r for r in exact if r['ln_mask_match']]
  print('AUDIT',relation,high,'full',sum(r['win'] for r in sub),len(sub),'sets',len(set(r['set_id'] for r in sub)),'width/act/depth/final',sum(r['win'] for r in exact),len(exact),'full LN masks',sum(r['win'] for r in twins),len(twins))
  if sub:
   print('set clusters')
   for sid in sorted(set(r['set_id'] for r in sub)):
    ss=[r for r in sub if r['set_id']==sid];print(sid,len(ss),sum(r['win'] for r in ss),sum(r['diff'] for r in ss)/len(ss),sum(r['ratio'] for r in ss)/len(ss))
with open('research/audit_complete.json','w') as f:json.dump(dict(history=audit['history'],lab=out,errors=audit['errors']),f,indent=2)
