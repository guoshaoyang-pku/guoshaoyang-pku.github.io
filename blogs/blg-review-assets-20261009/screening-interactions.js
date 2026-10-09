(() => {
  const data = SCREENING;
  const el = id => document.getElementById(id);
  const fmt = (v, decimals = 3) => Number(v).toFixed(decimals);
  const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const colors = {orange:'#d97757', blue:'#6a9bcc', green:'#788c5d', dark:'#141413', grey:'#b0aea5'};
  const metrics = {
    internal_U_meV: {label:'内部层势差', unit:'meV', value:o=>o.internal_U_meV},
    sampled_indirect_gap_meV: {label:'采样谱隙', unit:'meV', value:o=>o.sampled_indirect_gap_meV},
    conductivity: {label:'均匀材料电导', unit:'e²/h', value:o=>o.boltzmann_sigma_tensor_e2_over_h[0][0]},
    compressibility: {label:'固定层势压缩率', unit:'10¹⁰ cm⁻²/meV', value:o=>o.fixed_U_compressibility_nm2_per_meV*1e4},
    polarization: {label:'两层过量电荷之差', unit:'10¹² cm⁻²', value:o=>o.layer_polarization_nm2*100}
  };
  const samples = new Map(data.samples.map(s=>[s.sample_id,s.parameters]));
  let selectedN = 0, selectedD = .3;
  function svg(content, height, label, width=660) {
    return '<svg viewBox="0 0 '+width+' '+height+'" role="img" aria-label="'+esc(label)+'" xmlns="http://www.w3.org/2000/svg">'+content+'</svg>';
  }
  function text(x,y,value,extra='') {
    return '<text x="'+x+'" y="'+y+'" '+extra+'>'+esc(value)+'</text>';
  }
  function metricValue(point, name) { return metrics[name].value(point.observables); }
  function renderMap() {
    const epsilon=Number(el('screen-epsilon').value), top=Number(el('screen-gate').value), metric=el('screen-metric').value;
    const sample=data.samples.find(s=>s.parameters.epsilon_r===epsilon && s.parameters.gate_top_nm===top);
    const points=data.points.filter(p=>p.sample_id===sample.sample_id);
    const range=data.points.map(p=>metricValue(p,metric));
    const minimum=Math.min(...range), maximum=Math.max(...range), log=metric==='conductivity';
    const norm=v=>log ? (Math.log10(v)-Math.log10(minimum))/(Math.log10(maximum)-Math.log10(minimum)) : (v-minimum)/(maximum-minimum);
    let body='<g font-family="PingFang SC,system-ui,sans-serif" font-size="14" fill="'+colors.dark+'">';
    body+=text(330,23,metrics[metric].label+'（'+metrics[metric].unit+'）','text-anchor="middle"');
    for(let ni=0;ni<3;ni++)for(let di=0;di<3;di++){
      const n=data.axes.n_1e12_cm2[ni],d=data.axes.D_V_nm[di],p=points.find(q=>q.controls.n_1e12_cm2===n && q.controls.D_V_nm===d);
      const value=metricValue(p,metric),shade=norm(value),x=118+ni*155,y=46+(2-di)*69,active=n===selectedN && d===selectedD;
      body+='<g class="screen-point" tabindex="0" role="button" data-n="'+n+'" data-d="'+d+'" aria-label="密度 '+n+'，位移场 '+d+'，'+metrics[metric].label+' '+fmt(value)+'"><title>实际计算点；n='+n+'，D='+d+'</title><rect x="'+x+'" y="'+y+'" width="149" height="63" rx="2" fill="'+colors.orange+'" fill-opacity="'+(.08+.68*shade)+'" stroke="'+(active?colors.dark:'#e8e6dc')+'" stroke-width="'+(active?2.5:1)+'"/>'+text(x+74,y+37,fmt(value,metric==='conductivity'?2:3),'text-anchor="middle"')+'</g>';
    }
    for(let i=0;i<3;i++)body+=text(192+i*155,275,data.axes.n_1e12_cm2[i],'text-anchor="middle"');
    for(let i=0;i<3;i++)body+=text(105,82+(2-i)*69,data.axes.D_V_nm[i],'text-anchor="end"');
    body+=text(348,302,'载流子密度 n（10¹² cm⁻²）','text-anchor="middle"')+text(36,146,'位移场 D','text-anchor="middle" transform="rotate(-90 36 146)"');
    body+=text(330,328,'每格为一个实际计算点；点击看共同观测。'+(log?'共用对数颜色范围。':'所有样品共用颜色范围。'),'text-anchor="middle" font-size="14"')+'</g>';
    el('screen-map').innerHTML=svg(body,348,'20组样品的实际微观响应图');
    el('screen-map').querySelectorAll('.screen-point').forEach(point=>{
      const select=()=>{selectedN=Number(point.dataset.n);selectedD=Number(point.dataset.d);renderMap();};
      point.addEventListener('click',select);
      point.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();select();}});
    });
    const p=points.find(p=>p.controls.n_1e12_cm2===selectedN && p.controls.D_V_nm===selectedD),o=p.observables;
    el('screen-readout').textContent='n='+fmt(selectedN,1)+' ×10¹² cm⁻²，D='+fmt(selectedD,1)+' V/nm：外部层势 '+fmt(o.external_U_meV)+' → 内部层势 '+fmt(o.internal_U_meV)+' meV；采样谱隙 '+fmt(o.sampled_indirect_gap_meV)+' meV；电导 '+fmt(o.boltzmann_sigma_tensor_e2_over_h[0][0])+' e²/h。';
    el('screen-readout').dataset.sample=sample.sample_id;
    el('screen-readout').dataset.n=selectedN;el('screen-readout').dataset.d=selectedD;
    el('screen-readout').dataset.gap=o.sampled_indirect_gap_meV;
    el('screen-readout').dataset.sigma=o.boltzmann_sigma_tensor_e2_over_h[0][0];
  }
  function scatterPlot(target, points, getX, getY, xLabel, yLabel, options={}) {
    const width=660,height=330,left=78,right=632,top=36,bottom=260;
    const xs=points.map(getX),ys=points.map(getY),minX=options.minX??Math.min(0,...xs),maxX=options.maxX??Math.max(...xs)*1.06;
    const minY=options.minY??Math.min(0,...ys),maxY=options.maxY??Math.max(...ys)*1.10;
    const xp=x=>left+(x-minX)/(maxX-minX)*(right-left),yp=y=>bottom-(y-minY)/(maxY-minY)*(bottom-top);
    let body='<g font-family="PingFang SC,system-ui,sans-serif" font-size="14" fill="'+colors.dark+'">';
    for(let i=0;i<5;i++){
      const x=minX+(maxX-minX)*i/4,y=minY+(maxY-minY)*i/4;
      body+='<path d="M'+left+' '+yp(y)+'H'+right+'" stroke="#e8e6dc"/>'+text(left-9,yp(y)+4,fmt(y,2),'text-anchor="end"')+text(xp(x),bottom+23,fmt(x,1),'text-anchor="middle"');
    }
    body+='<path d="M'+left+' '+top+'V'+bottom+'H'+right+'" fill="none" stroke="'+colors.grey+'"/>';
    if(options.diagonal)body+='<path d="M'+xp(0)+' '+yp(0)+'L'+xp(maxX)+' '+yp(maxX)+'" stroke="'+colors.grey+'" stroke-dasharray="5 5"/>';
    for(const p of points){
      const x=xp(getX(p)),y=yp(getY(p)),color=p.color||colors.orange;
      const title=p.title||('U='+fmt(getX(p))+'，y='+fmt(getY(p)));
      body+=p.square?'<rect x="'+(x-4)+'" y="'+(y-4)+'" width="8" height="8" fill="'+color+'"><title>'+esc(title)+'</title></rect>':'<circle cx="'+x+'" cy="'+y+'" r="4" fill="'+color+'" fill-opacity=".72"><title>'+esc(title)+'</title></circle>';
    }
    body+=text((left+right)/2,height-21,xLabel,'text-anchor="middle"')+text(18,150,yLabel,'text-anchor="middle" transform="rotate(-90 18 150)"');
    if(options.annotation)body+=text(left+10,23,options.annotation,'fill="#5e5d59"');
    body+='</g>';
    el(target).innerHTML=svg(body,height,yLabel+'与'+xLabel+'的实际计算比较');
  }
  function renderCommon() {
    const points=data.points.filter(p=>p.controls.n_1e12_cm2===0).map(p=>({
      x:p.observables.internal_U_meV,y:p.observables.sampled_indirect_gap_meV,
      color:p.controls.D_V_nm===0?colors.grey:colors.orange,
      title:'介电常数 '+samples.get(p.sample_id).epsilon_r+'，上栅距 '+samples.get(p.sample_id).gate_top_nm+' nm；D='+p.controls.D_V_nm+'，U='+fmt(p.observables.internal_U_meV)+'，谱隙='+fmt(p.observables.sampled_indirect_gap_meV)
    }));
    scatterPlot('screen-common',points,p=>p.x,p=>p.y,'内部层势差 U（meV）','采样谱隙（meV）',{
      minX:0,maxX:22,minY:-1,maxY:22,diagonal:true,annotation:'60个中性点；灰虚线表示谱隙等于层势差。'});
  }
  function renderPrediction() {
    const selection=el('screen-prediction-wave').value;
    const selected=data.predictions.filter(p=>selection==='all'||p.wave===selection);
    const points=[];
    for(const p of selected)for(const name of ['linear','log'])points.push({
      x:p.observables.internal_U_meV,y:p.predicted_internal_U_meV[name],color:name==='linear'?colors.blue:colors.orange,
      square:p.wave==='stress',title:(name==='linear'?'固定系数':'随层势变化')+'；ε='+p.sample.epsilon_r+'，n='+p.controls.n_1e12_cm2+'，D='+p.controls.D_V_nm+'；预测误差 '+fmt(p.prediction_errors_meV[name])+' meV'
    });
    scatterPlot('screen-prediction',points,p=>p.x,p=>p.y,'新计算的内部层势差（meV）','冻结关系预测的层势差（meV）',{
      minX:0,maxX:16,minY:0,maxY:16,diagonal:true,annotation:'圆点：首轮新条件；方点：随后增加的高密度压力测试。'});
    el('screen-prediction-readout').textContent=selection==='new'
      ?'8个新条件：两条关系的最大误差分别为 0.203 和 0.289 meV，均小于冻结的 0.5 meV 范围。这批点无法排除其中任何一条。'
      :selection==='stress'
      ?'4个高密度条件：最大误差升到 1.316 和 1.397 meV。数值对照仍通过，两条简化关系都超出原预测范围。'
      :'首轮能预测，不代表任意密度都能预测。后续压力测试保留同一系数与误差范围，没有重新拟合失败点。';
    renderResiduals(selected);
  }
  function renderResiduals(selected) {
    const width=660,height=325,left=78,right=638,top=32,bottom=247;
    const xp=n=>left+(n+.65)/1.3*(right-left),yp=e=>bottom-e/1.6*(bottom-top);
    let body='<g font-family="PingFang SC,system-ui,sans-serif" font-size="14" fill="'+colors.dark+'">';
    for(let e=0;e<=1.5;e+=.5)body+='<path d="M'+left+' '+yp(e)+'H'+right+'" stroke="#e8e6dc"/>'+text(left-10,yp(e)+4,fmt(e,1),'text-anchor="end"');
    body+='<path d="M'+left+' '+yp(data.prediction_tolerance_meV)+'H'+right+'" stroke="#5e5d59" stroke-dasharray="5 4"/>'+text(right-5,yp(.5)-8,'冻结范围 0.5','text-anchor="end"');
    for(const n of [-.6,-.3,0,.3,.6])body+=text(xp(n),bottom+25,fmt(n,1),'text-anchor="middle"');
    for(const p of selected)for(const name of ['linear','log']){
      const x=xp(p.controls.n_1e12_cm2),y=yp(p.prediction_errors_meV[name]),color=name==='linear'?'#426787':'#98523c';
      const title=esc('ε='+p.sample.epsilon_r+'，n='+p.controls.n_1e12_cm2+'，D='+p.controls.D_V_nm+'；最大绝对误差点值 '+fmt(p.prediction_errors_meV[name])+' meV');
      body+=p.wave==='stress'?'<rect x="'+(x-4)+'" y="'+(y-4)+'" width="8" height="8" fill="'+color+'"><title>'+title+'</title></rect>':'<circle cx="'+x+'" cy="'+y+'" r="4" fill="'+color+'"><title>'+title+'</title></circle>';
    }
    body+='<path d="M'+left+' '+top+'V'+bottom+'H'+right+'" fill="none" stroke="'+colors.grey+'"/>'+text((left+right)/2,305,'载流子密度 n（10¹² cm⁻²）','text-anchor="middle"')+text(20,138,'层势绝对误差（meV）','text-anchor="middle" transform="rotate(-90 20 138)"')+'</g>';
    el('screen-residuals').innerHTML=svg(body,height,'冻结预测的误差随密度变化');
  }
  function renderOccupancy(selection='neutral') {
    const o=data.highlight[selection],gap=o.sampled_indirect_gap_meV,sigma=o.boltzmann_sigma_tensor_e2_over_h[0][0];
    el('screen-case-readout').textContent=(selection==='neutral'?'中性点 n=0':'加入空穴 n=−0.1 ×10¹² cm⁻²')+'：谱隙 '+fmt(gap)+' meV，电导 '+fmt(sigma)+' e²/h。两点电导之比 '+fmt(data.highlight.conductivity_ratio,1)+'。';
    Object.assign(el('screen-case-readout').dataset,{case:selection,gap,sigma});
    document.querySelectorAll('#screen-case-controls button').forEach(button=>button.setAttribute('aria-pressed',button.dataset.case===selection));
    let body='<g font-family="PingFang SC,system-ui,sans-serif" font-size="15" fill="'+colors.dark+'">';
    body+=text(38,28,'采样谱隙（meV）')+text(352,28,'均匀材料电导（e²/h）');
    for(const [i,name] of ['neutral','doped'].entries()){
      const obs=data.highlight[name],g=obs.sampled_indirect_gap_meV,s=obs.boltzmann_sigma_tensor_e2_over_h[0][0],y=66+76*i,active=selection===name;
      body+=text(38,y-10,name==='neutral'?'中性点':'加入空穴','font-weight="'+(active?'600':'400')+'"');
      body+='<rect x="38" y="'+y+'" width="'+(g/12*230)+'" height="20" fill="'+colors.orange+'" fill-opacity="'+(active?1:.45)+'"/>'+text(44+g/12*230,y+16,fmt(g));
      body+='<rect x="352" y="'+y+'" width="'+(s/7*220)+'" height="20" fill="'+colors.blue+'" fill-opacity="'+(active?1:.45)+'"/>'+text(358+s/7*220,y+16,fmt(s));
    }
    body+=text(330,204,'同一样品、同一外场、同一温度；只改变密度。','text-anchor="middle" font-size="14"')+'</g>';
    el('screen-occupancy').innerHTML=svg(body,222,'相近带隙的两个计算点与导电差异');
  }
  el('screen-epsilon').innerHTML=[3,4,6,8,12].map(x=>'<option value="'+x+'">'+x+'</option>').join('');
  el('screen-gate').innerHTML=[10,20,40,60].map(x=>'<option value="'+x+'">'+x+' nm</option>').join('');
  el('screen-metric').innerHTML=Object.entries(metrics).map(([id,m])=>'<option value="'+id+'">'+m.label+'</option>').join('');
  el('screen-epsilon').value='6';el('screen-gate').value='20';el('screen-metric').value='sampled_indirect_gap_meV';
  for(const id of ['screen-epsilon','screen-gate','screen-metric'])el(id).addEventListener('change',renderMap);
  el('screen-prediction-wave').addEventListener('change',renderPrediction);
  document.querySelectorAll('#screen-case-controls button').forEach(button=>button.addEventListener('click',()=>renderOccupancy(button.dataset.case)));
  renderMap();renderCommon();renderPrediction();renderOccupancy();
  window.screeningReady=true;
})();
