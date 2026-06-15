"""Golias v27 UI — language-first intake; scalars decoded from text."""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Golias Live</title>
<style>
:root{color-scheme:dark}
body{margin:0;font-family:ui-monospace,Menlo,monospace;background:#05070a;color:#d7e0f0}
header{padding:12px 24px;border-bottom:1px solid #1f2937;background:#0a0f1c;display:flex;justify-content:space-between;align-items:center}
h1{font-size:15px;margin:0;color:#8ab4ff}
.tag{font-size:11px;color:#64748b}
.wrap{max-width:1280px;margin:0 auto;padding:18px 24px;display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
h2{font-size:11px;text-transform:uppercase;letter-spacing:1.5px;color:#64748b;margin:0 0 8px}
.term{height:360px;border:1px solid #1f2937;border-radius:8px;background:#0a0f1c;overflow:auto;padding:8px}
#log{white-space:pre-wrap;color:#67e8f9;font-size:11px;line-height:1.4}
label{font-size:9px;color:#64748b;display:block;margin-bottom:2px}
textarea,input{width:100%;background:#0f1625;border:1px solid #243150;color:#e2e8f0;border-radius:5px;padding:8px;font-size:12px;box-sizing:border-box;font-family:inherit}
textarea{min-height:140px;resize:vertical;line-height:1.45}
button{background:linear-gradient(135deg,#3f7bff,#7a47ff);border:0;color:#fff;font-weight:600;padding:10px 16px;border-radius:6px;cursor:pointer;width:100%;margin-top:8px}
button.secondary{background:#1e2937;border:1px solid #334155}
#answer,#out,.outbox{background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;padding:10px;font-size:11px;min-height:70px;margin-top:8px;white-space:pre-wrap;color:#bae6fd}
#decoded{border-color:#334155;background:#0a0f18;color:#94a3b8;font-size:10px;line-height:1.5}
#languageOut{border-color:#3b2f6b}
#alignment{border-color:#854d0e}
#alignment.aligned{border-color:#166534;background:#0a1f14;color:#86efac}
#alignment.misaligned{border-color:#7f1d1d;background:#1f0a0a;color:#fca5a5}
.visRow{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:8px}
.visRow img,.visRow video{width:100%;border-radius:6px;border:1px solid #1f2937;background:#000}
.clip{margin-top:8px}
.clip video,.clip img{max-width:100%;border-radius:6px;border:1px solid #334155}
.hint{font-size:10px;color:#64748b;margin:6px 0 10px;line-height:1.4}
.dropZone{border:2px dashed #dc2626;border-radius:8px;padding:16px;text-align:center;background:#1a0a0a;color:#fca5a5;margin:10px 0;cursor:pointer}
.dropZone.hover{border-color:#f87171;background:#2a1010}
.dropZone p{margin:4px 0;font-size:11px}
.dropZone .hint{font-size:9px;color:#94a3b8}
#dropStatus{font-size:10px;color:#86efac;margin-top:6px;min-height:14px}
.trainRow{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.trainRow label{font-size:9px;display:flex;align-items:center;gap:4px;color:#94a3b8}
</style></head>
<body>
<header>
  <h1>GOLIAS LIVE — v27 τ STATE MACHINE</h1>
  <span class="tag" id="meta">loading...</span>
</header>
<div class="wrap">
  <section>
    <h2>Corpus replay log</h2>
    <div class="term"><div id="log">connecting...</div></div>
    <h2>Dataset drop — continuous retrain intake</h2>
    <div id="dropZone" class="dropZone">
      <p>Drop JSON / JSONL here</p>
      <p class="hint">L2 modular arrays, normalized JSONL, or doctrine rows</p>
      <input type="file" id="fileInput" accept=".json,.jsonl,application/json" style="display:none" />
    </div>
    <div id="dropStatus"></div>
    <div class="trainRow">
      <label><input type="checkbox" id="autoTrain" checked /> Auto-start hybrid train after upload</label>
      <button class="secondary" onclick="document.getElementById('fileInput').click()">Browse file</button>
    </div>
    <button class="secondary" onclick="startTrain()">Start hybrid train (JSONL + HF)</button>
    <button class="secondary" onclick="demoChallenge()">Demo: move the red block left</button>
  </section>
  <section>
    <h2>Language input (geometry → binary → language)</h2>
    <p class="hint">Write in natural language, or use corpus form:
      <code>geometry G=0.52 binary B=0.73 triangulation T=0.41 M1 explore=4.2 … language: move the red block left</code>.
      Scalars are encoded <em>inside</em> the text — not separate sliders.</p>
    <label>State description</label>
    <textarea id="language" placeholder="move the red block to the left&#10;&#10;Or full corpus row:&#10;geometry G=0.5200 anchors live query. binary B=0.7300 encodes spatial_command. triangulation T=0.4100. M1 explore=4.2 M2 effic=0.55 M3 meta=0.99 V judge=0.58 if7=0.5. language: move the red block to the left"></textarea>
    <div id="decoded" class="outbox">Decoded scalars (from language): waiting for run…</div>
    <button onclick="runForward()">Run M1→M2→M3 → of₁ + of₂</button>
    <button class="secondary" onclick="runJudge()">Judge(V) → Adapt(θ,V)</button>
    <div id="alignment" class="outbox">Alignment: waiting...</div>
    <h2>Next frame (of₁) — image + short clip</h2>
    <div class="visRow">
      <div><label>current</label><img id="imgCurrent" alt="current frame" /></div>
      <div><label>predicted next</label><img id="imgNext" alt="next frame" /></div>
    </div>
    <div class="clip"><label>preview clip</label><div id="clipHost">—</div></div>
    <h2>Language (of₂)</h2>
    <div id="languageOut" class="outbox">Language output: waiting...</div>
    <div id="answer">Summary</div>
    <div id="out"></div>
  </section>
</div>
<script>
fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('meta').textContent =
    (d.arch||d.mode||'v27') + ' | ' + (d.device||'?') + ' | if:' + (d.if_backend||'?') +
    ' | ckpt:' + (d.ckpt||'?') + ' | hf@' + (d.hf_stream_offset||'?');
});
const dropZone=document.getElementById('dropZone');
const fileInput=document.getElementById('fileInput');
const dropStatus=document.getElementById('dropStatus');
function uploadFile(file){
  if(!file) return;
  dropStatus.textContent='Uploading '+file.name+'...';
  const reader=new FileReader();
  reader.onload=async ()=>{
    try{
      const auto=document.getElementById('autoTrain').checked;
      const r=await fetch('/upload/dataset',{
        method:'POST',
        headers:{'Content-Type':'application/octet-stream','X-Filename':file.name,'X-Auto-Train':auto?'1':'0'},
        body:reader.result
      });
      const d=await r.json();
      if(d.error){ dropStatus.textContent='Error: '+d.error; return; }
      dropStatus.textContent='Added '+d.records_added+' rows → corpus '+(d.master_corpus_lines||'?')+' lines'+
        (d.train_started?' | training started':'');
      log.textContent+='\\n[drop] '+JSON.stringify(d)+'\\n';
    }catch(e){ dropStatus.textContent='Upload failed: '+e; }
  };
  reader.readAsArrayBuffer(file);
}
dropZone.addEventListener('click',()=>fileInput.click());
fileInput.addEventListener('change',e=>uploadFile(e.target.files[0]));
['dragenter','dragover'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();dropZone.classList.add('hover');}));
['dragleave','drop'].forEach(ev=>dropZone.addEventListener(ev,e=>{e.preventDefault();dropZone.classList.remove('hover');}));
dropZone.addEventListener('drop',e=>uploadFile(e.dataTransfer.files[0]));
const log=document.getElementById('log'); let logPos=0;
async function pollLog(){
  try{
    const r=await fetch('/log?pos='+logPos);
    const d=await r.json();
    if(d.lines&&d.lines.length){
      if(log.textContent.startsWith('connect')) log.textContent='';
      log.textContent+=d.lines.join('\\n')+'\\n';
      logPos=d.pos; log.parentElement.scrollTop=log.parentElement.scrollHeight;
    }
  }catch(e){}
  setTimeout(pollLog,1500);
}
pollLog();
function payload(){
  return { language: document.getElementById('language').value };
}
function showDecoded(rec){
  if(!rec){
    document.getElementById('decoded').textContent='Decoded scalars: (none)';
    return;
  }
  document.getElementById('decoded').textContent=
    'Decoded from language\\n'+
    'G='+rec.geometry+'  B='+rec.binary+'  T='+rec.triangulation+'\\n'+
    'M1='+rec.m1+'  M2='+rec.m2+'  M3='+rec.m3+'  V='+rec.V+'  if7='+rec.if7+'\\n\\n'+
    'Canonical record sent to model:\\n'+(rec.language||'');
}
function demoChallenge(){
  document.getElementById('language').value=
    'geometry G=0.5200 anchors live query. binary B=0.7300 encodes domain=Real_World_Execution, spatial_command. '+
    'triangulation T=0.4100 measures cross-modal coherence. M1 explore=4.2 M2 effic=0.55 M3 meta=0.99 V judge=0.58 if7=0.5. '+
    'language: move the red block to the left';
  runForward();
}
async function runForward(){
  document.getElementById('answer').textContent='Running M1→M2→M3...';
  document.getElementById('alignment').textContent='Alignment: computing...';
  document.getElementById('languageOut').textContent='Language (of₂): computing...';
  document.getElementById('imgCurrent').removeAttribute('src');
  document.getElementById('imgNext').removeAttribute('src');
  document.getElementById('clipHost').textContent='—';
  const r=await fetch('/forward',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
  const d=await r.json();
  if(d.error){
    document.getElementById('answer').textContent=d.error;
    document.getElementById('alignment').textContent='Alignment: error';
    return;
  }
  showDecoded(d.input_record);
  if(d.halt_source==='m2_sidecar'){
    document.getElementById('alignment').textContent='HALT — M2 sidecar (no visual forward)';
    document.getElementById('languageOut').textContent='Language halted by M2';
    document.getElementById('answer').textContent='HALT C_comp='+d.c_comp+' τ='+d.tau;
    return;
  }
  const alignEl=document.getElementById('alignment');
  const aligned=!!d.outputs_aligned;
  alignEl.className='outbox '+(aligned?'aligned':'misaligned');
  alignEl.textContent=(d.alignment_explanation||('aligned='+aligned+' score='+(d.alignment_score||'?')));
  if(d.current_frame_image) document.getElementById('imgCurrent').src=d.current_frame_image;
  if(d.next_frame_image) document.getElementById('imgNext').src=d.next_frame_image;
  const clip=document.getElementById('clipHost');
  clip.innerHTML='';
  if(d.next_frame_video){
    const v=document.createElement('video');
    v.src=d.next_frame_video; v.autoplay=true; v.loop=true; v.muted=true; v.playsInline=true;
    clip.appendChild(v);
  } else if(d.frame_sequence&&d.frame_sequence.length){
    const img=document.createElement('img');
    let i=0; img.src=d.frame_sequence[0];
    setInterval(()=>{ i=(i+1)%d.frame_sequence.length; img.src=d.frame_sequence[i]; }, 120);
    clip.appendChild(img);
  } else {
    clip.textContent='(no clip — install Pillow on GPU for GIF)';
  }
  const langOut=d.of2_language||d.of2_explanation||'';
  const body=(d.input_record&&d.input_record.language_body)||document.getElementById('language').value;
  document.getElementById('languageOut').textContent=
    'Semantic body: '+(body||'(empty)')+'\\n\\n'+
    'Output: '+langOut+'\\n\\n'+
    (d.of2_decode_tokens?'Decode: '+d.of2_decode_tokens+'\\n':'')+
    (d.rl_language_context?'RL: '+d.rl_language_context:'');
  document.getElementById('answer').textContent=
    'τ='+d.tau+' | G='+d.geometry+' B='+d.binary+
    ' | next_frame_scalar='+(d.next_frame_scalar??d.of1_scalar)+
    ' | mismatch='+d.mismatch+
    (d.sidecars?'\\nbackends: '+JSON.stringify(d.sidecars.backends):'');
  document.getElementById('out').textContent=JSON.stringify(d,null,2);
}
async function runJudge(){
  const r=await fetch('/judge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
  const d=await r.json();
  showDecoded(d.input_record);
  document.getElementById('out').textContent=JSON.stringify(d,null,2);
}
async function startTrain(){
  const r=await fetch('/train/jsonl',{method:'POST',body:'{}'});
  const d=await r.json();
  log.textContent+='\\n[train] '+JSON.stringify(d)+'\\n';
}
</script>
</body></html>"""
