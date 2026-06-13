"""Golias v27 UI — geometry → binary → language; of₁/of₂ outputs."""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Golias v27</title>
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
input,textarea{width:100%;background:#0f1625;border:1px solid #243150;color:#e2e8f0;border-radius:5px;padding:5px;font-size:12px;box-sizing:border-box}
textarea{min-height:88px;resize:vertical;font-family:inherit}
button{background:linear-gradient(135deg,#3f7bff,#7a47ff);border:0;color:#fff;font-weight:600;padding:10px 16px;border-radius:6px;cursor:pointer;width:100%;margin-top:8px}
button.secondary{background:#1e2937;border:1px solid #334155}
#answer,#out{background:#0a1628;border:1px solid #1e3a5f;border-radius:6px;padding:10px;font-size:11px;min-height:90px;margin-top:8px;white-space:pre-wrap;color:#bae6fd}
.order{display:grid;grid-template-columns:1fr;gap:8px}
</style></head>
<body>
<header>
  <h1>GOLIAS v27 — τ STATE MACHINE</h1>
  <span class="tag" id="meta">loading...</span>
</header>
<div class="wrap">
  <section>
    <h2>Corpus replay log</h2>
    <div class="term"><div id="log">connecting...</div></div>
    <button class="secondary" onclick="startTrain()">Start JSONL corpus train (v11→v27)</button>
  </section>
  <section>
    <h2>IF inputs (order: geometry → binary → language)</h2>
    <div class="order">
      <div><label>1. geometry (G)</label><input id="g" type="number" step="0.01" value="0.47"></div>
      <div><label>2. binary (B)</label><input id="b" type="number" step="0.01" value="0.73"></div>
      <div><label>3. language (text)</label><textarea id="language" placeholder="Natural language context drives of₂ and RL on mismatch..."></textarea></div>
      <div><label>triangulation</label><input id="tri" type="number" step="0.01" placeholder="auto"></div>
      <div><label>V (human judge 0-1)</label><input id="v" type="number" step="0.01" value="0.58"></div>
    </div>
    <button onclick="runForward()">Run of₁ + of₂ forward</button>
    <button class="secondary" onclick="runJudge()">Judge(V) → Adapt(θ,V)</button>
    <div id="answer">of₁ next_frame (224-d) + of₂ explanation appear here.</div>
    <div id="out"></div>
  </section>
</div>
<script>
fetch('/info').then(r=>r.json()).then(d=>{
  document.getElementById('meta').textContent =
    (d.arch||'v27') + ' | ' + (d.device||'?') + ' | ckpt:' + (d.ckpt||'?');
});
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
  const tri=document.getElementById('tri').value;
  return {
    geometry:parseFloat(document.getElementById('g').value),
    binary:parseFloat(document.getElementById('b').value),
    language:document.getElementById('language').value,
    triangulation: tri===''?null:parseFloat(tri),
    V:parseFloat(document.getElementById('v').value)
  };
}
async function runForward(){
  document.getElementById('answer').textContent='Running...';
  const r=await fetch('/forward',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
  const d=await r.json();
  if(d.error){ document.getElementById('answer').textContent=d.error; return; }
  document.getElementById('answer').textContent=
    'τ='+d.tau+' halt='+d.halt+'\\n'+
    'of₁ scalar='+d.of1_scalar+' (224-d vector in JSON)\\n'+
    'of₂: '+d.of2_explanation+
    (d.rl_language_context?'\\nRL: '+d.rl_language_context:'');
  document.getElementById('out').textContent=JSON.stringify(d,null,2);
}
async function runJudge(){
  const r=await fetch('/judge',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload())});
  document.getElementById('out').textContent=JSON.stringify(await r.json(),null,2);
}
async function startTrain(){
  const r=await fetch('/train/jsonl',{method:'POST',body:'{}'});
  const d=await r.json();
  log.textContent+='\\n[train] '+JSON.stringify(d)+'\\n';
}
</script>
</body></html>"""
