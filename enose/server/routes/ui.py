"""Serves the chemist-friendly browser UI at GET /ui."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>E-Nose — Smell Classifier</title>
<style>
:root{--pri:#2563eb;--ok:#16a34a;--err:#dc2626;--bg:#f8fafc;--card:#fff;--bdr:#e2e8f0;--txt:#1e293b;--muted:#64748b}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:var(--bg);color:var(--txt)}
.hdr{background:var(--pri);color:#fff;padding:1rem 2rem}
.hdr h1{font-size:1.4rem;font-weight:600}
.hdr p{font-size:.85rem;opacity:.8;margin-top:.25rem}
.wrap{max-width:960px;margin:2rem auto;padding:0 1rem}
.sbar{display:flex;gap:.75rem;margin-bottom:1.5rem;align-items:center;flex-wrap:wrap}
.badge{padding:.25rem .75rem;border-radius:999px;font-size:.75rem;font-weight:600}
.badge.g{background:#dcfce7;color:#15803d}
.badge.r{background:#fee2e2;color:#dc2626}
.badge.y{background:#fef9c3;color:#92400e}
.badge.gr{background:#f1f5f9;color:#64748b}
.tabs{display:flex;gap:.25rem;border-bottom:2px solid var(--bdr);margin-bottom:1.5rem;flex-wrap:wrap}
.tb{padding:.6rem 1.2rem;border:none;background:none;cursor:pointer;font-size:.95rem;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-2px;transition:all .15s}
.tb.on{color:var(--pri);border-bottom-color:var(--pri);font-weight:600}
.tb:hover:not(.on){color:var(--txt)}
.card{background:var(--card);border:1px solid var(--bdr);border-radius:.75rem;padding:1.5rem;margin-bottom:1.5rem}
.card h2{font-size:1rem;font-weight:600;margin-bottom:.25rem}
.hint{font-size:.8rem;color:var(--muted);margin-bottom:1rem}
label{display:block;font-size:.85rem;font-weight:500;margin-bottom:.35rem}
input,textarea,select{width:100%;padding:.5rem .75rem;border:1px solid var(--bdr);border-radius:.5rem;font-size:.9rem;font-family:inherit}
input:focus,textarea:focus{outline:none;border-color:var(--pri);box-shadow:0 0 0 2px #dbeafe}
textarea{resize:vertical;font-family:monospace;font-size:.85rem}
.sgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:.45rem;margin-bottom:1rem}
.sf label{font-size:.72rem;margin-bottom:.15rem;color:var(--muted)}
.sf.r label{color:#1d4ed8}.sf.e label{color:#15803d}
.sf input{padding:.3rem .45rem;font-size:.82rem}
.slbl{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);margin:.75rem 0 .35rem}
.btn{display:inline-block;padding:.55rem 1.25rem;border:none;border-radius:.5rem;font-size:.9rem;font-weight:600;cursor:pointer;transition:opacity .15s}
.btn:hover{opacity:.82}
.bp{background:var(--pri);color:#fff}
.bs{background:#f1f5f9;color:var(--txt)}
.bsuc{background:var(--ok);color:#fff}
.brow{display:flex;gap:.5rem;margin-top:.75rem;flex-wrap:wrap;align-items:center}
.rbox{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:.75rem;padding:1.25rem;margin-top:1rem}
.rbox.err{background:#fef2f2;border-color:#fecaca}
.rlbl{font-size:1.7rem;font-weight:700;color:#15803d;margin-bottom:.4rem}
.rlbl.lo{color:#b45309}
.rconf{font-size:.85rem;color:var(--muted);margin-bottom:1rem}
.plist{display:flex;flex-direction:column;gap:.35rem}
.prow{display:flex;align-items:center;gap:.5rem;font-size:.82rem}
.pname{width:95px;flex-shrink:0;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pbg{flex:1;background:#e2e8f0;border-radius:999px;height:10px}
.pbar{height:10px;border-radius:999px;transition:width .3s}
.ppct{width:42px;text-align:right;color:var(--muted);flex-shrink:0}
.htable{width:100%;border-collapse:collapse;font-size:.84rem}
.htable th{text-align:left;padding:.4rem .5rem;background:#f8fafc;border-bottom:1px solid var(--bdr);font-weight:600}
.htable td{padding:.4rem .5rem;border-bottom:1px solid #f1f5f9}
.htable tr:last-child td{border-bottom:none}
.ctags{display:flex;flex-wrap:wrap;gap:.4rem;margin-top:.5rem}
.ctag{background:#eff6ff;color:#1d4ed8;border-radius:999px;padding:.2rem .6rem;font-size:.8rem}
.div{font-size:.78rem;color:var(--muted);margin:.75rem 0;text-align:center;position:relative}
.div::before,.div::after{content:'';position:absolute;top:50%;width:44%;height:1px;background:var(--bdr)}
.div::before{left:0}.div::after{right:0}
.spin{display:inline-block;width:15px;height:15px;border:2px solid #ccc;border-top-color:var(--pri);border-radius:50%;animation:sp .6s linear infinite;vertical-align:middle}
@keyframes sp{to{transform:rotate(360deg)}}
.hidden{display:none!important}
.msg{padding:.7rem 1rem;border-radius:.5rem;font-size:.84rem;margin-top:.75rem;line-height:1.5}
.msg.ok{background:#f0fdf4;color:#166534;border:1px solid #bbf7d0}
.msg.err{background:#fef2f2;color:#991b1b;border:1px solid #fecaca}
.msg.info{background:#eff6ff;color:#1e40af;border:1px solid #bfdbfe}
.tgrid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-bottom:.75rem}
@media(max-width:500px){.tgrid{grid-template-columns:1fr}}
.lvmeta{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:.5rem;margin:.5rem 0 1rem}
.lvstat{background:#f8fafc;border:1px solid var(--bdr);border-radius:.5rem;padding:.5rem .75rem}
.lvstat .k{font-size:.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.lvstat .v{font-size:1rem;font-weight:600;margin-top:.15rem;font-variant-numeric:tabular-nums}
.lvchart{background:#fff;border:1px solid var(--bdr);border-radius:.5rem;margin-top:.5rem;width:100%;display:block}
.lvdot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:.35rem;vertical-align:middle}
.lvdot.on{background:var(--ok);box-shadow:0 0 0 3px #bbf7d055}
.lvdot.off{background:#94a3b8}
</style>
</head>
<body>

<div class="hdr">
  <h1>&#9879;&#65039; E-Nose &mdash; Smell Classifier</h1>
  <p>Classify odour samples &middot; Train the model &middot; Inspect results</p>
</div>

<div class="wrap">

  <div class="sbar">
    <span style="font-size:.85rem;font-weight:500">Server:</span>
    <span id="srv-b" class="badge gr" title="Server address this page is talking to">&hellip;</span>
    <span id="bk-b" class="badge gr" title="Classifier backend currently loaded on the server">Backend: &hellip;</span>
    <span id="vlm-b" class="badge gr">VLM: &hellip;</span>
    <span id="clf-b" class="badge gr">Classifier: &hellip;</span>
    <span id="cls-b" class="badge gr">Known smells: &mdash;</span>
    <button class="btn bs" style="padding:.25rem .7rem;font-size:.78rem" onclick="refreshStatus()">&#8635; Refresh</button>
  </div>

  <div class="tabs">
    <button class="tb on" onclick="showTab('classify')">&#128269; Classify</button>
    <button class="tb" onclick="showTab('train')">&#128640; Train</button>
    <button class="tb" onclick="showTab('live')">&#128200; Live</button>
    <button class="tb" onclick="showTab('status')">&#128202; Model Info</button>
  </div>

  <!-- CLASSIFY -->
  <div id="tab-classify">
    <div class="card">
      <h2>Enter one sensor reading</h2>
      <p class="hint">Fill in individual fields <em>or</em> paste all 22 numbers in the box below. R1&ndash;R17 are resistance values; T, H, CO2, H2S, CH2O are environmental.</p>

      <div class="slbl">&#128309; Resistance sensors R1 &ndash; R17</div>
      <div class="sgrid" id="rgrid"></div>

      <div class="slbl">&#128994; Environmental sensors</div>
      <div class="sgrid" id="egrid"></div>

      <div class="div">or paste all 22 values at once</div>

      <div style="margin-bottom:.75rem">
        <label for="qp">22 comma-separated values &mdash; R1, R2, &hellip;, R17, T, H, CO2, H2S, CH2O</label>
        <textarea id="qp" rows="2" placeholder="e.g. 15.2, 8.3, 12.1, 3.4, 18.9, 11.2, 9.8, 6.7, 14.3, 10.5, 13.2, 7.8, 5.6, 12.4, 8.9, 6.3, 11.7, 21.0, 49.0, 400, 0.0, 5.0"></textarea>
      </div>

      <div class="brow">
        <button class="btn bp" onclick="classify()">&#128269; Classify smell</button>
        <button class="btn bs" onclick="clearAll()">Clear</button>
        <button class="btn bs" style="font-size:.8rem;padding:.3rem .7rem" onclick="fillEx('coffee')">Try: coffee</button>
        <button class="btn bs" style="font-size:.8rem;padding:.3rem .7rem" onclick="fillEx('air')">Try: air</button>
      </div>
      <div id="cr"></div>
    </div>

    <div class="card">
      <h2>Session history</h2>
      <p class="hint">All classifications made during this browser session.</p>
      <table class="htable">
        <thead><tr><th>#</th><th>Prediction</th><th>Confidence</th><th>Time</th></tr></thead>
        <tbody id="hbody"><tr><td colspan="4" style="color:var(--muted);padding:.5rem">No classifications yet.</td></tr></tbody>
      </table>
    </div>
  </div>

  <!-- TRAIN -->
  <div id="tab-train" class="hidden">
    <div class="card">
      <h2>Train from CSV data</h2>
      <p class="hint">Upload a CSV file where each row is one sensor reading. The file must include columns <strong>R1&ndash;R17</strong>, <strong>T, H, CO2, H2S, CH2O</strong>, and a label column (default name: <em>Gas name</em>).</p>

      <div style="margin-bottom:.75rem">
        <label>Upload CSV file <span style="font-weight:400;color:var(--muted)">(or paste text below)</span></label>
        <input type="file" id="csvf" accept=".csv,.txt" onchange="loadFile(this)">
      </div>

      <div style="margin-bottom:.75rem">
        <label for="csvp">Paste CSV text</label>
        <textarea id="csvp" rows="7" placeholder="R1,R2,...,R17,T,H,CO2,H2S,CH2O,Gas name&#10;15.2,8.3,12.1,...,21.0,49.0,400,0,5,coffee&#10;0.1,0.05,...,21.0,49.0,400,0,5,air"></textarea>
      </div>

      <div class="tgrid">
        <div>
          <label for="tcol">Label column name</label>
          <input id="tcol" value="Gas name">
        </div>
        <div>
          <label for="naug">Augmentations (0 = none, 5 = recommended)</label>
          <input id="naug" type="number" value="5" min="0" max="20">
        </div>
      </div>

      <label style="display:flex;align-items:center;gap:.5rem;font-size:.85rem;cursor:pointer;margin-bottom:.5rem">
        <input type="checkbox" id="uaug" checked style="width:auto"> Use data augmentation (recommended for small datasets)
      </label>
      <label style="display:flex;align-items:center;gap:.5rem;font-size:.85rem;cursor:pointer;margin-bottom:.75rem">
        <input type="checkbox" id="lc" checked style="width:auto"> Convert labels to lowercase automatically
      </label>

      <div class="brow">
        <button class="btn bsuc" onclick="trainCsv()">&#128640; Train model</button>
        <button class="btn bs" onclick="document.getElementById('csvp').value='';document.getElementById('csvf').value=''">Clear</button>
      </div>
      <div id="tr"></div>
    </div>
  </div>

  <!-- LIVE -->
  <div id="tab-live" class="hidden">
    <div class="card">
      <h2>Live sensor stream</h2>
      <p class="hint">Shows samples the client is pushing to <code>/sensor/live/push</code>. Start a client with <code>--live</code> or menu option 8 to begin streaming. Polls the server every second &mdash; works through the tuna.am tunnel.</p>

      <div class="brow" style="margin-bottom:.25rem">
        <button id="lv-tgl" class="btn bp" onclick="lvToggle()">&#9654; Start polling</button>
        <button class="btn bs" onclick="lvClear()" title="Empty the server-side ring buffer">&#128465;&#65039; Clear buffer</button>
        <label style="display:flex;align-items:center;gap:.35rem;font-size:.82rem;margin-left:auto;margin-bottom:0;cursor:pointer">
          <input type="checkbox" id="lv-norm" checked style="width:auto"> Normalize R sensors (per-trace)
        </label>
        <label style="display:flex;align-items:center;gap:.35rem;font-size:.82rem;margin-bottom:0">
          Window:
          <select id="lv-win" style="width:auto;padding:.2rem .4rem;font-size:.8rem" onchange="lvRedraw()">
            <option value="60">60&thinsp;s</option>
            <option value="120" selected>2&thinsp;min</option>
            <option value="300">5&thinsp;min</option>
            <option value="0">all</option>
          </select>
        </label>
      </div>

      <div class="lvmeta">
        <div class="lvstat"><div class="k">Status</div><div class="v"><span id="lv-dot" class="lvdot off"></span><span id="lv-state">idle</span></div></div>
        <div class="lvstat"><div class="k">Samples (buffer)</div><div class="v" id="lv-n">&mdash;</div></div>
        <div class="lvstat"><div class="k">Shown</div><div class="v" id="lv-shown">0</div></div>
        <div class="lvstat"><div class="k">Last label</div><div class="v" id="lv-last">&mdash;</div></div>
        <div class="lvstat"><div class="k">Last conf.</div><div class="v" id="lv-conf">&mdash;</div></div>
      </div>

      <div class="slbl">&#128309; Resistance sensors (R1&ndash;R17)</div>
      <svg id="lv-svgR" class="lvchart" viewBox="0 0 800 220" preserveAspectRatio="none"></svg>

      <div class="slbl">&#128994; Environmental sensors (T / H / CO2 / H2S / CH2O)</div>
      <svg id="lv-svgE" class="lvchart" viewBox="0 0 800 180" preserveAspectRatio="none"></svg>

      <div class="slbl">&#127919; Prediction confidence</div>
      <svg id="lv-svgC" class="lvchart" viewBox="0 0 800 120" preserveAspectRatio="none"></svg>

      <div id="lv-msg"></div>
    </div>

    <div class="card">
      <h2>Sensor drift vs. training data</h2>
      <p class="hint">Compares the live buffer's per-sensor statistics against the mean/std of the <em>training</em> set. Large |z-shift| or a std-ratio far from 1 means the sensor is seeing a distribution the model hasn't been trained on &mdash; so its predictions shouldn't be trusted until the model is retrained on the new regime.</p>
      <div class="brow" style="margin-bottom:.5rem">
        <button class="btn bs" onclick="drRefresh()">&#8635; Refresh drift report</button>
        <span id="dr-sum" style="font-size:.85rem;color:var(--muted);align-self:center"></span>
      </div>
      <div id="dr-body"><em style="color:var(--muted)">Click Refresh to compute drift.</em></div>
    </div>
  </div>

  <!-- STATUS -->
  <div id="tab-status" class="hidden">
    <div class="card">
      <h2>Model information</h2>
      <button class="btn bs" style="font-size:.8rem;padding:.3rem .7rem;margin-bottom:1rem" onclick="loadInfo()">&#8635; Refresh</button>
      <div id="sdet"><span style="color:var(--muted)">Click Refresh to load.</span></div>
    </div>
  </div>

</div><!-- /wrap -->

<script>
const RES=['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10','R11','R12','R13','R14','R15','R16','R17'];
const ENV=['T','H','CO2','H2S','CH2O'];
const ALL=[...RES,...ENV];
const UNITS={T:'°C',H:'%',CO2:'ppm',H2S:'ppm',CH2O:'ppm'};
const EX={
  coffee:[15.2,8.3,12.1,3.4,18.9,11.2,9.8,6.7,14.3,10.5,13.2,7.8,5.6,12.4,8.9,6.3,11.7,21.0,49.0,400,0.0,5.0],
  air:[0.1,0.05,0.08,0.02,0.12,0.09,0.06,0.04,0.11,0.07,0.08,0.05,0.03,0.09,0.06,0.04,0.08,21.0,49.0,400,0.0,5.0],
};

function mkGrid(){
  const rg=document.getElementById('rgrid');
  RES.forEach(n=>{
    const d=document.createElement('div');d.className='sf r';
    d.innerHTML=`<label>${n}</label><input id="s_${n}" type="number" step="any" placeholder="0" oninput="g2p()">`;
    rg.appendChild(d);
  });
  const eg=document.getElementById('egrid');
  ENV.forEach(n=>{
    const d=document.createElement('div');d.className='sf e';
    d.innerHTML=`<label>${n}<span style="font-weight:400;opacity:.7"> ${UNITS[n]||''}</span></label><input id="s_${n}" type="number" step="any" placeholder="0" oninput="g2p()">`;
    eg.appendChild(d);
  });
}
function g2p(){
  document.getElementById('qp').value=ALL.map(n=>{const e=document.getElementById('s_'+n);return e&&e.value!==''?e.value:'0';}).join(', ');
}
function p2g(){
  const parts=document.getElementById('qp').value.split(',').map(x=>x.trim());
  if(parts.length!==22)return;
  ALL.forEach((n,i)=>{const e=document.getElementById('s_'+n);if(e)e.value=parts[i]||'';});
}
document.addEventListener('DOMContentLoaded',()=>{mkGrid();document.getElementById('qp').addEventListener('input',p2g);refreshStatus();});

function showTab(t){
  ['classify','train','live','status'].forEach((x,i)=>{
    document.getElementById('tab-'+x).classList.toggle('hidden',x!==t);
    document.querySelectorAll('.tb')[i].classList.toggle('on',x===t);
  });
  if(t==='status')loadInfo();
  if(t==='live')lvOnShow();
}

function vals(){
  const p=document.getElementById('qp').value.trim();
  if(p){const a=p.split(',').map(x=>parseFloat(x.trim()));if(a.length===22&&a.every(x=>!isNaN(x)))return a;}
  return ALL.map(n=>{const e=document.getElementById('s_'+n);return e?(parseFloat(e.value)||0):0;});
}
function fillEx(k){const v=EX[k];if(!v)return;document.getElementById('qp').value=v.join(', ');p2g();}
function clearAll(){ALL.forEach(n=>{const e=document.getElementById('s_'+n);if(e)e.value='';});document.getElementById('qp').value='';document.getElementById('cr').innerHTML='';}
function msg(id,cls,html){document.getElementById(id).innerHTML=`<div class="msg ${cls}">${html}</div>`;}

let hc=0;
async function classify(){
  const v=vals();
  if(v.length!==22){msg('cr','err','Need exactly 22 values.');return;}
  document.getElementById('cr').innerHTML='<div class="msg info"><span class="spin"></span> Classifying&hellip;</div>';
  try{
    const r=await fetch('/smell/test_console',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({values:v.join(',')})});
    const d=await r.json();
    if(!r.ok){msg('cr','err',d.detail||JSON.stringify(d));return;}
    renderResult(d);addHist(d.predicted_smell,d.confidence);
  }catch(e){msg('cr','err','Request failed: '+e.message);}
}

function renderResult(d){
  const conf=d.confidence||0;const hi=conf>=0.6;
  const probs=d.all_probabilities||{};
  const sorted=Object.entries(probs).sort((a,b)=>b[1]-a[1]);
  const bars=sorted.map(([n,p])=>`<div class="prow">
    <div class="pname" title="${n}">${n}</div>
    <div class="pbg"><div class="pbar" style="width:${(p*100).toFixed(1)}%;background:${p===conf?'#2563eb':'#94a3b8'}"></div></div>
    <div class="ppct">${(p*100).toFixed(1)}%</div></div>`).join('');

  // OOD traffic light — green/yellow/red badge derived from the server's
  // ood.status field. When `available=false` (legacy classifier without
  // centroid stats) we quietly skip the indicator rather than flashing a
  // red "unknown".
  let oodHtml='';
  const ood=d.ood||{};
  if(ood.available){
    const COL={ok:['#16a34a','In-distribution','#dcfce7','#15803d'],
               warn:['#ca8a04','Unusual reading','#fef9c3','#92400e'],
               out:['#dc2626','Out-of-distribution','#fee2e2','#991b1b']};
    const c=COL[ood.status]||COL.warn;
    const help=ood.status==='out'
      ? 'This sample does not resemble any known class &mdash; treat the prediction with caution.'
      : (ood.status==='warn'
          ? 'Partial overlap with training data. Re-measure if you expect a clean reading.'
          : 'Sample falls within the training distribution.');
    oodHtml=`<div style="margin-top:.6rem;padding:.5rem .75rem;background:${c[2]};color:${c[3]};
      border-radius:.4rem;font-size:.82rem">
      <strong style="font-size:.9rem">&#9679; ${c[1]}</strong>
      &nbsp;<span style="opacity:.8">OOD score: ${ood.score}</span>
      ${ood.min_centroid_L2!=null?` &middot; nearest centroid L2: ${ood.min_centroid_L2}`:''}
      <div style="margin-top:.2rem;font-size:.78rem;opacity:.85">${help}</div>
    </div>`;
  }

  document.getElementById('cr').innerHTML=`<div class="rbox">
    <div class="rlbl ${hi?'':'lo'}">${d.predicted_smell}</div>
    <div class="rconf">Confidence: <strong>${(conf*100).toFixed(1)}%</strong>${!hi?' &mdash; low confidence, consider re-testing':''}</div>
    ${oodHtml}
    <div class="plist" style="margin-top:.6rem">${bars}</div></div>`;
}

function addHist(smell,conf){
  hc++;const tb=document.getElementById('hbody');
  if(hc===1)tb.innerHTML='';
  const t=new Date().toLocaleTimeString();
  tb.insertAdjacentHTML('afterbegin',`<tr><td>${hc}</td><td><strong>${smell}</strong></td><td>${(conf*100).toFixed(1)}%</td><td>${t}</td></tr>`);
}

function loadFile(inp){
  const f=inp.files[0];if(!f)return;
  const rd=new FileReader();rd.onload=e=>{document.getElementById('csvp').value=e.target.result;};rd.readAsText(f);
}

async function trainCsv(){
  const csv=document.getElementById('csvp').value.trim();
  if(!csv){msg('tr','err','No CSV data provided.');return;}
  const tc=document.getElementById('tcol').value.trim()||'Gas name';
  const ua=document.getElementById('uaug').checked;
  const na=parseInt(document.getElementById('naug').value)||5;
  const lc=document.getElementById('lc').checked;
  msg('tr','info','<span class="spin"></span> Training&hellip; this may take a minute.');
  try{
    const r=await fetch('/smell/learn_from_csv',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({csv_data:csv,target_column:tc,use_augmentation:ua,n_augmentations:na,lowercase_labels:lc,noise_std:0.0015})});
    const d=await r.json();
    if(!r.ok){msg('tr','err',d.detail||JSON.stringify(d));return;}
    const cls=(d.classes||[]).join(', ')||'(none)';
    const acc=d.current_accuracy!=null?(d.current_accuracy*100).toFixed(2)+'%':'&mdash;';
    msg('tr','ok',`&#10003; Training complete!<br>Samples: ${d.samples_processed} &nbsp;|&nbsp; Accuracy: ${acc}<br>Update: ${d.update_type}<br>Known smells: <strong>${cls}</strong>`);
    refreshStatus();
  }catch(e){msg('tr','err','Request failed: '+e.message);}
}

async function refreshStatus(){
  // Show the server origin the page is talking to — crucial when the same
  // UI can be reached at both http://localhost:8080/ui and via the tuna.am
  // tunnel (http://ru.tuna.am:34200/ui). `window.location.origin` resolves
  // to whichever was actually typed into the browser, so the badge never
  // drifts from reality even behind reverse proxies.
  const sb0=document.getElementById('srv-b');
  sb0.className='badge g';sb0.textContent=window.location.origin;
  try{
    const r=await fetch('/');const d=await r.json();const m=d.models||{};
    const vb=document.getElementById('vlm-b');
    vb.className='badge '+(m.vlm_loaded?'g':'gr');
    vb.textContent='VLM: '+(m.vlm_loaded?'loaded':'not loaded');
    const cb=document.getElementById('clf-b');
    cb.className='badge '+(m.smell_classifier_fitted?'g':(m.smell_classifier_loaded?'y':'r'));
    cb.textContent='Classifier: '+(m.smell_classifier_fitted?'ready':(m.smell_classifier_loaded?'not trained yet':'not loaded'));
    const bk=document.getElementById('bk-b');
    const bn=m.smell_classifier_backend||'unknown';
    bk.className='badge '+(m.smell_classifier_backend?'g':'gr');
    bk.textContent='Backend: '+bn;
  }catch(e){
    // Tunnel down / server unreachable — flip badges red so researchers see
    // the problem instead of a silent stale page.
    const sb=document.getElementById('srv-b');
    sb.className='badge r';sb.textContent=window.location.origin+' (unreachable)';
  }
  try{
    const r=await fetch('/smell/model_info');const d=await r.json();
    const cls=d.classes||[];
    const sb=document.getElementById('cls-b');
    sb.className='badge '+(cls.length?'g':'gr');
    sb.textContent='Known smells: '+(cls.length||'none');
  }catch(e){}
}

async function loadInfo(){
  const el=document.getElementById('sdet');
  el.innerHTML='<span class="spin"></span> Loading&hellip;';
  try{
    const r=await fetch('/smell/model_info');const d=await r.json();
    if(!d.is_fitted){
      el.innerHTML='<div class="msg info">Model is not trained yet. Go to <strong>Train</strong> to upload a CSV file.</div>';return;
    }
    const cls=d.classes||[];
    const h=d.training_history||{};
    const acc=h.accuracy?.slice(-1)[0];
    const bal=h.balanced_accuracy?.slice(-1)[0];
    const tags=cls.map(c=>`<span class="ctag">${c}</span>`).join('');

    // Per-class P/R/F1 table — empty dict when the loaded model predates
    // metric persistence; the hint below nudges the user to retrain.
    const pcm=d.per_class_metrics||{};
    const pcmKeys=Object.keys(pcm);
    let pcmHtml='';
    if(pcmKeys.length){
      const rows=pcmKeys.sort().map(k=>{
        const m=pcm[k]||{};
        const p=(m.precision!=null)?(m.precision*100).toFixed(1)+'%':'—';
        const rc=(m.recall!=null)?(m.recall*100).toFixed(1)+'%':'—';
        const f=(m.f1!=null)?(m.f1*100).toFixed(1)+'%':'—';
        const sup=m.support||0;
        // Colour-code F1 so weak classes jump out at a glance.
        const fv=+m.f1||0;
        const col=fv>=0.85?'#16a34a':(fv>=0.7?'#ca8a04':'#dc2626');
        return `<tr><td><strong>${k}</strong></td><td>${p}</td><td>${rc}</td>
          <td style="color:${col};font-weight:600">${f}</td><td>${sup}</td></tr>`;
      }).join('');
      pcmHtml=`<h3 style="font-size:.95rem;margin-top:1rem;margin-bottom:.5rem">Per-class metrics
        <span style="font-weight:400;color:var(--muted);font-size:.8rem">(last test split: ${d.last_test_size||0} samples)</span></h3>
        <table class="htable">
          <thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>`;
    }else{
      pcmHtml=`<div class="msg info" style="margin-top:1rem">Per-class metrics not yet available &mdash; <strong>retrain</strong> the model to populate this table.</div>`;
    }

    // Confusion matrix — small enough to render as an HTML table.
    let cmHtml='';
    const cm=d.confusion_matrix,cl=d.confusion_labels||[];
    if(cm&&cm.length&&cl.length===cm.length){
      // Find max so we can shade cells by intensity.
      let mx=1;cm.forEach(row=>row.forEach(v=>{if(v>mx)mx=v;}));
      const thead='<tr><th></th>'+cl.map(c=>`<th>${c}</th>`).join('')+'</tr>';
      const body=cm.map((row,i)=>{
        const cells=row.map((v,j)=>{
          const intensity=Math.min(1,v/mx);
          const isDiag=i===j;
          const bg=isDiag?`rgba(22,163,74,${intensity.toFixed(2)})`
                         :(v>0?`rgba(220,38,38,${(intensity*0.75).toFixed(2)})`:'transparent');
          const fg=(intensity>0.5)?'#fff':'var(--txt)';
          return `<td style="text-align:center;background:${bg};color:${fg}">${v||''}</td>`;
        }).join('');
        return `<tr><th style="text-align:right">${cl[i]}</th>${cells}</tr>`;
      }).join('');
      cmHtml=`<h3 style="font-size:.95rem;margin-top:1rem;margin-bottom:.35rem">Confusion matrix
        <span style="font-weight:400;color:var(--muted);font-size:.8rem">rows = true, cols = predicted</span></h3>
        <table class="htable" style="font-variant-numeric:tabular-nums">
          <thead>${thead}</thead><tbody>${body}</tbody></table>`;
    }

    el.innerHTML=`
      <p><strong>Known smells (${cls.length}):</strong></p>
      <div class="ctags">${tags||'<em style="color:var(--muted)">none yet</em>'}</div>
      ${acc!=null?`<p style="margin-top:.75rem"><strong>Last accuracy:</strong> ${(acc*100).toFixed(2)}%
        &nbsp;|&nbsp; Balanced: ${bal!=null?(bal*100).toFixed(2)+'%':'&mdash;'}</p>`:''}
      <p style="margin-top:.5rem"><strong>Training runs:</strong> ${h.accuracy?.length||0}</p>
      ${pcmHtml}
      ${cmHtml}
      <p style="margin-top:1rem;font-size:.82rem;color:var(--muted)">Full API docs: <a href="/docs" target="_blank">/docs</a></p>`;
  }catch(e){el.innerHTML=`<div class="msg err">Error: ${e.message}</div>`;}
}

// ── LIVE TAB ────────────────────────────────────────────────────────────────
// Polling-based (not WebSocket) because tuna.am is a plain TCP tunnel to
// FastAPI/uvicorn. /sensor/live/recent?since=<id> is incremental so the
// wire cost stays bounded no matter how long the tab is left open.
const LV={on:false,timer:null,lastId:0,rows:[],err:0};
const R_PALETTE=['#1d4ed8','#2563eb','#3b82f6','#60a5fa','#7c3aed','#9333ea','#a855f7','#c084fc',
                 '#0891b2','#06b6d4','#22d3ee','#10b981','#059669','#65a30d','#84cc16','#eab308','#f59e0b'];
const E_PALETTE={T:'#dc2626',H:'#2563eb',CO2:'#16a34a',H2S:'#a855f7',CH2O:'#f59e0b'};

function lvOnShow(){
  // Lazy first poll — surface the current buffer state even without starting
  // the stream. Useful to just *peek* what the client has been sending.
  lvPollOnce();
}
function lvToggle(){
  LV.on=!LV.on;
  const btn=document.getElementById('lv-tgl');
  const dot=document.getElementById('lv-dot');
  const st=document.getElementById('lv-state');
  if(LV.on){
    btn.textContent='\u25A0 Stop polling';btn.classList.remove('bp');btn.classList.add('bs');
    dot.className='lvdot on';st.textContent='streaming';
    lvPollOnce();LV.timer=setInterval(lvPollOnce,1000);
  }else{
    btn.textContent='\u25B6 Start polling';btn.classList.remove('bs');btn.classList.add('bp');
    dot.className='lvdot off';st.textContent='idle';
    clearInterval(LV.timer);LV.timer=null;
  }
}
async function lvPollOnce(){
  try{
    const r=await fetch(`/sensor/live/recent?since=${LV.lastId}&limit=500`);
    if(!r.ok)throw new Error('HTTP '+r.status);
    const d=await r.json();
    const items=d.items||[];
    if(items.length){
      // Rebuild from scratch if the server was cleared (last_id regressed).
      if(d.last_id<LV.lastId)LV.rows=[];
      LV.rows=LV.rows.concat(items);
      // Keep memory bounded even when the tab stays open for hours.
      if(LV.rows.length>2000)LV.rows.splice(0,LV.rows.length-2000);
      LV.lastId=d.last_id||LV.lastId;
    }
    document.getElementById('lv-n').textContent=(d.buffered||0)+' / '+(d.capacity||'?');
    LV.err=0;
    lvRedraw();
  }catch(e){
    LV.err++;
    if(LV.err>=3)document.getElementById('lv-msg').innerHTML=
      `<div class="msg err">Polling failed (${e.message}). Is the server reachable?</div>`;
  }
}
async function lvClear(){
  try{
    const r=await fetch('/sensor/live/clear',{method:'DELETE'});
    if(r.ok){LV.rows=[];LV.lastId=0;lvRedraw();
      document.getElementById('lv-msg').innerHTML='<div class="msg ok">Buffer cleared.</div>';
      setTimeout(()=>{document.getElementById('lv-msg').innerHTML='';},2000);}
  }catch(e){}
}

function lvWindowRows(){
  const w=parseFloat(document.getElementById('lv-win').value)||0;
  if(!LV.rows.length)return [];
  if(w<=0)return LV.rows;
  const tmax=LV.rows[LV.rows.length-1].t;
  return LV.rows.filter(r=>tmax-r.t<=w);
}

function lvRedraw(){
  const rows=lvWindowRows();
  document.getElementById('lv-shown').textContent=rows.length;
  if(rows.length){
    const last=rows[rows.length-1];
    document.getElementById('lv-last').textContent=last.predicted||last.label||'\u2014';
    document.getElementById('lv-conf').textContent=
      (last.confidence!=null?(last.confidence*100).toFixed(1)+'%':'\u2014');
  }else{
    document.getElementById('lv-last').textContent='\u2014';
    document.getElementById('lv-conf').textContent='\u2014';
  }
  const norm=document.getElementById('lv-norm').checked;
  lvDrawPanel('lv-svgR',rows,RES,R_PALETTE,{normPerTrace:norm,width:800,height:220});
  lvDrawPanel('lv-svgE',rows,ENV,ENV.map(n=>E_PALETTE[n]),{normPerTrace:true,width:800,height:180});
  lvDrawConf('lv-svgC',rows);
}

function lvDrawPanel(id,rows,names,palette,opt){
  const svg=document.getElementById(id);
  const W=opt.width,H=opt.height,PAD_L=28,PAD_R=90,PAD_T=8,PAD_B=20;
  const plotW=W-PAD_L-PAD_R,plotH=H-PAD_T-PAD_B;
  if(!rows.length){
    svg.innerHTML=`<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="#94a3b8" font-size="13">
      No live samples yet. Start a client with <tspan font-family="monospace">--live</tspan>.</text>`;
    return;
  }
  const t0=rows[0].t,t1=rows[rows.length-1].t||t0+1,dt=Math.max(t1-t0,1e-6);
  // Pre-extract per-sensor series.
  const series=names.map(n=>rows.map(r=>{const v=(r.sample||{})[n];return v==null||isNaN(v)?null:+v;}));
  let parts=[];
  // Axes.
  parts.push(`<rect x="${PAD_L}" y="${PAD_T}" width="${plotW}" height="${plotH}" fill="#fafafa" stroke="#e2e8f0"/>`);
  for(let g=0;g<=4;g++){
    const y=PAD_T+(plotH*g/4);
    parts.push(`<line x1="${PAD_L}" y1="${y}" x2="${PAD_L+plotW}" y2="${y}" stroke="#f1f5f9"/>`);
  }
  parts.push(`<text x="${PAD_L}" y="${H-4}" font-size="10" fill="#64748b">t-${(dt).toFixed(1)}s</text>`);
  parts.push(`<text x="${PAD_L+plotW}" y="${H-4}" font-size="10" fill="#64748b" text-anchor="end">now</text>`);
  // Series.
  names.forEach((n,i)=>{
    const s=series[i];
    const clean=s.filter(v=>v!=null);
    if(!clean.length)return;
    let mn,mx;
    if(opt.normPerTrace){mn=Math.min(...clean);mx=Math.max(...clean);}
    else{const all=series.flat().filter(v=>v!=null);mn=Math.min(...all);mx=Math.max(...all);}
    const rng=mx-mn||1;
    let path='',first=true;
    rows.forEach((r,k)=>{
      const v=s[k];if(v==null){first=true;return;}
      const x=PAD_L+((r.t-t0)/dt)*plotW;
      const y=PAD_T+plotH-((v-mn)/rng)*plotH;
      path+=(first?'M':'L')+x.toFixed(1)+' '+y.toFixed(1)+' ';
      first=false;
    });
    parts.push(`<path d="${path}" fill="none" stroke="${palette[i%palette.length]}" stroke-width="1.3" stroke-linejoin="round"/>`);
    // Legend.
    const ly=PAD_T+12+i*13;
    if(ly<H-4){
      parts.push(`<rect x="${PAD_L+plotW+8}" y="${ly-8}" width="10" height="3" fill="${palette[i%palette.length]}"/>`);
      const lastVal=clean[clean.length-1];
      parts.push(`<text x="${PAD_L+plotW+22}" y="${ly-4}" font-size="10" fill="#475569" font-family="monospace">${n} ${lastVal.toFixed(2)}</text>`);
    }
  });
  svg.innerHTML=parts.join('');
}

function lvDrawConf(id,rows){
  const svg=document.getElementById(id);
  const W=800,H=120,PAD_L=28,PAD_R=90,PAD_T=8,PAD_B=20;
  const plotW=W-PAD_L-PAD_R,plotH=H-PAD_T-PAD_B;
  const confd=rows.filter(r=>r.confidence!=null);
  if(!confd.length){
    svg.innerHTML=`<text x="${W/2}" y="${H/2}" text-anchor="middle" fill="#94a3b8" font-size="12">
      Run the client with <tspan font-family="monospace">--live-classify</tspan> to see confidence over time.</text>`;
    return;
  }
  const t0=rows[0].t,t1=rows[rows.length-1].t||t0+1,dt=Math.max(t1-t0,1e-6);
  let parts=[];
  parts.push(`<rect x="${PAD_L}" y="${PAD_T}" width="${plotW}" height="${plotH}" fill="#fafafa" stroke="#e2e8f0"/>`);
  // 0.6 threshold line.
  const ythr=PAD_T+plotH*(1-0.6);
  parts.push(`<line x1="${PAD_L}" y1="${ythr}" x2="${PAD_L+plotW}" y2="${ythr}" stroke="#fca5a5" stroke-dasharray="4 3"/>`);
  parts.push(`<text x="${PAD_L+plotW+4}" y="${ythr+3}" font-size="10" fill="#dc2626">0.6</text>`);
  // Colour each segment by its predicted class for a cheap visual grouping.
  const classColor={};let palIdx=0;
  const basePal=['#2563eb','#16a34a','#dc2626','#9333ea','#f59e0b','#0891b2','#db2777','#65a30d'];
  function colorFor(c){if(!c)return '#94a3b8';if(!(c in classColor))classColor[c]=basePal[(palIdx++)%basePal.length];return classColor[c];}
  // Draw as coloured dots + thin connecting line.
  let path='',first=true;
  confd.forEach(r=>{
    const x=PAD_L+((r.t-t0)/dt)*plotW;
    const y=PAD_T+plotH-Math.max(0,Math.min(1,r.confidence))*plotH;
    path+=(first?'M':'L')+x.toFixed(1)+' '+y.toFixed(1)+' ';
    first=false;
    parts.push(`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="2.5" fill="${colorFor(r.predicted)}"/>`);
  });
  parts.push(`<path d="${path}" fill="none" stroke="#cbd5e1" stroke-width="1"/>`);
  // Legend of classes seen.
  let ly=PAD_T+12;
  Object.keys(classColor).forEach(c=>{
    if(ly<H-4){
      parts.push(`<rect x="${PAD_L+plotW+8}" y="${ly-8}" width="10" height="8" fill="${classColor[c]}"/>`);
      parts.push(`<text x="${PAD_L+plotW+22}" y="${ly-1}" font-size="10" fill="#475569">${c}</text>`);
      ly+=13;
    }
  });
  parts.push(`<text x="${PAD_L-4}" y="${PAD_T+6}" font-size="10" fill="#64748b" text-anchor="end">1.0</text>`);
  parts.push(`<text x="${PAD_L-4}" y="${PAD_T+plotH}" font-size="10" fill="#64748b" text-anchor="end">0</text>`);
  svg.innerHTML=parts.join('');
}

// ── DRIFT ───────────────────────────────────────────────────────────────────
async function drRefresh(){
  const body=document.getElementById('dr-body');
  const sum=document.getElementById('dr-sum');
  body.innerHTML='<span class="spin"></span> Computing drift&hellip;';
  sum.textContent='';
  try{
    const r=await fetch('/smell/drift');
    const d=await r.json();
    if(!r.ok){body.innerHTML=`<div class="msg err">${d.detail||'error'}</div>`;return;}
    const ps=d.per_sensor||{};
    const sensors=Object.keys(ps);
    if(!sensors.length){body.innerHTML='<div class="msg info">No sensors to compare yet.</div>';return;}
    // Overall badge.
    const st=d.overall_status||'ok';
    const label={ok:'In-range',warn:'Shifting',out:'Drifted'}[st]||st;
    const badgeCls={ok:'g',warn:'y',out:'r'}[st]||'gr';
    sum.innerHTML=`<span class="badge ${badgeCls}">${label}</span>
      &nbsp;live=${d.live_buffer_n} / train=${d.train_n}`;
    // Sort: drifted first, then warn, then ok, then na — within a tier by |z|.
    const rank={out:0,warn:1,ok:2,na:3};
    const ordered=sensors.slice().sort((a,b)=>{
      const ra=rank[ps[a].status]||3,rb=rank[ps[b].status]||3;
      if(ra!==rb)return ra-rb;
      return (Math.abs(ps[b].z_shift||0))-(Math.abs(ps[a].z_shift||0));
    });
    const rows=ordered.map(n=>{
      const m=ps[n];
      const col={out:'#dc2626',warn:'#ca8a04',ok:'#16a34a',na:'#94a3b8'}[m.status]||'#64748b';
      const z=(m.z_shift!=null)?m.z_shift.toFixed(2):'&mdash;';
      const sr=(m.std_ratio!=null)?m.std_ratio.toFixed(2):'&mdash;';
      const lm=(m.live_mean!=null)?m.live_mean.toFixed(3):'&mdash;';
      const tm=(m.train_mean!=null)?m.train_mean.toFixed(3):'&mdash;';
      return `<tr>
        <td><strong>${n}</strong></td>
        <td style="color:${col};font-weight:600">${m.status.toUpperCase()}</td>
        <td style="font-variant-numeric:tabular-nums">${tm}</td>
        <td style="font-variant-numeric:tabular-nums">${lm}</td>
        <td style="font-variant-numeric:tabular-nums">${z}</td>
        <td style="font-variant-numeric:tabular-nums">${sr}</td>
        <td>${m.live_n||0}</td>
      </tr>`;
    }).join('');
    body.innerHTML=`<table class="htable">
      <thead><tr><th>Sensor</th><th>Status</th><th>Train mean</th><th>Live mean</th>
        <th>z-shift</th><th>std ratio</th><th>Live n</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <p style="margin-top:.5rem;font-size:.78rem;color:var(--muted)">
        Thresholds &mdash; warn: |z|&gt;${d.thresholds.z_warn}, std-ratio outside [${d.thresholds.std_ratio_warn.join(', ')}];
        out: |z|&gt;${d.thresholds.z_out}, std-ratio outside [${d.thresholds.std_ratio_out.join(', ')}].
      </p>`;
  }catch(e){body.innerHTML=`<div class="msg err">Request failed: ${e.message}</div>`;}
}
</script>
</body>
</html>"""


@router.get("/ui", response_class=HTMLResponse, include_in_schema=False)
async def web_ui() -> HTMLResponse:
    """Browser UI for classifying smells and training the model without any coding."""
    return HTMLResponse(content=_HTML)
