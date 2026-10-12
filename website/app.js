(() => {
'use strict';
const DATA=JSON.parse(document.getElementById('case-data').textContent), app=document.getElementById('app'), dialog=document.getElementById('evidence-dialog');
const SITE_TITLE='ARIA: Runtime Inspection and Amendment for Multi-Agent Systems';
const wording=value=>String(value??'').replace(/\bdetect(?:ion)?\b/gi,'inspect').replace(/\brepaired\b/gi,'amended').replace(/\brepairs?\b/gi,'amend');
const keys=Object.keys(DATA.cases),motion=matchMedia('(prefers-reduced-motion: reduce)');
const state={key:DATA.default,selected:0,ariaAttached:true,heroPhase:motion.matches?4:0,heroPlaying:!motion.matches,heroElapsed:0,heroManual:motion.matches,casePhase:0,casePlaying:!motion.matches,caseElapsed:0,caseManual:motion.matches};
const heroDurations=[2200,3400,4000,3800,3400],caseDurations=[3600,3000,3600,3000];
let clockFrame,lastFrame=0,heroVisible=true,caseVisible=false,caseEvent=-1,evidenceReturnFocus='[data-action="evidence"]';
const progressTrack='<span class="tab-progress" aria-hidden="true"><i></i></span>';
const icons={arrow:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>',check:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',close:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',search:'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></svg>',branch:'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M6 7v10m12-10c0 5-12 2-12 8"/></svg>',file:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H5v18h14V8ZM14 3v5h5M8 12h8m-8 4h6"/></svg>',play:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7Z"/></svg>',pause:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>',chip:'<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="3"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4M10 10h4v4h-4Z"/></svg>'};
const resourceIcons={"github":"<svg viewBox=\"0 0 16 16\" aria-hidden=\"true\"><path fill=\"currentColor\" d=\"M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82A7.65 7.65 0 0 1 8 3.86c.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z\"/></svg>","huggingface":"<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 95 88\" aria-hidden=\"true\" fill=\"none\"> <path fill=\"#FFD21E\" d=\"M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z\" /> <path fill=\"#FF9D0B\" d=\"M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z\" /> <path fill=\"#3A3B45\" d=\"M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z\" /> <path fill=\"#FF323D\" d=\"M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z\" /> <path fill=\"#3A3B45\" fill-rule=\"evenodd\" d=\"M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z\" clip-rule=\"evenodd\" /> <path fill=\"#FF9D0B\" d=\"M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z\" /> <path fill=\"#FFD21E\" d=\"M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z\" /> <path fill=\"#FF9D0B\" d=\"M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z\" /> <path fill=\"#FFD21E\" d=\"M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z\" /> </svg>"};
const resourceLinks=()=>`<a class="resource-link" href="https://github.com/LabRAI/ARIA" target="_blank" rel="noopener noreferrer" aria-label="ARIA on GitHub (opens in a new tab)" title="GitHub">${resourceIcons.github}</a><a class="resource-link" href="https://huggingface.co/datasets/Edlison/ARIA" target="_blank" rel="noopener noreferrer" aria-label="ARIA dataset on Hugging Face (opens in a new tab)" title="Hugging Face">${resourceIcons.huggingface}</a>`;
const esc=value=>String(value??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]));
const c=()=>DATA.cases[state.key];
const actor=id=>c().actors.find(a=>a.id===id)||{id,name:id==='host'?'Host':id,role:'',kind:'host'};
const identity=id=>{const a=actor(id);return `<strong>${esc(a.name)}</strong>${a.native_name&&a.native_name!==a.name?`<span class="native-name">${esc(a.native_name)}</span>`:''}`;};
const badge=(text,cls='')=>`<span class="badge ${cls}">${esc(text)}</span>`;
const code=value=>`<pre><code>${esc(value)}</code></pre>`;
const stepId=e=>e.event_id.startsWith('source:event:')?e.event_id.split(':').pop():'artifact';
const badIndex=()=>c().conversation.findIndex(e=>e.first_bad);
function readRoute(){const key=location.hash.slice(1).split('/')[0];if(keys.includes(key))state.key=key;state.selected=badIndex();}
readRoute();
const phases=['Connect','Run','Inspect','Amend','Resume'];
const captions=['Connect the ARIA module.','The agents work. ARIA watches.','Inspect the faulty decision.','Amend it. Keep the good work.','The amended trajectory succeeds.'];
const synopsis={
 'humaneval':'QA flips the correct even-index condition to odd.',
 'mmlu':'QA replaces the team’s correct answer C with B.',
 'gsm-plus':'The task contract mistakes break-even for positive profit.',
 'gaia':'The solver counts west-facing fronts instead of west-facing backs.',
 'olympiadbench':'The solver presents a low-speed approximation as an exact result.',
 'swe-bench-lite':'The plan assumes __add__ exists and implements only __radd__.'
};
function shell(){return `<main id="pages" aria-label="ARIA interactive exhibit" tabindex="-1">
<section class="screen landing" id="top" aria-labelledby="hero-title">
<header class="topbar"><a class="brand" href="#top"><span class="brand-icon">${icons.chip}</span>ARIA</a><nav class="header-actions" aria-label="Project links">${resourceLinks()}</nav></header>
<div class="landing-content"><div class="hero-copy"><h1 id="hero-title"><span class="hero-name">ARIA:</span> Runtime Inspection and Amendment for Multi-Agent Systems</h1><p>Connect ARIA to your agent team. Inspect and amend its trajectory while the work unfolds.</p><a class="primary-button" href="#studies">See what changes ${icons.arrow}</a></div>
<div class="hero-exhibit" data-phase="${state.heroPhase}" data-attached="true">
<div class="module-toolbar"><button class="module-toggle" data-action="aria-toggle" aria-pressed="true"><i class="toggle-indicator" aria-hidden="true"></i><span id="module-toggle-label">Remove ARIA</span></button></div>
<div class="machine" role="group" aria-label="Concept animation: ARIA connects to a multi-agent system, locates a faulty decision in Agent 03, replaces it with a corrected decision, and resumes the trajectory.">

<div class="machine-top"><span class="engraved">MULTI-AGENT SYSTEM</span><span class="system-status"><i class="led"></i><span id="system-status">Connecting ARIA</span></span></div>
<div class="machine-stage"><div class="module-bay"><span class="bay-empty">No module connected</span><div class="aria-module"><div class="chip-face">${icons.chip}<strong>ARIA</strong><span class="chip-state">INSPECT + AMEND</span><i class="led"></i></div></div></div>
<svg class="hardware-wires" viewBox="0 0 720 360" preserveAspectRatio="none" aria-hidden="true"><path class="cable cable-shadow" d="M360 67V98Q360 111 380 111H454Q474 111 474 132V168"/><path class="cable" d="M360 67V98Q360 111 380 111H454Q474 111 474 132V168"/><path class="connection-flow" d="M360 67V98Q360 111 380 111H454Q474 111 474 132V168"/><path class="track-shadow" d="M75 196H642"/><path class="track-line" d="M75 196H642"/><path class="track-pulse pulse-in" d="M75 196H470"/><path class="track-pulse pulse-out" d="M470 196H642"/><path class="probe-line" d="M470 224V278Q470 290 452 290H360V307"/></svg>
<div class="runtime-agents">${['Plan','Build','Review','Deliver'].map((name,i)=>`<div class="agent-module agent-${i}"><div class="agent-tile"><span class="agent-topline"><i class="module-led"></i></span><span class="agent-glyph" ${i===3?'id="hero-delivery-icon"':''} aria-hidden="true">${[icons.branch,icons.file,icons.search,icons.check][i]}</span><strong>${name}</strong></div>${i===2?'<span class="fault-flag">! FAULT HERE</span><div class="scanner" aria-hidden="true"></div>':''}</div>`).join('')}</div>
<div class="decision-slot"><span class="slot-label" id="hero-decision-label">Inspect</span><div class="decision-carriage"><div class="old-cartridge"><span>×</span>Violates the task constraint</div><div class="new-cartridge"><span>✓</span>Respects the task constraint</div></div><span class="slot-status" id="slot-status">Awaiting inspection</span></div>
</div><div class="machine-bottom"><span class="hero-result" id="hero-result" data-status="passed" role="status">${icons.check}<span id="machine-outcome">AMENDED OUTCOME</span></span></div></div>
<div class="demo-caption"><h2 id="hero-caption">Deploy ARIA onto the team.</h2><button class="round-button" data-action="hero-toggle" aria-label="Pause animation">${icons.pause}</button></div>
<div class="demo-controls" role="group" aria-label="Concept animation phases">${phases.map((name,i)=>`<button data-hero-phase="${i}" aria-pressed="false">${progressTrack}${name}</button>`).join('')}</div>

</div></div><footer class="screen-footer"><a href="#studies">Explore benchmarks ↓</a></footer></section>
<section class="screen studies" id="studies" aria-labelledby="studies-title"><header class="case-topbar"><a class="brand" href="#top"><span class="brand-icon">${icons.chip}</span>ARIA</a><h2 id="studies-title">Inspect. <em>Amend.</em> Succeed.</h2><nav class="header-actions" aria-label="Project links"><a class="back-top" href="#top" aria-label="Back to concept animation">↑</a>${resourceLinks()}</nav></header>
<nav class="benchmark-nav" aria-label="Benchmarks">${DATA.manifest.cases.map((entry,i)=>keys.includes(entry.benchmark)?`<button class="case-link" data-case="${entry.benchmark}">${progressTrack}${esc(entry.label)}<span class="benchmark-progress" role="progressbar" aria-label="${esc(entry.label)} playback" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0"></span></button>`:`<a class="case-link" href="${entry.benchmark}.html#${entry.benchmark}">${progressTrack}${esc(entry.label)}</a>`).join('')}</nav>

<div id="case-content" class="case-canvas" aria-label="Selected case study"></div></section></main>`;}
function diffParts(before,after){
 const tokenize=s=>s.match(/\s+|[\p{L}_][\p{L}\p{N}_]*|\d+(?:\.\d+)?|==|!=|>=|<=|[^\s]/gu)||[];
 const a=tokenize(before),b=tokenize(after);let start=0,end=0;
 while(start<Math.min(a.length,b.length)&&a[start]===b[start])start++;
 while(end<Math.min(a.length,b.length)-start&&a[a.length-1-end]===b[b.length-1-end])end++;
 const paint=t=>esc(t.slice(0,start).join(''))+`<mark>${esc(t.slice(start,end?t.length-end:t.length).join(''))}</mark>`+esc(end?t.slice(-end).join(''):'');
 return [paint(a),paint(b)];
}
function renderCase(){
 const item=c(),n=item.review,bad=item.conversation[badIndex()], [before,after]=diffParts(n.before,n.after);
 document.title=SITE_TITLE;
 document.querySelectorAll('[data-case]').forEach(b=>{b.classList.toggle('active',b.dataset.case===state.key);b.setAttribute('aria-current',b.dataset.case===state.key?'page':'false');});
 state.casePhase=motion.matches?3:0;state.casePlaying=!motion.matches;state.caseElapsed=0;state.caseManual=motion.matches;caseEvent=-1;
 document.getElementById('case-content').innerHTML=`<article class="case-study" aria-labelledby="case-title"><div class="case-heading"><h3 id="case-title">${esc(wording(n.title))}</h3><div class="case-actions"><button class="play-button" data-action="case-play" aria-label="Pause amendment playback">${icons.pause}</button><button class="reset-button" data-action="case-reset" aria-label="Restart amendment playback">↺</button><button class="text-button" data-action="evidence">Evidence ↗</button></div></div>
<div class="task-line"><p>${esc(wording(n.problem_summary))}</p><button class="text-button" data-action="task">Full task ↗</button></div>
<div class="case-console" data-phase="${state.casePhase}"><div class="trace-heading"><span class="engraved">Agent trajectory</span><span class="trace-state" id="trace-state"></span></div>
<div class="trajectory-rail" tabindex="0" aria-label="Original events in recorded order"><div class="trajectory-track"><div class="task-node"><span class="node-cap">INPUT</span>${icons.file}<strong>Task</strong></div>${item.conversation.map((e,i)=>`<button class="trace-node ${e.first_bad?'fault-node':i>badIndex()?'downstream-node':'retained-node'}" data-step="${i}" data-event="${esc(e.event_id)}" ${e.first_bad?'aria-label="Error starts here: '+esc(actor(e.actor).name)+' at event '+esc(stepId(e))+'"':''}><span class="node-cap">${e.first_bad?'! ERROR STARTS HERE':'EVENT '+esc(stepId(e))}</span><span class="node-body"><span class="agent-medallion">${e.first_bad?'!':esc(actor(e.actor).initials||'·')}</span><strong>${esc(actor(e.actor).name)}</strong></span></button>`).join('')}<div class="failed-end"><span>×</span>Failed</div></div></div>
<div class="amend-bench"><div class="fault-location"><div><h4 class="intervention-title">${icons.chip} ARIA <em>Inspect</em></h4><h4>${esc(actor(bad.actor).name)} <span class="inspect-event">Event ${esc(stepId(bad))}</span></h4><p>${esc(synopsis[state.key])}</p><button class="text-button" data-source-step="${badIndex()}">Inspect message ↗</button></div></div>
<div class="replacement"><div class="replacement-head"><span class="mini-chip">${icons.chip}ARIA</span><strong id="replacement-title">Amend</strong><button class="text-button" data-action="patch">Full amendment ↗</button></div><div class="diff-pair"><div class="diff-side before"><div class="diff-label"><span>−</span> Wrong decision</div><pre class="code-face ${n.before.length>120?'long-code':''}"><code>${before}</code></pre></div><div class="swap-arrow" aria-hidden="true">${icons.arrow}</div><div class="diff-side after"><div class="diff-label"><span>+</span> Amended decision</div><pre class="code-face ${n.after.length>120?'long-code':''}"><code>${after}</code></pre></div></div></div></div>
<div class="outcome-rail"><div class="outcome failed"><span class="label">Original</span><strong>${icons.close}${esc(wording(n.result_values.failed))}</strong><button data-run="failed" aria-label="Original saved evaluation">↗</button></div><span class="outcome-arrow" aria-hidden="true">→</span><div class="outcome passed"><span class="label">After ARIA</span><strong>${icons.check}${esc(wording(n.result_values.repaired))}</strong><button data-run="repaired" aria-label="Amended saved evaluation">↗</button></div></div></div></article>`;
 centerRails();
 setCasePhase(state.casePhase);updateCaseButton();
 paintProgress();

}
function centerRails(){
 const rail=document.querySelector('.trajectory-rail'),fault=document.querySelector('.fault-node');
 const current=state.casePhase===0?document.querySelector('.trace-node.is-current'):fault;
 if(rail&&current)rail.scrollLeft=current.offsetLeft-rail.offsetLeft-(rail.clientWidth-current.offsetWidth)/2;
 const nav=document.querySelector('.benchmark-nav'),active=nav.querySelector('.active');
 if(active)nav.scrollLeft=active.offsetLeft-nav.offsetLeft-(nav.clientWidth-active.offsetWidth)/2;
}
window.addEventListener('resize',()=>requestAnimationFrame(centerRails));
function updateHero(){
 const p=state.heroPhase,exhibit=document.querySelector('.hero-exhibit');exhibit.dataset.phase=p;
 exhibit.dataset.attached=String(state.ariaAttached);
 document.querySelector('.bay-empty').setAttribute('aria-hidden',String(state.ariaAttached));
 document.querySelector('.aria-module').setAttribute('aria-hidden',String(!state.ariaAttached));
 document.querySelector('.fault-flag').setAttribute('aria-hidden',String(state.ariaAttached&&p!==2));
 document.querySelector('.old-cartridge').setAttribute('aria-hidden',String(state.ariaAttached&&p!==2));
 document.querySelector('.new-cartridge').setAttribute('aria-hidden',String(!state.ariaAttached||p<3));
 exhibit.classList.toggle('is-playing',heroCanPlay());
 document.getElementById('hero-caption').textContent=state.ariaAttached?captions[p]:'Without ARIA, the mistake reaches the output.';
 document.getElementById('hero-decision-label').textContent=state.ariaAttached&&p>=3?'Amend':'Inspect';
 document.getElementById('system-status').textContent=state.ariaAttached?['Connecting ARIA','System running','Inspecting decision','Applying amendment','Execution succeeded'][p]:'ARIA disconnected';
 document.getElementById('slot-status').textContent=state.ariaAttached?['Observer connected','Monitoring decisions','Fault localized at Review','Amendment applied','Amended continuation'][p]:'Fault continues without amendment';
 const result=document.getElementById('hero-result');result.dataset.status=state.ariaAttached?'passed':'failed';
 result.innerHTML=`${state.ariaAttached?icons.check:icons.close}<span id="machine-outcome">${state.ariaAttached?'AMENDED OUTCOME':'FAILED OUTCOME'}</span>`;
 document.getElementById('hero-delivery-icon').innerHTML=state.ariaAttached?icons.check:icons.close;
 document.querySelectorAll('[data-hero-phase]').forEach(b=>{b.disabled=!state.ariaAttached;b.setAttribute('aria-pressed',String(state.ariaAttached&&Number(b.dataset.heroPhase)===p));});
 const toggle=document.querySelector('[data-action="aria-toggle"]');toggle.setAttribute('aria-pressed',String(state.ariaAttached));
 document.getElementById('module-toggle-label').textContent=state.ariaAttached?'Remove ARIA':'Connect ARIA';
 const b=document.querySelector('[data-action="hero-toggle"]');b.disabled=!state.ariaAttached;b.innerHTML=motion.matches?icons.arrow:state.heroPlaying?icons.pause:icons.play;b.setAttribute('aria-label',motion.matches?'Next animation phase':state.heroPlaying?'Pause animation':'Play animation');

}
function heroCanPlay(){return state.ariaAttached&&state.heroPlaying&&heroVisible&&!document.hidden&&!motion.matches&&!dialog.open;}
function caseCanPlay(){return state.casePlaying&&caseVisible&&!document.hidden&&!motion.matches&&!dialog.open;}
function paintPhaseProgress(selector,phase,fraction,enabled=true){
 document.querySelectorAll(selector).forEach(b=>{const i=Number(b.dataset.heroPhase??b.dataset.casePhase);b.style.setProperty('--progress',enabled?(i<phase?1:i===phase?fraction:0):0);b.classList.toggle('is-complete',enabled&&i<phase);});
}
function paintProgress(){
 const hf=state.heroManual?1:Math.min(1,state.heroElapsed/heroDurations[state.heroPhase]),cf=state.caseManual?1:Math.min(1,state.caseElapsed/caseDurations[state.casePhase]);
 paintPhaseProgress('[data-hero-phase]',state.heroPhase,hf,state.ariaAttached);

 const cp=Math.floor((state.casePhase+cf)/4*100);
 document.querySelectorAll('[data-case]').forEach(b=>{
  const active=b.dataset.case===state.key;b.style.setProperty('--progress',active?cp/100:0);
  const indicator=b.querySelector('.benchmark-progress');
  indicator.setAttribute('aria-valuenow',String(active?cp:0));
 });
 updateTrajectory(cf);
}
// A single clock drives phase changes and fills, so pause/resume keeps the same position.
function startClock(){
 if(clockFrame)cancelAnimationFrame(clockFrame);
 clockFrame=undefined;lastFrame=performance.now();paintProgress();
 if(heroCanPlay()||caseCanPlay())clockFrame=requestAnimationFrame(tick);
}
function tick(now){
 const delta=Math.max(0,now-lastFrame);lastFrame=Math.max(lastFrame,now);clockFrame=undefined;
 if(heroCanPlay()){
  state.heroElapsed+=delta;
  while(state.heroElapsed>=heroDurations[state.heroPhase]){state.heroElapsed-=heroDurations[state.heroPhase];state.heroPhase=(state.heroPhase+1)%phases.length;updateHero();}
 }
 if(caseCanPlay()){
  state.caseElapsed+=delta;
  while(state.caseElapsed>=caseDurations[state.casePhase]&&state.casePlaying){
   if(state.casePhase===3){state.caseElapsed=caseDurations[3];stopCase();}
   else{state.caseElapsed-=caseDurations[state.casePhase];setCasePhase(state.casePhase+1);}
  }
 }
 paintProgress();
 if(heroCanPlay()||caseCanPlay())clockFrame=requestAnimationFrame(tick);
}
function scheduleHero(){updateHero();startClock();}
function setCasePhase(phase){
 state.casePhase=phase;caseEvent=-1;
 document.querySelector('.case-console').dataset.phase=phase;

 document.getElementById('trace-state').textContent=['Running',`ARIA Inspect · Event ${stepId(c().conversation[badIndex()])}`,'ARIA Amend · Decision replaced','Amendment succeeded'][phase];
 const fault=document.querySelector('.fault-node');
 fault.querySelector('.node-cap').textContent=phase===0?`EVENT ${stepId(c().conversation[badIndex()])}`:phase===1?'ARIA INSPECT':'ARIA AMEND';
 fault.querySelector('.agent-medallion').textContent=phase===0?(actor(c().conversation[badIndex()].actor).initials||'·'):phase===1?'!':'✓';
 document.getElementById('replacement-title').textContent=['Amend','Amend','Amend · Decision replaced','Amend · Complete'][phase];
 const end=document.querySelector('.failed-end');end.classList.toggle('is-passed',phase===3);end.innerHTML=phase===3?`${icons.check} Success`:phase===0?`${icons.arrow} Running`:`${icons.close} Failed`;
 // Reveal saved evidence as the story reaches its phase; all source links stay available.
 document.querySelector('.fault-location').setAttribute('aria-hidden','false');
 document.querySelector('.after .code-face').setAttribute('aria-hidden',String(phase<2));
 document.querySelector('.outcome.passed strong').setAttribute('aria-hidden',String(phase<3));
 document.querySelector('.outcome.failed strong').setAttribute('aria-hidden',String(phase===0));
 centerRails();
}
function updateTrajectory(fraction){
 const nodes=[...document.querySelectorAll('.trace-node')];
 const index=state.casePhase===0?Math.min(nodes.length-1,Math.floor(fraction*nodes.length)):badIndex();
 if(index===caseEvent)return;caseEvent=index;
 nodes.forEach((node,i)=>{node.classList.toggle('is-current',state.casePhase===0&&i===index);node.classList.toggle('is-visited',state.casePhase===0&&i<index);});
 if(state.casePhase===0){document.getElementById('trace-state').textContent=`Running · Event ${stepId(c().conversation[index])} · ${actor(c().conversation[index].actor).name}`;centerRails();}
}
function caseComplete(){return state.casePhase===3&&(state.caseManual||state.caseElapsed>=caseDurations[3]);}
function updateCaseButton(){
 const button=document.querySelector('[data-action="case-play"]');
 button.innerHTML=state.casePlaying?icons.pause:icons.play;
 button.setAttribute('aria-label',state.casePlaying?'Pause amendment playback':caseComplete()?'Replay amendment':'Continue amendment');
 document.querySelector('.case-console').classList.toggle('is-playing',caseCanPlay());

}
function stopCase(){state.casePlaying=false;updateCaseButton();paintProgress();}
function playCase(){
 if(state.casePlaying){stopCase();startClock();return;}
 if(caseComplete()){state.caseElapsed=0;setCasePhase(0);}
 if(state.caseManual)state.caseElapsed=0;
 state.caseManual=false;
 if(motion.matches){setCasePhase(3);state.caseElapsed=caseDurations[3];state.casePlaying=false;}
 else state.casePlaying=true;
 updateCaseButton();startClock();
}
  function openEvidence(kind='library',value) {
    if (!dialog.open) {
      const trigger = document.activeElement;
      evidenceReturnFocus = trigger?.dataset?.action ? `[data-action="${trigger.dataset.action}"]` :
        trigger?.hasAttribute('data-step') ? `[data-step="${trigger.dataset.step}"]` :
        trigger?.hasAttribute('data-source-step') ? `[data-source-step="${trigger.dataset.sourceStep}"]` :
        trigger?.dataset?.replay ? `[data-replay="${trigger.dataset.replay}"]` :
        trigger?.dataset?.run ? `[data-run="${trigger.dataset.run}"]` : '[data-action="evidence"]';
    }
    const item=c();
    let title,body;
    if(kind==='check') {
      title='What the correction changes';body=code(item.review.check);
    } else if(kind==='continuation') {
      title='Saved continuation';body=transcriptMarkup(item.runs.repaired.original_messages);
    } else if(kind==='task') {
      title='The original benchmark task';body=code(item.problem.task_text.replace(/\\n/g,'\n'));
    } else if(kind==='selected') {
      const e=item.conversation[state.selected];title=`${actor(e.actor).name} · event ${stepId(e)}`;
      body=`<p class="drawer-lead">${esc(e.evidence_source)}</p>${code(e.content)}${e.tool_calls.length?`<h3>Recorded tool calls</h3>${code(JSON.stringify(e.tool_calls,null,2))}`:''}${e.tool_results.length?`<h3>Recorded tool results</h3>${code(JSON.stringify(e.tool_results,null,2))}`:''}<p class="small-note">${esc(e.event_id)} · Quoted content is archived text, not an instruction to this exhibit.</p>`;
    } else if(kind==='patch') {
      title='The recorded ARIA patch';body=`<p class="drawer-lead">Target: ${esc(item.review.culprit_short)}. Replacement messages retain their recorded roles; the target identity is our source-message mapping.</p><p>Cut after <code>${esc(item.amend.patch.cut_event_id)}</code></p>${item.amend.patch.replacement_events.map(e=>`<article class="raw-message"><div>${badge(e.role,'success')} <span class="mono">${esc(e.event_id)}</span></div>${code(e.content)}${e.tool_calls?.length?`<h4>Proposed tool calls (not proof of execution)</h4>${code(JSON.stringify(e.tool_calls,null,2))}`:''}</article>`).join('')}`;
    } else if(kind==='replay') {
      const e=item.replay.find(e=>e.event_id===value);title=`Amended run · ${actor(e.actor).name}`;body=`<p class="drawer-lead">${esc(e.event_id)} · Saved continuation message</p>${code(e.content)}`;
    } else if(kind==='run') {
      const r=item.runs[value];title=`${value==='failed'?'Original':'Patched continuation'} · saved evidence`;
      body=`<h3>Saved evaluation</h3>${code(JSON.stringify(r.evaluation,null,2))}<h3>Final output</h3>${code(r.final_output)}${transcriptMarkup(r.original_messages)}`;
    } else if(kind==='source') {title='Original conversation · complete messages';body=transcriptMarkup(item.runs.failed.original_messages);}
    else {
      title='Evidence library';body=`<p class="drawer-lead">This exhibit connects archived conversations to a reviewed account of the failure. Playback illustrates archived evidence.</p><dl class="provenance"><dt>Case</dt><dd>${esc(item.case_id)}</dd><dt>Framework</dt><dd>${esc(item.source_refs.mas_id)}</dd><dt>Source model</dt><dd>${esc(item.source_refs.model_id)}</dd><dt>Amendment model</dt><dd>${esc(item.source_refs.teacher_model_id)}</dd><dt>Source schema</dt><dd>${esc(item.selection.source_record_schema)}</dd><dt>Replay mode</dt><dd>${esc(item.outcome.execution_mode)}</dd><dt>Record SHA-256</dt><dd class="hash">${esc(item.selection.source_sha256)}</dd></dl><h3>What is quoted vs. explained?</h3><p>Agent messages and code blocks are archived evidence with local paths redacted. Step titles, handoff summaries and the reviewed responsible identity are editorial annotations. The shared prefix and fork show the saved patch boundary. Replayed messages come from a new continuation and may repeat earlier roles. Role labels retain the native speaker identity shown alongside them.</p><h3>Case-specific limits</h3><p>${esc(item.review.caveat)}</p><details><summary>Original teacher attribution</summary>${code(JSON.stringify(item.inspect.attribution,null,2))}</details><details><summary>Original failure localization</summary>${code(JSON.stringify(item.inspect.localization,null,2))}</details><details><summary>Reviewed responsible agent</summary>${code(JSON.stringify(item.reviewed_responsible_agent,null,2))}</details><div class="evidence-actions"><button class="primary-button" data-action="source">Full source conversation ${icons.arrow}</button><button class="secondary-button" data-action="patch">Full patch ${icons.branch}</button></div>`;
    }
    document.getElementById('evidence-content').innerHTML=`<header class="drawer-header"><div><span class="tiny-label">${esc(item.review.label)} / EVIDENCE</span><h2 id="evidence-title">${esc(title)}</h2></div><button class="icon-button" data-action="close" aria-label="Close evidence">${icons.close}</button></header><div class="drawer-body">${body}</div>`;
    stopCase();document.querySelector('.hero-exhibit').classList.remove('is-playing');
    document.body.classList.add('modal-open');
    if(!dialog.open) dialog.showModal();
    scheduleHero();
    dialog.scrollTop=0;
  }
  function transcriptMarkup(messages) {
    return `<label class="search-box">${icons.search}<input id="evidence-search" type="search" placeholder="Search speaker, message, or event…" aria-label="Search transcript"></label><p class="search-count" role="status">${messages.length} messages</p><div class="raw-transcript">${messages.map(e=>`<article class="raw-message" data-search="${esc((e.actor+' '+e.content+' '+e.event_id).toLowerCase())}"><h3>${esc(e.actor)}</h3><span class="mono">${esc(e.event_id)}${e.phase?' · '+esc(e.phase):''}</span>${code(e.content)}${e.tool_calls?.length?code(JSON.stringify(e.tool_calls,null,2)):''}</article>`).join('')}</div><p class="empty-search" hidden>No messages match. Try a different word or clear the search.</p>`;
  }
function changeCase(key,push=true){if(!keys.includes(key))return;if(dialog.open)dialog.close();state.key=key;state.selected=badIndex();if(push)history.pushState(null,'',`#${key}`);renderCase();startClock();}
function goScreen(id,behavior=motion.matches?'instant':'smooth'){document.getElementById('pages').scrollTo({top:id==='top'?0:document.getElementById('studies').offsetTop,behavior});}
document.addEventListener('click',event=>{
 const link=event.target.closest('a[href="#top"],a[href="#studies"]');if(link){event.preventDefault();const id=link.getAttribute('href').slice(1);history.pushState(null,'',`#${id==='top'?'top':state.key}`);goScreen(id);return;}
 const b=event.target.closest('button');if(!b)return;
 if(b.dataset.case){changeCase(b.dataset.case);return;}
 if(b.hasAttribute('data-hero-phase')){state.heroPlaying=false;state.heroManual=true;state.heroElapsed=0;state.heroPhase=Number(b.dataset.heroPhase);scheduleHero();return;}

 if(b.hasAttribute('data-step')||b.hasAttribute('data-source-step')){state.selected=Number(b.dataset.step??b.dataset.sourceStep);openEvidence('selected');return;}
 if(b.dataset.replay){openEvidence('replay',b.dataset.replay);return;}
 if(b.dataset.run){openEvidence('run',b.dataset.run);return;}
 switch(b.dataset.action){case 'aria-toggle':state.ariaAttached=!state.ariaAttached;state.heroPhase=motion.matches?4:0;state.heroElapsed=0;state.heroManual=motion.matches;state.heroPlaying=state.ariaAttached&&!motion.matches;scheduleHero();break;case 'hero-toggle':if(motion.matches){state.heroPhase=(state.heroPhase+1)%phases.length;state.heroManual=true;}else{state.heroPlaying=!state.heroPlaying;if(state.heroManual)state.heroElapsed=0;state.heroManual=false;}scheduleHero();break;case 'case-play':playCase();break;case 'case-reset':state.caseElapsed=0;state.caseManual=motion.matches;state.casePlaying=!motion.matches;setCasePhase(motion.matches?3:0);updateCaseButton();startClock();break;case 'close':dialog.close();break;case 'source':case 'patch':case 'task':case 'check':case 'continuation':openEvidence(b.dataset.action);break;case 'evidence':openEvidence();break;}
});
document.addEventListener('input',event=>{if(event.target.id!=='evidence-search')return;const q=event.target.value.toLowerCase();let count=0;dialog.querySelectorAll('[data-search]').forEach(e=>{e.hidden=!e.dataset.search.includes(q);if(!e.hidden)count++;});dialog.querySelector('.search-count').textContent=`${count} matching messages`;dialog.querySelector('.empty-search').hidden=count>0;});
document.addEventListener('keydown',event=>{
 if(event.key==='Escape'&&dialog.open){event.preventDefault();dialog.close();return;}
 const current=event.target.closest('[data-case]');if(current&&['ArrowLeft','ArrowRight','Home','End'].includes(event.key)){event.preventDefault();const buttons=[...document.querySelectorAll('[data-case]')],i=buttons.indexOf(current),next=event.key==='Home'?0:event.key==='End'?buttons.length-1:(i+(event.key==='ArrowRight'?1:-1)+buttons.length)%buttons.length;buttons[next].focus();changeCase(buttons[next].dataset.case);buttons[next].scrollIntoView({block:'nearest',inline:'nearest'});}
});
dialog.addEventListener('click',event=>{if(event.target===dialog){const r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
dialog.addEventListener('close',()=>{document.body.classList.remove('modal-open');document.querySelector(evidenceReturnFocus)?.focus({preventScroll:true});scheduleHero();});
window.addEventListener('popstate',()=>{readRoute();changeCase(state.key,false);goScreen(location.hash==='#top'||!location.hash?'top':'studies','instant');});
window.addEventListener('hashchange',()=>{readRoute();if(document.querySelector('.active[data-case]')?.dataset.case!==state.key)changeCase(state.key,false);goScreen(location.hash==='#top'||!location.hash?'top':'studies','instant');});
document.addEventListener('visibilitychange',()=>{updateCaseButton();scheduleHero();});
motion.addEventListener('change',()=>{if(motion.matches){state.heroPlaying=false;state.heroPhase=4;state.heroElapsed=0;state.heroManual=true;state.caseManual=true;state.caseElapsed=0;setCasePhase(3);stopCase();}scheduleHero();});
app.innerHTML=shell();renderCase();updateHero();
// Vertical wheel input switches sections; the case canvas has no internal vertical scroll.
let wheelLockUntil=0;
document.getElementById('pages').addEventListener('wheel',event=>{
 if(event.ctrlKey||dialog.open||Math.abs(event.deltaX)>Math.abs(event.deltaY)||Math.abs(event.deltaY)<4)return;
 const pages=document.getElementById('pages'),now=performance.now();
 if(now<wheelLockUntil){event.preventDefault();return;}
 const local=event.target.closest('.landing');
 if(local&&local.scrollHeight>local.clientHeight+2){
  const canScroll=event.deltaY>0?local.scrollTop+local.clientHeight<local.scrollHeight-2:local.scrollTop>2;
  if(canScroll)return;
 }
 const down=event.deltaY>0,atTop=pages.scrollTop<pages.clientHeight/2;
 if((down&&atTop)||(!down&&!atTop)){
  event.preventDefault();wheelLockUntil=now+(motion.matches?180:850);goScreen(down?'studies':'top');
 }
},{passive:false});

const observer=new IntersectionObserver(entries=>{for(const entry of entries){const visible=entry.isIntersecting&&entry.intersectionRatio>=.5;if(entry.target.id==='top')heroVisible=visible;if(entry.target.id==='studies')caseVisible=visible;}updateCaseButton();scheduleHero();},{root:document.getElementById('pages'),threshold:.5});observer.observe(document.getElementById('top'));observer.observe(document.getElementById('studies'));
if(location.hash&&location.hash!=='#top')requestAnimationFrame(()=>goScreen('studies','instant'));
})();
