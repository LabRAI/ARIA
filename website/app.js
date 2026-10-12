(() => {
'use strict';
const DATA=JSON.parse(document.getElementById('case-data').textContent), app=document.getElementById('app'), dialog=document.getElementById('evidence-dialog');
const keys=Object.keys(DATA.cases),motion=matchMedia('(prefers-reduced-motion: reduce)');
const state={key:DATA.default,selected:0,heroPhase:motion.matches?5:0,heroPlaying:!motion.matches,casePhase:1,casePlaying:false,caseStarted:false};
let heroTimer,caseTimer,heroVisible=true,evidenceReturnFocus='[data-action="evidence"]';
const icons={arrow:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-6-6 6 6-6 6"/></svg>',check:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m5 12 4 4L19 6"/></svg>',close:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg>',search:'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="10" cy="10" r="6"/><path d="m15 15 5 5"/></svg>',branch:'<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="6" cy="5" r="2"/><circle cx="6" cy="19" r="2"/><circle cx="18" cy="5" r="2"/><path d="M6 7v10m12-10c0 5-12 2-12 8"/></svg>',file:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 3H5v18h14V8ZM14 3v5h5M8 12h8m-8 4h6"/></svg>',play:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m8 5 11 7-11 7Z"/></svg>',pause:'<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14M16 5v14"/></svg>',chip:'<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="3"/><path d="M9 2v4m6-4v4M9 18v4m6-4v4M2 9h4m-4 6h4m12-6h4m-4 6h4M10 10h4v4h-4Z"/></svg>'};
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
const phases=['Attach','Run','Detect','Inspect','Amend','Resume'];
const captions=['Deploy ARIA onto the team.','The agents work. ARIA watches.','Agent 03 takes a wrong turn.','Pinpoint the faulty decision.','Replace it. Keep the good work.','The repaired trajectory continues.'];
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
<header class="topbar"><a class="brand" href="#top"><span class="brand-icon">${icons.chip}</span>ARIA<span class="brand-sub">Runtime Inspect & Amend</span></a><a class="nav-jump" href="#studies">Explore the cases ${icons.arrow}</a></header>
<div class="landing-content"><div class="hero-copy"><div class="eyebrow">A REPAIR LAYER FOR MULTI-AGENT SYSTEMS</div><h1 id="hero-title">Find the fault.<br><em>Fix the course.</em></h1><p>Deploy ARIA onto your agent team. <br>Inspect and amend its trajectory <br>while the work unfolds.</p><a class="primary-button" href="#studies">See what changes ${icons.arrow}</a><span class="hero-note">Keep the context. Replace the mistake.</span></div>
<div class="hero-exhibit" data-phase="${state.heroPhase}">
<div class="machine" role="img" aria-label="Concept animation: ARIA attaches to a multi-agent system, locates a faulty decision in Agent 03, replaces it with a corrected decision, and resumes the trajectory.">
<i class="screw screw-tl" aria-hidden="true"></i><i class="screw screw-tr" aria-hidden="true"></i><i class="screw screw-bl" aria-hidden="true"></i><i class="screw screw-br" aria-hidden="true"></i>
<div class="machine-top"><span class="engraved">MULTI-AGENT SYSTEM</span><span class="system-status"><i class="led"></i><span id="system-status">Connecting ARIA</span></span></div>
<div class="machine-stage"><div class="aria-cartridge"><div class="chip-pins" aria-hidden="true"></div><div class="chip-face">${icons.chip}<strong>ARIA</strong><span class="chip-state">RUNTIME REPAIR</span><i class="led"></i></div></div>
<svg class="hardware-wires" viewBox="0 0 720 360" preserveAspectRatio="none" aria-hidden="true"><path class="cable cable-shadow" d="M360 67V98Q360 111 380 111H454Q474 111 474 132V168"/><path class="cable" d="M360 67V98Q360 111 380 111H454Q474 111 474 132V168"/><path class="connection-flow" d="M360 67V98Q360 111 380 111H454Q474 111 474 132V168"/><path class="track-shadow" d="M75 196H642"/><path class="track-line" d="M75 196H642"/><path class="track-pulse pulse-in" d="M75 196H470"/><path class="track-pulse pulse-out" d="M470 196H642"/><path class="probe-line" d="M470 224V278Q470 290 452 290H360V307"/></svg>
<div class="runtime-agents">${['Plan','Build','Review','Deliver'].map((name,i)=>`<div class="agent-module agent-${i}"><span class="socket" aria-hidden="true"></span><div class="agent-tile"><span class="agent-topline"><span>AGENT 0${i+1}</span><i class="module-led"></i></span><span class="agent-glyph" aria-hidden="true">${[icons.branch,icons.file,icons.search,icons.check][i]}</span><strong>${name}</strong></div>${i===2?'<span class="fault-flag">! FAULT HERE</span><div class="scanner" aria-hidden="true"></div>':''}</div>`).join('')}</div>
<div class="decision-slot"><span class="slot-label">AGENT 03 · DECISION</span><div class="decision-carriage"><div class="old-cartridge"><span>×</span>Violates the task constraint</div><div class="new-cartridge"><span>✓</span>Respects the task constraint</div></div><span class="slot-status" id="slot-status">Awaiting inspection</span></div>
</div><div class="machine-bottom"><span><i class="status-notch"></i> RETAINED CONTEXT</span><div class="mini-trace" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><span id="machine-outcome">TRAJECTORY READY</span></div></div>
<div class="demo-caption"><span class="phase-counter" id="phase-counter">01</span><h2 id="hero-caption">Deploy ARIA onto the team.</h2><button class="round-button" data-action="hero-toggle" aria-label="Pause animation">${icons.pause}</button></div>
<div class="demo-controls" role="group" aria-label="Concept animation phases">${phases.map((name,i)=>`<button data-hero-phase="${i}" aria-pressed="false"><span>0${i+1}</span>${name}</button>`).join('')}</div>
</div></div><footer class="screen-footer"><span>Concept animation</span><a href="#studies"><span class="scroll-wheel" aria-hidden="true"></span>Scroll to the real cases<span>↓</span></a><span>01 / 02</span></footer></section>
<section class="screen studies" id="studies" aria-labelledby="studies-title"><header class="case-topbar"><a class="brand" href="#top"><span class="brand-icon">${icons.chip}</span>ARIA</a><h2 id="studies-title">Where it went wrong. <em>What ARIA changed.</em></h2><a class="back-top" href="#top" aria-label="Back to concept animation">↑</a></header>
<nav class="benchmark-nav" aria-label="Benchmarks">${DATA.manifest.cases.map((entry,i)=>keys.includes(entry.benchmark)?`<button class="case-link" data-case="${entry.benchmark}">${esc(entry.label)}</button>`:`<a class="case-link" href="${entry.benchmark}.html#${entry.benchmark}">${esc(entry.label)}</a>`).join('')}</nav>
<div id="case-content" class="case-scroll" tabindex="0" aria-label="Selected case study"></div><footer class="screen-footer case-screen-footer"><span>Archived trajectories · Illustrative playback</span><button class="text-button" data-action="evidence">Evidence & limitations ↗</button><span>02 / 02</span></footer></section></main>`;}
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
 document.title=`ARIA / ${n.label} — Find the fault. Fix the course.`;
 document.querySelectorAll('[data-case]').forEach(b=>{b.classList.toggle('active',b.dataset.case===state.key);b.setAttribute('aria-current',b.dataset.case===state.key?'page':'false');});
 clearInterval(caseTimer);state.casePhase=1;state.casePlaying=false;state.caseStarted=false;
 document.getElementById('case-content').innerHTML=`<article class="case-study" aria-labelledby="case-title"><div class="case-heading"><div><span class="eyebrow">${esc(item.source_refs.mas_id==='metagpt'?'MetaGPT':'AG2')} / ${esc(n.label)}</span><h3 id="case-title">${esc(n.title)}</h3></div><button class="play-button" data-action="case-play">${icons.play} Replay repair</button></div>
<div class="task-line"><span class="label">TASK</span><p>${esc(n.problem_summary)}</p><button class="text-button" data-action="task">Full task ↗</button></div>
<div class="case-console" data-phase="1"><div class="trace-heading"><span class="engraved">ORIGINAL TRAJECTORY</span><span class="trace-state" id="trace-state">Error localized · Event ${esc(stepId(bad))}</span></div>
<div class="trajectory-rail" tabindex="0" aria-label="Original events in recorded order"><div class="trajectory-track"><div class="task-node"><span class="node-cap">INPUT</span>${icons.file}<strong>Task</strong></div>${item.conversation.map((e,i)=>`<button class="trace-node ${e.first_bad?'fault-node':i>badIndex()?'downstream-node':'retained-node'}" data-step="${i}" data-event="${esc(e.event_id)}" ${e.first_bad?'aria-label="Error starts here: '+esc(actor(e.actor).name)+' at event '+esc(stepId(e))+'"':''}><span class="node-cap">${e.first_bad?'! ERROR STARTS HERE':'EVENT '+esc(stepId(e))}</span><span class="node-body"><span class="agent-medallion">${e.first_bad?'!':esc(actor(e.actor).initials||'·')}</span><strong>${esc(actor(e.actor).name)}</strong></span></button>`).join('')}<div class="failed-end"><span>×</span>Failed</div></div></div>
<div class="repair-bench"><div class="fault-location"><span class="fault-target" aria-hidden="true">!</span><div><span class="label">THE FAULT · EVENT ${esc(stepId(bad))}</span><h4>${esc(actor(bad.actor).name)}</h4><p>${esc(synopsis[state.key])}</p><button class="text-button" data-source-step="${badIndex()}">Read faulty message ↗</button></div></div>
<div class="replacement"><div class="replacement-head"><span class="mini-chip">${icons.chip}ARIA</span><strong>Replace this decision</strong><button class="text-button" data-action="patch">Full patch ↗</button></div><div class="diff-pair"><div class="diff-side before"><div class="diff-label"><span>−</span> Wrong decision</div><pre class="code-face ${n.before.length>120?'long-code':''}"><code>${before}</code></pre></div><div class="swap-arrow" aria-hidden="true">${icons.arrow}</div><div class="diff-side after"><div class="diff-label"><span>+</span> Amended decision</div><pre class="code-face ${n.after.length>120?'long-code':''}"><code>${after}</code></pre></div></div></div></div>
<div class="outcome-rail"><div class="retained-note">${icons.branch}<span>Keep context through <strong>${esc(item.amend.patch.cut_event_id.split(':').pop())}</strong></span></div><div class="outcome failed"><span class="label">ORIGINAL</span><strong>${esc(n.result_values.failed)}</strong><button data-run="failed" aria-label="Original saved evaluation">↗</button></div><span class="outcome-arrow" aria-hidden="true">→</span><div class="outcome passed"><span class="label">SAVED CONTINUATION · PASSED</span><strong>${esc(n.result_values.repaired)}</strong><button data-run="repaired" aria-label="Repaired saved evaluation">↗</button></div></div>
<div class="case-playback"><div class="phase-buttons" role="group" aria-label="Case playback phases">${['Run','Fault','Amend','Resume'].map((label,i)=>`<button data-case-phase="${i}" aria-pressed="${i===1}"><span>0${i+1}</span>${label}</button>`).join('')}</div><p id="case-playback-caption">${esc(synopsis[state.key])}</p><button class="text-button" data-action="case-reset" aria-label="Reset case playback">↺</button></div></div>
<div class="case-evidence-links"><button class="text-button" data-action="source">Original conversation ↗</button><button class="text-button" data-action="continuation">Saved continuation ↗</button><button class="text-button" data-action="check">Behavior check ↗</button></div></article>`;
 centerRails();
 document.getElementById('case-content').scrollTop=0;
}
function centerRails(){
 const rail=document.querySelector('.trajectory-rail'),fault=document.querySelector('.fault-node');
 if(rail&&fault)rail.scrollLeft=fault.offsetLeft-rail.offsetLeft-(rail.clientWidth-fault.offsetWidth)/2;
 const nav=document.querySelector('.benchmark-nav'),active=nav.querySelector('.active');
 if(active)nav.scrollLeft=active.offsetLeft-nav.offsetLeft-(nav.clientWidth-active.offsetWidth)/2;
}
window.addEventListener('resize',()=>requestAnimationFrame(centerRails));
function updateHero(){
 const p=state.heroPhase,exhibit=document.querySelector('.hero-exhibit');exhibit.dataset.phase=p;
 exhibit.classList.toggle('is-playing',state.heroPlaying&&heroVisible&&!document.hidden&&!motion.matches&&!dialog.open);
 document.getElementById('hero-caption').textContent=captions[p];document.getElementById('phase-counter').textContent=`0${p+1}`;
 document.getElementById('system-status').textContent=['Connecting ARIA','System running','Execution at risk','Fault localized','Applying amendment','Execution resumed'][p];
 document.getElementById('slot-status').textContent=['Observer connected','Monitoring decisions','Constraint conflict','Fault isolated at Agent 03','Replacement applied','Repaired continuation'][p];
 document.getElementById('machine-outcome').textContent=['TRAJECTORY READY','EXECUTING','FAULT DETECTED','CONTEXT RETAINED','DECISION REPLACED','TRAJECTORY REPAIRED'][p];
 document.querySelectorAll('[data-hero-phase]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.heroPhase)===p)));
 const b=document.querySelector('[data-action="hero-toggle"]');b.innerHTML=state.heroPlaying?icons.pause:icons.play;b.setAttribute('aria-label',state.heroPlaying?'Pause animation':'Play animation');
}
function scheduleHero(){clearTimeout(heroTimer);updateHero();if(state.heroPlaying&&heroVisible&&!document.hidden&&!dialog.open)heroTimer=setTimeout(()=>{state.heroPhase=(state.heroPhase+1)%6;scheduleHero();},[1900,2200,1800,2400,2600,3200][state.heroPhase]);}
function setCasePhase(phase){state.casePhase=phase;document.querySelector('.case-console').dataset.phase=phase;document.querySelectorAll('[data-case-phase]').forEach(b=>b.setAttribute('aria-pressed',String(Number(b.dataset.casePhase)===phase)));document.getElementById('trace-state').textContent=['Reading the saved trajectory',`Error localized · Event ${stepId(c().conversation[badIndex()])}`,'Replacement applied at the retained frontier','Saved continuation · Evaluation passed'][phase];document.getElementById('case-playback-caption').textContent=[`Follow the recorded ${c().source_refs.mas_id==='metagpt'?'MetaGPT':'AG2'} trajectory.`,synopsis[state.key],c().review.repair_summary,`Saved repaired result: ${c().review.result_values.repaired}.`][phase];}
function updateCaseButton(){document.querySelector('[data-action="case-play"]').innerHTML=state.casePlaying?`${icons.pause} Pause repair`:`${icons.play} ${!state.caseStarted||state.casePhase===3?'Replay repair':'Resume repair'}`;}
function stopCase(){clearInterval(caseTimer);state.casePlaying=false;updateCaseButton();}
function playCase(){if(state.casePlaying){stopCase();return;}if(state.casePhase===3||!state.caseStarted)setCasePhase(0);state.caseStarted=true;state.casePlaying=true;updateCaseButton();caseTimer=setInterval(()=>{setCasePhase(state.casePhase+1);if(state.casePhase===3)stopCase();},2100);}
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
      const e=item.replay.find(e=>e.event_id===value);title=`Repaired run · ${actor(e.actor).name}`;body=`<p class="drawer-lead">${esc(e.event_id)} · Saved continuation message</p>${code(e.content)}`;
    } else if(kind==='run') {
      const r=item.runs[value];title=`${value==='failed'?'Original':'Patched continuation'} · saved evidence`;
      body=`<h3>Saved evaluation</h3>${code(JSON.stringify(r.evaluation,null,2))}<h3>Final output</h3>${code(r.final_output)}${transcriptMarkup(r.original_messages)}`;
    } else if(kind==='source') {title='Original conversation · complete messages';body=transcriptMarkup(item.runs.failed.original_messages);}
    else {
      title='Evidence library';body=`<p class="drawer-lead">This exhibit connects archived conversations to a reviewed account of the failure. Nothing here runs a model or changes an experiment.</p><dl class="provenance"><dt>Case</dt><dd>${esc(item.case_id)}</dd><dt>Framework</dt><dd>${esc(item.source_refs.mas_id)}</dd><dt>Source model</dt><dd>${esc(item.source_refs.model_id)}</dd><dt>Repair model</dt><dd>${esc(item.source_refs.teacher_model_id)}</dd><dt>Source schema</dt><dd>${esc(item.selection.source_record_schema)}</dd><dt>Replay mode</dt><dd>${esc(item.outcome.execution_mode)}</dd><dt>Record SHA-256</dt><dd class="hash">${esc(item.selection.source_sha256)}</dd></dl><h3>What is quoted vs. explained?</h3><p>Agent messages and code blocks are archived evidence with local paths redacted. Step titles, handoff summaries and the reviewed responsible identity are editorial annotations. The shared prefix and fork show the saved patch boundary. Replayed messages come from a new continuation and may repeat earlier roles. Role labels retain the native speaker identity shown alongside them.</p><h3>Case-specific limits</h3><p>${esc(item.review.caveat)}</p><details><summary>Original teacher attribution</summary>${code(JSON.stringify(item.inspect.attribution,null,2))}</details><details><summary>Original failure localization</summary>${code(JSON.stringify(item.inspect.localization,null,2))}</details><details><summary>Reviewed responsible agent</summary>${code(JSON.stringify(item.reviewed_responsible_agent,null,2))}</details><div class="evidence-actions"><button class="primary-button" data-action="source">Full source conversation ${icons.arrow}</button><button class="secondary-button" data-action="patch">Full patch ${icons.branch}</button></div>`;
    }
    document.getElementById('evidence-content').innerHTML=`<header class="drawer-header"><div><span class="tiny-label">${esc(item.review.label)} / EVIDENCE</span><h2 id="evidence-title">${esc(title)}</h2></div><button class="icon-button" data-action="close" aria-label="Close evidence">${icons.close}</button></header><div class="drawer-body">${body}</div>`;
    stopCase();clearTimeout(heroTimer);document.querySelector('.hero-exhibit').classList.remove('is-playing');
    document.body.classList.add('modal-open');
    if(!dialog.open) dialog.showModal();
    dialog.scrollTop=0;
  }
  function transcriptMarkup(messages) {
    return `<label class="search-box">${icons.search}<input id="evidence-search" type="search" placeholder="Search speaker, message, or event…" aria-label="Search transcript"></label><p class="search-count" role="status">${messages.length} messages</p><div class="raw-transcript">${messages.map(e=>`<article class="raw-message" data-search="${esc((e.actor+' '+e.content+' '+e.event_id).toLowerCase())}"><h3>${esc(e.actor)}</h3><span class="mono">${esc(e.event_id)}${e.phase?' · '+esc(e.phase):''}</span>${code(e.content)}${e.tool_calls?.length?code(JSON.stringify(e.tool_calls,null,2)):''}</article>`).join('')}</div><p class="empty-search" hidden>No messages match. Try a different word or clear the search.</p>`;
  }
function changeCase(key,push=true){if(!keys.includes(key))return;if(dialog.open)dialog.close();state.key=key;state.selected=badIndex();if(push)history.pushState(null,'',`#${key}`);renderCase();}
function goScreen(id,behavior=motion.matches?'instant':'smooth'){document.getElementById('pages').scrollTo({top:id==='top'?0:document.getElementById('studies').offsetTop,behavior});}
document.addEventListener('click',event=>{
 const link=event.target.closest('a[href="#top"],a[href="#studies"]');if(link){event.preventDefault();const id=link.getAttribute('href').slice(1);history.pushState(null,'',`#${id==='top'?'top':state.key}`);goScreen(id);return;}
 const b=event.target.closest('button');if(!b)return;
 if(b.dataset.case){changeCase(b.dataset.case);return;}
 if(b.hasAttribute('data-hero-phase')){state.heroPlaying=false;state.heroPhase=Number(b.dataset.heroPhase);scheduleHero();return;}
 if(b.hasAttribute('data-case-phase')){stopCase();state.caseStarted=true;setCasePhase(Number(b.dataset.casePhase));updateCaseButton();return;}
 if(b.hasAttribute('data-step')||b.hasAttribute('data-source-step')){state.selected=Number(b.dataset.step??b.dataset.sourceStep);openEvidence('selected');return;}
 if(b.dataset.replay){openEvidence('replay',b.dataset.replay);return;}
 if(b.dataset.run){openEvidence('run',b.dataset.run);return;}
 switch(b.dataset.action){case 'hero-toggle':state.heroPlaying=!state.heroPlaying;scheduleHero();break;case 'case-play':playCase();break;case 'case-reset':stopCase();state.caseStarted=false;setCasePhase(1);updateCaseButton();break;case 'close':dialog.close();break;case 'source':case 'patch':case 'task':case 'check':case 'continuation':openEvidence(b.dataset.action);break;case 'evidence':openEvidence();break;}
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
document.addEventListener('visibilitychange',()=>{scheduleHero();if(document.hidden)stopCase();});
motion.addEventListener('change',()=>{if(motion.matches){state.heroPlaying=false;state.heroPhase=5;stopCase();}scheduleHero();});
app.innerHTML=shell();renderCase();updateHero();
// One deliberate wheel gesture switches screens; inner case/evidence scrolling stays native.
let wheelLockUntil=0;
document.getElementById('pages').addEventListener('wheel',event=>{
 if(event.ctrlKey||dialog.open||Math.abs(event.deltaX)>Math.abs(event.deltaY)||Math.abs(event.deltaY)<4)return;
 const pages=document.getElementById('pages'),now=performance.now();
 if(now<wheelLockUntil){event.preventDefault();return;}
 const local=event.target.closest('.case-scroll,.landing');
 if(local&&local.scrollHeight>local.clientHeight+2){
  const canScroll=event.deltaY>0?local.scrollTop+local.clientHeight<local.scrollHeight-2:local.scrollTop>2;
  if(canScroll)return;
 }
 const down=event.deltaY>0,atTop=pages.scrollTop<pages.clientHeight/2;
 if((down&&atTop)||(!down&&!atTop)){
  event.preventDefault();wheelLockUntil=now+(motion.matches?180:850);goScreen(down?'studies':'top');
 }
},{passive:false});

const observer=new IntersectionObserver(entries=>{heroVisible=entries[0].isIntersecting;if(heroVisible)stopCase();scheduleHero();},{root:document.getElementById('pages'),threshold:.4});observer.observe(document.querySelector('.hero-exhibit'));
if(location.hash&&location.hash!=='#top')requestAnimationFrame(()=>goScreen('studies','instant'));
})();
