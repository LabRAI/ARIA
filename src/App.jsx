import { useEffect, useState } from "react";
import "./App.css";
const labels = {
  "gsm-plus": "GSM-Plus",
  mmlu: "MMLU",
  humaneval: "HumanEval",
  olympiadbench: "OlympiadBench",
  gaia: "GAIA",
  "swe-bench-lite": "SWE-Bench-Lite",
};
const domains = [
  "Mathematical reasoning",
  "Knowledge & reasoning",
  "Code generation",
  "Scientific reasoning",
  "Real-world assistance",
  "Software engineering",
];
const text = (v) =>
  typeof v === "string" ? v : JSON.stringify(v ?? {}, null, 2);
const url = (id) => `${import.meta.env.BASE_URL}data/${id}.json`;
function route() {
  const id =
    location.hash.slice(1) ||
    location.pathname.split("/").pop().replace(".html", "");
  return labels[id] ? id : "overview";
}
function Badge({ success }) {
  return (
    <span className={`badge ${success ? "pass" : "fail"}`}>
      {success ? "✓ Pass" : "× Fail"}
    </span>
  );
}
function Record({ title, value }) {
  return (
    <details className="record">
      <summary>{title}</summary>
      <pre>{text(value)}</pre>
    </details>
  );
}
function Overview({ manifest, select }) {
  const cases = manifest.cases;
  const max = Math.max(
    ...cases.flatMap((c) => [
      c.failed_display_events,
      c.repaired_display_events,
    ]),
  );
  return (
    <>
      <div className="eyebrow">THE REPAIR OBSERVATORY</div>
      <h1>
        Small interventions.
        <br />
        <span>Traceable outcomes.</span>
      </h1>
      <p className="intro">
        Explore where an agent went wrong, what ARIA changed, and what happened
        next. Six case studies, from reasoning to software engineering.
      </p>
      <div className="metrics">
        <div>
          <span>Selected case studies</span>
          <strong>0{cases.length}</strong>
          <small>Across six benchmarks</small>
        </div>
        <div>
          <span>Patched runs accepted</span>
          <strong>
            {cases.filter((c) => c.repaired_success).length}
            <em> / {cases.length}</em>
          </strong>
          <small>Saved evaluator results</small>
        </div>
        <div>
          <span>Controls accepted</span>
          <strong>
            {cases.filter((c) => c.control_success).length}
            <em> / {cases.length}</em>
          </strong>
          <small>Read the content-level caveats</small>
        </div>
      </div>
      <section className="panel overview-chart">
        <div className="section-heading">
          <div>
            <div className="eyebrow">AT A GLANCE</div>
            <h2>From source to continuation</h2>
          </div>
          <div className="legend">
            <i /> Source <i className="green" /> Patched
          </div>
        </div>
        <p className="muted">
          Recorded display events per run. Select a benchmark to inspect its
          repair.
        </p>
        <div className="chart-axis">
          <span>BENCHMARK</span>
          <span>EVENT COUNT · SHARED SCALE 0–{max}</span>
          <span>SAVED RESULT</span>
        </div>
        {cases.map((c) => (
          <button
            key={c.benchmark}
            className="chart-row"
            onClick={() => select(c.benchmark)}
          >
            <strong>{labels[c.benchmark]}</strong>
            <div className="bar-pair">
              {[
                ["failed", c.failed_display_events],
                ["repaired", c.repaired_display_events],
              ].map(([mode, count]) => (
                <div className="bar-track" key={mode}>
                  <div
                    className={`bar ${mode}`}
                    style={{ width: `${(count / max) * 86}%` }}
                  />
                  <span>{count}</span>
                </div>
              ))}
            </div>
            <span className="result-pair">
              <Badge success={c.failed_success} />
              <span>→</span>
              <Badge success={c.repaired_success} />
            </span>
          </button>
        ))}
        <p className="chart-note">
          These selected cases use archived non-exact continuations and do not
          estimate benchmark-wide repair effectiveness. Event counts describe
          trace length, not cost or quality.
        </p>
      </section>
      <div className="section-heading">
        <div>
          <div className="eyebrow">LOOK CLOSER</div>
          <h2>Every repair has a story</h2>
        </div>
        <span className="muted">06 case files</span>
      </div>
      <div className="case-grid">
        {cases.map((c, i) => (
          <button
            className="case-card"
            key={c.benchmark}
            onClick={() => select(c.benchmark)}
          >
            <div className="card-top">
              <span className="case-number">0{i + 1}</span>
              <span>{domains[i]}</span>
              <span className="arrow">↗</span>
            </div>
            <div className="eyebrow">{labels[c.benchmark]}</div>
            <h3>{c.title}</h3>
            <div className="card-bottom">
              Explore case study <span>→</span>
            </div>
          </button>
        ))}
      </div>
    </>
  );
}
function Trajectory({ study, mode }) {
  const run = study.runs[mode],
    events = run.display_trajectory.events;
  const [query, setQuery] = useState(""),
    [all, setAll] = useState(false),
    [expanded, setExpanded] = useState(false),
    [selected, setSelected] = useState(null);
  const bad = study.inspect.localization.first_bad_event_ids;
  const frontier = study.inspect.localization.retained_frontier_ids || [];
  const focus =
    mode === "failed"
      ? [...study.review.focus, ...bad]
      : mode === "repaired" && study.review.replay_focus
        ? study.review.replay_focus
        : events.slice(-2).map((e) => e.event_id);
  const visible = events.filter((e) =>
    query
      ? text(e).toLowerCase().includes(query.toLowerCase())
      : all || focus.includes(e.event_id) || selected === e.event_id,
  );
  return (
    <>
      <div className="timeline" aria-label="Recorded event sequence">
        {events.map((e, i) => (
          <button
            key={e.event_id}
            title={`${e.event_id} · ${e.display_role || e.role || e.kind}`}
            className={`${mode === "repaired" ? "repaired" : ""} ${mode === "failed" && bad.includes(e.event_id) ? "bad" : ""} ${selected === e.event_id ? "selected" : ""}`}
            onClick={() => {
              setSelected(e.event_id);
              setQuery("");
            }}
          >
            {i + 1}
          </button>
        ))}
      </div>
      <p className="muted tiny">
        Select an event to reveal its evidence. Coral marks the localized error;
        numbers show sequence position.
      </p>
      <div className="toolbar">
        <input
          type="search"
          aria-label="Search trajectory"
          placeholder="Search events, tool calls, or IDs…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <label>
          <input
            type="checkbox"
            checked={all}
            onChange={(e) => setAll(e.target.checked)}
          />{" "}
          All events
        </label>
        <button className="secondary" onClick={() => setExpanded(!expanded)}>
          {expanded ? "Collapse" : "Expand"} all
        </button>
      </div>
      <p className="muted tiny" aria-live="polite">
        {visible.length} of {events.length} events
      </p>
      <div className="events">
        {visible.map((e) => (
          <details
            key={`${e.event_id}-${expanded}-${selected}`}
            open={
              expanded ||
              selected === e.event_id ||
              (mode === "failed" && bad.includes(e.event_id))
            }
            className={`event ${mode === "failed" && bad.includes(e.event_id) ? "error-event" : ""}`}
          >
            <summary>
              <span className="mono">{e.event_id}</span>
              <span className="role">
                {e.display_role || e.mas_agent_role || e.role || "agent"} ·{" "}
                {e.kind}
              </span>
              {mode === "failed" && bad.includes(e.event_id) && (
                <span className="badge fail">First bad event</span>
              )}
              {mode === "failed" && frontier.includes(e.event_id) && (
                <span className="badge pass">Retained frontier</span>
              )}
            </summary>
            <div className="event-body">
              {e.content && <pre>{text(e.content)}</pre>}
              {e.tool_calls?.length > 0 && (
                <Record title="Tool calls" value={e.tool_calls} />
              )}
              {(e.tool_results || e.metadata?.tool_results)?.length > 0 && (
                <Record
                  title="Tool results"
                  value={e.tool_results || e.metadata.tool_results}
                />
              )}
              {(e.artifact || e.metadata?.artifact) && (
                <Record
                  title="Artifact"
                  value={e.artifact || e.metadata.artifact}
                />
              )}
              <p className="muted tiny">
                Evidence: {e.display_source || "Recorded trajectory"}
              </p>
              <Record title="Complete event record" value={e} />
            </div>
          </details>
        ))}
      </div>
      {!visible.length && (
        <p className="empty">No matching events. Try another search.</p>
      )}
      {run.display_trajectory.runtime_events?.length > 0 && (
        <Record
          title={`${run.display_trajectory.runtime_events.length} runtime records · launch, usage and completion`}
          value={run.display_trajectory.runtime_events}
        />
      )}
    </>
  );
}
function CaseStudy({ study }) {
  const [tab, setTab] = useState("repair"),
    [mode, setMode] = useState("failed");
  const r = study.review;
  const runs = [
    ["failed", "Original source"],
    ["control", "Unpatched control"],
    ["repaired", "Patched continuation"],
  ];
  return (
    <>
      <div className="eyebrow">
        {labels[study.benchmark]} <span className="separator">/</span>{" "}
        {study.selection.mas}
      </div>
      <h1 className="case-title">{r.title}</h1>
      <p className="intro">{r.subtitle}</p>
      <div className="case-meta">
        <span className="status-dot" />
        {r.status_label}
        <a className="download" href={url(study.benchmark)} download>
          Download case JSON ↓
        </a>
      </div>
      <div className="run-summary">
        {runs.map(([key, label]) => (
          <div key={key}>
            <span>{label}</span>
            <Badge success={study.runs[key].evaluation?.success} />
            <small>
              {study.runs[key].display_trajectory.events.length} display events
            </small>
          </div>
        ))}
      </div>
      <div className="tabs" aria-label="Case sections">
        {[
          ["repair", "01", "Repair story"],
          ["trajectory", "02", "Trajectory explorer"],
          ["verification", "03", "Verification"],
        ].map(([id, n, label]) => (
          <button
            key={id}
            aria-pressed={tab === id}
            onClick={() => setTab(id)}
            className={tab === id ? "active" : ""}
          >
            <span>{n}</span>
            {label}
          </button>
        ))}
      </div>
      {tab === "repair" && (
        <div className="story">
          <section className="panel">
            <div className="eyebrow">THE TASK</div>
            <h2>Start with the problem</h2>
            <pre className="problem">{study.problem.task_text}</pre>
          </section>
          <section className="panel">
            <div className="eyebrow">01 / INSPECT</div>
            <h2>Find the turning point</h2>
            <p>{r.inspect_summary}</p>
            <div className="chips">
              <span>{study.inspect.attribution.primary_code}</span>
              <span>{study.inspect.attribution.patch_operator}</span>
            </div>
            {study.inspect.localization.evidence?.map((e, i) => (
              <blockquote key={i}>
                <p>{e.quote}</p>
                <button
                  className="text-button"
                  onClick={() => {
                    setMode("failed");
                    setTab("trajectory");
                  }}
                >
                  {e.event_id} ↗
                </button>
              </blockquote>
            ))}
            <Record
              title="Recorded localization and attribution"
              value={study.inspect}
            />
          </section>
          <section className="panel">
            <div className="eyebrow">02 / AMEND</div>
            <h2>Change the faulty step</h2>
            <p>{r.amend_summary}</p>
            <div className="diff">
              <div>
                <span className="eyebrow">− BEFORE</span>
                <pre>{r.before}</pre>
              </div>
              <div>
                <span className="eyebrow">+ AFTER</span>
                <pre>{r.after}</pre>
              </div>
            </div>
            <p className="muted tiny">
              Retain through{" "}
              {study.amend.patch.cut_event_id || "start of trajectory"} · Insert{" "}
              {study.amend.patch.replacement_events.length} replacement event(s)
            </p>
            <Record title="Complete recorded patch" value={study.amend.patch} />
          </section>
          <section className="panel takeaway">
            <div className="eyebrow">03 / VERIFY</div>
            <h2>Read beyond the pass label</h2>
            <p>{r.verification_summary}</p>
            <pre>{r.check}</pre>
            <button
              className="secondary"
              onClick={() => setTab("verification")}
            >
              Inspect evaluations →
            </button>
          </section>
        </div>
      )}
      {tab === "trajectory" && (
        <section className="panel">
          <div className="section-heading">
            <div>
              <div className="eyebrow">FOLLOW THE EVIDENCE</div>
              <h2>Trajectory explorer</h2>
            </div>
            <select
              aria-label="Trajectory run"
              value={mode}
              onChange={(e) => setMode(e.target.value)}
            >
              {runs.map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <Trajectory key={mode} study={study} mode={mode} />
        </section>
      )}
      {tab === "verification" && (
        <div className="story">
          <section className="panel takeaway">
            <div className="eyebrow">CONTENT REVIEW</div>
            <h2>What the result actually shows</h2>
            <p>{r.verification_summary}</p>
            <pre>{r.check}</pre>
            <p>
              <strong>Control review. </strong>
              {r.control_note}
            </p>
          </section>
          <div className="evaluation-grid">
            {runs.map(([key, label]) => (
              <section className="panel" key={key}>
                <h3>{label}</h3>
                <Badge success={study.runs[key].evaluation?.success} />
                <p className="answer">
                  {text(
                    study.runs[key].evaluation?.predicted_answer ??
                      study.runs[key].evaluation?.reason ??
                      "See recorded output",
                  )}
                </p>
                <Record
                  title="Final output"
                  value={study.runs[key].final_output}
                />
                <Record
                  title="Saved evaluation"
                  value={study.runs[key].evaluation}
                />
              </section>
            ))}
          </div>
          <section className="panel">
            <h2>Evidence & provenance</h2>
            <p>
              The replay uses the archived non-exact continuation protocol. Its
              saved paired result describes this case and does not estimate a
              benchmark-wide effect.
            </p>
            {r.review_notes.map((note, i) => (
              <p key={i}>{note}</p>
            ))}
            <Record
              title="Case metadata and provenance"
              value={{
                case_id: study.case_id,
                selection: study.selection,
                generation: study.generation,
                outcome: study.outcome,
                quality: study.quality,
                source_refs: study.source_refs,
              }}
            />
          </section>
        </div>
      )}
    </>
  );
}
export default function App() {
  const [manifest, setManifest] = useState(null),
    [active, setActive] = useState(route),
    [study, setStudy] = useState(null),
    [error, setError] = useState(""),
    [retry, setRetry] = useState(0);
  useEffect(() => {
    const c = new AbortController();
    fetch(url("manifest"), { signal: c.signal })
      .then((r) => {
        if (!r.ok) throw Error("Unable to load case index");
        return r.json();
      })
      .then(setManifest)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => c.abort();
  }, [retry]);
  useEffect(() => {
    const listener = () => setActive(route());
    window.addEventListener("hashchange", listener);
    return () => window.removeEventListener("hashchange", listener);
  }, []);
  useEffect(() => {
    if (active === "overview") return;
    const c = new AbortController();
    fetch(url(active), { signal: c.signal })
      .then((r) => {
        if (!r.ok) throw Error("Unable to load case data");
        return r.json();
      })
      .then(setStudy)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => c.abort();
  }, [active, retry]);
  function select(id) {
    setError("");
    setActive(id);
    location.hash = id;
    window.scrollTo({ top: 0, behavior: "instant" });
  }
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          href="#overview"
          className="brand"
          onClick={() => select("overview")}
        >
          <span className="brand-mark">a</span>ARIA
          <span className="brand-label">RESEARCH</span>
        </a>
        <div className="workspace-label">CASE STUDY WORKSPACE</div>
        <button
          className={`nav-overview ${active === "overview" ? "active" : ""}`}
          onClick={() => select("overview")}
        >
          <span>▦</span> Overview <span>06</span>
        </button>
        <div className="nav-label">BENCHMARKS</div>
        <nav aria-label="Benchmarks">
          {Object.entries(labels).map(([id, label], i) => (
            <button
              key={id}
              className={active === id ? "active" : ""}
              onClick={() => select(id)}
              aria-current={active === id ? "page" : undefined}
            >
              <span className="nav-index">0{i + 1}</span>
              <span>{label}</span>
              <span className="nav-dot" />
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="mini-mark">↳</div>
          <strong>A closer look at agent repair.</strong>
          <p>
            Inspect the failure.
            <br />
            Amend the decision.
            <br />
            Verify the continuation.
          </p>
          <details>
            <summary>About this workspace</summary>
            <p>
              A React explorer for six archived ARIA repair case studies.
              Outcomes and evidence come from the supplied case data.
            </p>
          </details>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <span>
            Research <span className="separator">/</span> Case studies{" "}
            <span className="separator">/</span>{" "}
            <strong>
              {active === "overview" ? "Overview" : labels[active]}
            </strong>
          </span>
          <span className="archive-label">
            <span className="status-dot" /> ARCHIVED EVIDENCE
          </span>
        </header>
        <main>
          {error ? (
            <div className="panel" role="alert">
              <h2>Couldn’t load the data</h2>
              <p>{error}</p>
              <button
                className="secondary"
                onClick={() => {
                  setError("");
                  setRetry(retry + 1);
                }}
              >
                Try again
              </button>
            </div>
          ) : !manifest ||
            (active !== "overview" && study?.benchmark !== active) ? (
            <p className="loading" role="status">
              Loading case evidence…
            </p>
          ) : active === "overview" ? (
            <Overview manifest={manifest} select={select} />
          ) : (
            <CaseStudy key={active} study={study} />
          )}
        </main>
        <footer>
          <span>ARIA / REPAIR CASE STUDIES</span>
          <span>Evidence first. Every step inspectable.</span>
        </footer>
      </div>
    </div>
  );
}
