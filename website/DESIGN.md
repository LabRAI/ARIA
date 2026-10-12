# V5: a repair instrument, in two screens

V4 made the story shorter but left the key act of intervention abstract. A changing border and a second result were insufficient: the viewer needs to see the fault's location and the exact replacement as one continuous action.

## Screen 1 — attach, detect, replace, resume

The hero uses a generic multi-agent system rather than a benchmark example. Four ceramic agent modules sit in a metal chassis on a shared execution rail. The purple ARIA cartridge connects to the system as a runtime repair layer. This is a conceptual representation, not a claim of universal plug-and-play deployment or a live model run.

Six phases have distinct physical behavior:

1. ARIA docks into its connection.
2. A packet moves along the original trajectory.
3. Agent 03 signals a fault through a red lamp, raised warning flag, and brief mechanical shake.
4. A scan frame isolates that node. Its faulty decision appears in an inset inspection slot.
5. The faulty decision slides out and the amended decision slides into that same slot. The node changes to the ARIA color.
6. Execution travels forward to the delivery agent, and the output signal resumes.

Beveled housings, inset slots, pin contacts, screws, directional lighting, and contact shadows give the apparatus a tangible structure. The artifact is made with CSS and inline SVG, preserving sharp text and offline operation. Motion is causal; the enclosure itself does not drift or rotate continuously.

## Screen 2 — location plus literal change

The horizontal benchmark selector stays outside the case's scroll panel. Each selected case opens directly at its fault; playback is optional.

The original trajectory preserves the recorded order and actors. The fault tile is labeled **ERROR STARTS HERE**, colored red, and automatically centered when the rail overflows. Clicking any event opens its archived source message. Later source events remain source events; the amended replay is separately accessible.

The workbench beneath the rail answers two questions together:

- **Where is the error?** The responsible role, original event ID, and one concise editorial diagnosis.
- **What changed?** The reviewed before/after content, with the changed token span highlighted. HumanEval explicitly emphasizes `!=` → `==`; MMLU emphasizes `B` → `C`. Token highlighting never changes the underlying text. These are concise reviewed explanations; the full recorded replacement remains linked.

An adjacent outcome strip presents original and saved repaired results, independently linked to their evaluations. The retained frontier is identified. Optional playback traces execution, emphasizes the fault, inserts the corrected decision, and highlights the saved continuation. It is an illustrative playback over archived evidence, not execution of a model or an exact restoration of agent internals.

## Two-screen navigation

The page contains exactly two top-level sections, each one viewport tall. A deliberate vertical wheel gesture moves between them. The case screen has native internal scrolling for long patches and small viewports; scrolling within content is not intercepted until a boundary. Trackpad horizontal gestures and Ctrl-wheel zoom are not intercepted. Navigation links, keyboard navigation, direct case links, and browser history remain available.

On short landscape screens, the hero can also scroll internally so content stays reachable. Reduced-motion users see the repaired conceptual state with autoplay disabled. Each phase is manually selectable; the evidence dialog pauses playback. Background and offscreen animation timers stop.

## Evidence contract

Raw projections, selections, and reviewed case notes remain byte-identical to V4. Additional one-sentence diagnoses in `app.js` summarize those notes; no source claims or evaluations are changed. Source messages, complete patches, original benchmark wording, saved continuations, behavior checks, source provenance, and qualifications remain in the evidence dialog. Previous versions remain untouched.
