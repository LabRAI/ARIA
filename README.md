<<<<<<< HEAD
# ARIA
=======
﻿# ARIA case study explorer

React + Vite recreation of the six supplied ARIA case studies.

- `npm install` installs the application dependencies.
- `npm run dev` starts the development server.
- `npm run build` creates the deployable `dist` directory.
- `npm run preview` serves the production build.
- `npm run lint` checks JavaScript and React code.

The overview compares recorded display-event counts on a shared scale. Each case includes the original problem, localization evidence, the complete patch, searchable source/control/repaired trajectories, runtime records, evaluator results, and review notes. Case navigation is saved in the URL hash (for example `/#mmlu`). JSON downloads retain the full archived evidence.

Data in `public/data` is copied unchanged from the supplied `case_study/raw` JSON files. The React app replaces the HTML pages and imperative DOM rendering; the original Python build pipeline is not required to run it. These are selected archived cases, not aggregate benchmark results. In particular, the GSM-Plus control has correct numeric reasoning despite its saved failing score.

Implementation: `src/App.jsx` contains the overview, case viewer, trajectory explorer, and reusable evidence components; `src/App.css` controls the responsive layout; `src/index.css` provides global defaults. Data loads per case to keep the initial JavaScript bundle small.

Run `npm test` to build and check navigation, all six cases, trajectory filtering, run switching, and verification in an in-memory DOM. Run `npm run format` to format source files. These checks do not replace visual browser testing.
>>>>>>> 7f6aea3 (initial commit of code)
