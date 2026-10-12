# ARIA case study website

The static website is copied from `MASFM/visualization/case_study_v5`. It contains
the animated runtime inspection and amendment concept and six archived case studies. The homepage
and six benchmark pages embed their styles, scripts, and case data, so they also
work offline. No Node.js dependencies or ARIA experiment environment are needed.

Both sections and the evidence dialog share a dark console design. The concept
demo starts with ARIA enabled: the outcome is a green check with ARIA and a red
cross without it. Reconnecting starts a new concept cycle.

Each benchmark automatically shows the agent trajectory, ARIA's Inspect
location, the Amend content, and the final green success check. Only the
benchmark tab displays progress. Pause and restart controls stay beside the
case title. The case canvas fits one viewport and has no internal vertical
scroll; full tasks and archived evidence open in a separate dialog. On narrow
screens, original and amended decisions appear in their respective phases.
Offscreen and background playback freeze without losing progress. Reduced-motion
mode shows the completed state without autoplay.

## Local preview

Open `index.html` directly, or run from the repository root:

```bash
python3 -m http.server 8000 --directory website
```

Visit `http://localhost:8000/`.

## Editing and building

Edit `index.template.html`, `app.js`, `styles.css`, or the archived projections in
`raw/`, then regenerate the checked-in pages:

```bash
python3 website/build_case_studies.py
```

Build the deployment artifact with:

```bash
python3 website/build_case_studies.py --output website/dist
```

The build uses only Python's standard library and the files in this directory.
It does not read MASFM archives or execute experiments. `dist/` contains only the
seven standalone pages and `.nojekyll`. Relative links work under the GitHub
Pages repository path without a base URL setting.

## GitHub Pages

1. In the GitHub repository, open **Settings → Pages** and select **GitHub Actions**
   as the source under **Build and deployment**.
2. Push these changes to `main`. The workflow in
   [`.github/workflows/deploy-pages.yml`](../.github/workflows/deploy-pages.yml)
   builds and deploys the website. Later pushes affecting `website/` or the
   workflow redeploy it automatically. Pull requests build without publishing.
3. To redeploy manually, open **Actions → Deploy ARIA website to GitHub Pages →
   Run workflow** and select `main`.

The website URL is `https://labrai.github.io/ARIA/`, matching the repository
`LabRAI/ARIA`. The deployment environment also shows this URL after deployment.

The concept animation and amendment playback illustrate saved evidence. They do
not invoke models or rerun experiments. Selected amendments do not estimate an
aggregate success rate. See `DESIGN.md` for the current design rationale.
