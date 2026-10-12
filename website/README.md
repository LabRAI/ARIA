# ARIA case study website

The static website is copied from `MASFM/visualization/case_study_v5`. It contains
the animated runtime repair concept and six archived case studies. The homepage
and six benchmark pages embed their styles, scripts, and case data, so they also
work offline. No Node.js dependencies or ARIA experiment environment are needed.

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

The concept animation and repair playback illustrate saved evidence. They do
not invoke models or rerun experiments. Selected repairs do not estimate an
aggregate success rate. See `DESIGN.md` for the original V5 design rationale.
