#!/usr/bin/env python3
"""Build the standalone V5 website from the included archived case projections."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def embedded_json(value):
    # Escape HTML script terminators in recorded code and web content.
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build(output):
    manifest = json.loads((ROOT / "raw/manifest.json").read_text(encoding="utf-8"))
    cases = {
        entry["benchmark"]: json.loads(
            (ROOT / "raw" / f"{entry['benchmark']}.json").read_text(encoding="utf-8")
        )
        for entry in manifest["cases"]
    }
    for case in cases.values():
        case["runs"].pop("control", None)
    template = (ROOT / "index.template.html").read_text(encoding="utf-8")
    template = template.replace(
        "/*__STYLES__*/", (ROOT / "styles.css").read_text(encoding="utf-8")
    ).replace("/*__APP__*/", (ROOT / "app.js").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    for key in ["index", *cases]:
        payload = {
            "manifest": manifest,
            "cases": cases if key == "index" else {key: cases[key]},
            "default": "humaneval" if key == "index" else key,
        }
        (output / f"{key}.html").write_text(
            template.replace("__DATA__", embedded_json(payload)), encoding="utf-8"
        )
    (output / ".nojekyll").touch()
    print(f"Built {len(cases)} cases and {len(cases) + 1} standalone HTML pages in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT,
        help="Output directory (default: website/; use website/dist for deployment).",
    )
    build(parser.parse_args().output.resolve())
