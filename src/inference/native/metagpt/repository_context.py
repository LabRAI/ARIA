"""Answer-blind source retrieval for explicitly selected MetaGPT host profiles."""

from __future__ import annotations

import ast
from collections import Counter
import keyword
import math
from pathlib import Path
import re
import subprocess

SUFFIXES = ("py", "js", "ts", "tsx", "java", "go", "rs")
STOP = {word.lower() for word in keyword.kwlist} | {
    "that", "this", "with", "from", "have", "when", "where", "which", "model", "return",
    "should", "does", "into", "using", "correct", "expected", "error", "test", "tests",
    "python", "class", "function", "value", "issue", "following", "print", "version",
}


def _tokens(task):
    return Counter(token for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", task) if token.lower() not in STOP)


def _imported_symbols(task):
    symbols = set()
    for line in task.splitlines():
        line = line.strip().removeprefix(">>>").strip()
        if not line.startswith("from "):
            continue
        try:
            tree = ast.parse(line)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                symbols.update(alias.name for alias in node.names if alias.name != "*")
    return symbols


def candidate_files(workspace: Path, task: str) -> list[Path]:
    workspace = workspace.resolve()
    tracked = subprocess.run(["git", "-C", str(workspace), "ls-files", "--", *[f"*.{ext}" for ext in SUFFIXES]],
                             capture_output=True, text=True, check=True, timeout=20).stdout.splitlines()
    tracked = {name for name in tracked if (workspace / name).resolve().is_relative_to(workspace)}
    scores = Counter()
    for name in re.findall(r"(?<![\w/])([\w./-]+\.(?:py|js|ts|tsx|java|go|rs))", task):
        if name in tracked:
            scores[name] += 100
    tokens = _tokens(task)
    imported = _imported_symbols(task)
    query_tokens = [*sorted(imported), *(token for token, _ in tokens.most_common() if token not in imported)]
    for token in query_tokens[:24]:
        result = subprocess.run(["git", "-C", str(workspace), "grep", "-l", "-F", "-w", token,
                                 "--", *[f"*.{ext}" for ext in SUFFIXES]],
                                capture_output=True, text=True, check=False, timeout=20)
        if result.returncode not in (0, 1):
            raise RuntimeError(f"repository search failed: {result.stderr[-1000:]}")
        matches = set(result.stdout.splitlines()) & tracked
        weight = math.log1p((len(tracked) + 1) / (len(matches) + 1)) * (4 if token in imported else 1)
        for name in matches:
            scores[name] += weight
    # Definition matches distinguish code owners from generic references.
    for name, _ in scores.most_common(80):
        if Path(name).suffix != ".py":
            continue
        try:
            tree = ast.parse((workspace / name).read_text())
        except (OSError, UnicodeError, SyntaxError):
            continue
        owners = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        scores[name] += 8 * len(owners & tokens.keys()) + 64 * len(owners & imported)
    ranked = sorted(scores or dict.fromkeys(tracked, 0), key=lambda name: (
        -scores[name], "test" in name.lower(), len(name), name))
    return [workspace / name for name in ranked[:10]]


def relevant_source_context(text: str, task: str, max_chars: int = 18_000) -> str:
    if len(text) <= max_chars:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[:max_chars]
    identifiers = _tokens(task)
    words = {token.lower() for token in identifiers}
    ranked = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        parts = {part.lower() for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)", node.name)}
        score = 100 * (node.name in identifiers) + 2 * len(parts & words)
        if score:
            ranked.append((score, node))
    if not ranked:
        return text[:max_chars]
    lines = text.splitlines(keepends=True)
    chunks, covered, remaining = [], [], max_chars
    for _, node in sorted(ranked, key=lambda item: (-item[0], item[1].end_lineno - item[1].lineno, item[1].lineno)):
        start, end = max(0, node.lineno - 4), min(len(lines), node.end_lineno + 3)
        if any(lo <= start and end <= hi for lo, hi in covered):
            continue
        heading = f"\n# Source lines {start + 1}-{end}\n"
        if remaining < len(heading) + 128:
            break
        chunk = (heading + "".join(lines[start:end]))[:remaining]
        chunks.append(chunk)
        covered.append((start, end))
        remaining -= len(chunk)
    return "\n".join(chunks)[:max_chars]
