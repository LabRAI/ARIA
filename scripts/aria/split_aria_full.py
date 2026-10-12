#!/usr/bin/env python3
"""Split the aggregate ARIA-Full release deterministically."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aria.corpus import DEFAULT_SPLIT_PROPORTIONS, split_aria_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=REPO_ROOT / "data" / "ARIA-Full" / "training_ready",
        help="Aggregate ARIA-Full training_ready directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "ARIA-Full-Processed",
        help="Directory receiving train/val/test views and STATS.md.",
    )
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=DEFAULT_SPLIT_PROPORTIONS[0],
        help="Train proportion (default: 0.60).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=DEFAULT_SPLIT_PROPORTIONS[1],
        help="Validation proportion (default: 0.20).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=DEFAULT_SPLIT_PROPORTIONS[2],
        help="Test proportion (default: 0.20).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = split_aria_full(
        args.source,
        args.output,
        seed=args.split_seed,
        proportions=(args.train_ratio, args.val_ratio, args.test_ratio),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
