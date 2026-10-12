"""Cache gated GAIA metadata parquet files locally.

Run after accepting access to https://huggingface.co/datasets/gaia-benchmark/GAIA
and configuring HF_TOKEN or HUGGINGFACE_HUB_TOKEN in the environment or the
repository's ignored .env file.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from common.preparation import huggingface_token


REPO_ID = "gaia-benchmark/GAIA"
DEFAULT_SPLITS = ("validation", "test")
DEFAULT_CONFIGS = ("all",)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache gated GAIA parquet metadata under data/benchmarks/gaia.")
    parser.add_argument("--data-dir", default="data", help="Repo data directory.")
    parser.add_argument("--split", action="append", choices=("validation", "test"), help="Split to cache.")
    parser.add_argument(
        "--config",
        action="append",
        choices=("all", "level1", "level2", "level3"),
        help="GAIA config to cache. Defaults to all.",
    )
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    token = huggingface_token()
    if not token:
        raise SystemExit(
            "Missing HF_TOKEN or HUGGINGFACE_HUB_TOKEN. Accept access to "
            f"{REPO_ID} on Hugging Face, then export a read token."
        )

    data_dir = Path(args.data_dir)
    splits = tuple(args.split or DEFAULT_SPLITS)
    configs = tuple(args.config or DEFAULT_CONFIGS)
    root = data_dir / "benchmarks" / "gaia" / "2023"

    for split in splits:
        for config in configs:
            suffix = "" if config == "all" else f".{config}"
            filename = f"metadata{suffix}.parquet"
            remote_path = f"2023/{split}/{filename}"
            local_download = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=remote_path,
                    token=token,
                )
            )
            destination = root / split / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_download, destination)
            print(destination)


if __name__ == "__main__":
    main()
