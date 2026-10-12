"""Config-only entrypoint for MASFM jobs."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from masfm.config import load_config
from masfm.dispatcher import run_from_config


def main() -> None:
    load_repo_dotenv()
    config_path = parse_config_path(sys.argv[1:])
    config = load_config(config_path)
    run_from_config(config)


def parse_config_path(argv: list[str]) -> Path:
    if len(argv) == 2 and argv[0] in {"-c", "--config"}:
        return Path(argv[1])
    raise SystemExit("usage: python main.py -c config.yml")


def load_repo_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)


if __name__ == "__main__":
    main()
