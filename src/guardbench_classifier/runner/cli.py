"""``guardbench-run``: inspect runs. Reads files only; never calls a model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from guardbench_classifier.runner.status import read_status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="guardbench-run")
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="print status.json and whether its pid is alive")
    status.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)

    try:
        result = read_status(args.run_dir)
    except FileNotFoundError:
        print(f"no status.json in {args.run_dir}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
