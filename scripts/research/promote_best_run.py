#!/usr/bin/env python3
"""Promote a backtest run under outputs/backtest/ into outputs/best_run/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.research.publish import DEST_ROOT, promote_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote one backtest run into outputs/best_run")
    parser.add_argument("--combo", required=True, help="Backtest folder name under outputs/backtest")
    parser.add_argument("--dest", type=Path, default=DEST_ROOT, help="Destination folder, default outputs/best_run")
    parser.add_argument("--clean", action="store_true", help="Remove destination reports/ and figures/ before copying")
    parser.add_argument("--no-translate-summary", action="store_true", help="Copy summary.txt as-is instead of translating known labels to English")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = promote_output(
        args.combo,
        dest=args.dest.resolve(),
        clean=args.clean,
        translate_summary_file=not args.no_translate_summary,
    )
    print(f"Promoted {args.combo} -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
