#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from paper_trading.config import PaperTradingProfile
from paper_trading.replay import run_paper_trading_cycle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper trading from the configured start date with isolated outputs")
    parser.add_argument("--config", type=Path, required=True, help="Paper trading profile JSON")
    parser.add_argument("--target-date", default=None, help="Replay end date (YYYY-MM-DD). Default: today")
    parser.add_argument(
        "--refresh-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refresh Data/tw_data before replaying. Default: refresh; pass --no-refresh-data to skip.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    profile = PaperTradingProfile.from_json(args.config)
    target_date = pd.Timestamp(args.target_date).normalize() if args.target_date else pd.Timestamp.today().normalize()
    outputs = run_paper_trading_cycle(profile, target_date=target_date, refresh_data=args.refresh_data)
    print("Paper trading outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
