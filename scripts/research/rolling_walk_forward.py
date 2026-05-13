#!/usr/bin/env python3
"""Rolling walk-forward research entrypoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.research.rolling import RollingWalkForwardConfig, run_rolling_walk_forward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-window rolling walk-forward backtest")
    parser.add_argument("--config", type=Path, required=True, help="JSON config for rolling walk-forward")
    parser.add_argument("--dry-run", action="store_true", help="Only generate and print rolling splits; do not train/backtest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = RollingWalkForwardConfig.from_json(args.config)
    result = run_rolling_walk_forward(config, dry_run=args.dry_run)
    logger = logging.getLogger("qlib_tw.research.rolling")
    logger.info("Output root: %s", result["output_root"])
    if args.dry_run:
        for split in result["splits"]:
            logger.info(
                "Round %d: train %s~%s, valid %s~%s, trade %s~%s",
                split.round_id,
                split.train_start.strftime("%Y-%m-%d"),
                split.train_end.strftime("%Y-%m-%d"),
                split.valid_start.strftime("%Y-%m-%d"),
                split.valid_end.strftime("%Y-%m-%d"),
                split.trade_start.strftime("%Y-%m-%d"),
                split.trade_end.strftime("%Y-%m-%d"),
            )
    else:
        logger.info(
            "Backtest recorder: %s/%s",
            result["backtest_experiment"],
            result["backtest_recorder_id"],
        )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    raise SystemExit(main())
