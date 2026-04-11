#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.trade.execution_search import ExecutionGridConfig, run_execution_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run execution-parameter grid search on a fixed trained model")
    parser.add_argument("--config", type=Path, required=True, help="JSON config for execution grid search")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = ExecutionGridConfig.from_json(args.config)
    result = run_execution_grid(config)
    logging.getLogger("qlib_tw.trade.execution_search").info(
        "Best variant %s => %.6f",
        result["best_result"]["variant_slug"],
        result["best_result"]["ranking_value"],
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    raise SystemExit(main())
