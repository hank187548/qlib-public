#!/usr/bin/env python3
"""Backtest-only pipeline using trained models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from research.runner import backtest_combo, init_qlib
from research.settings import COMBO_CONFIGS, combo_choices, resolve_combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest with trained models")
    parser.add_argument("--combo", choices=combo_choices(), nargs="+", help="Combos to backtest (default alpha158_lgb). Use all to run all combos.")
    parser.add_argument("--n-drop", type=int, default=None, help="Override TopkDropout n_drop")
    parser.add_argument("--topk", type=int, default=None, help="Override TopkDropout topk")
    parser.add_argument("--rebalance", choices=["day", "week"], default="day", help="Backtest rebalance frequency")
    parser.add_argument("--strategy", choices=["bucket", "equal"], default="bucket", help="bucket weighting / equal weighting")
    parser.add_argument("--deal-price", choices=["close", "open"], default="close", help="Execution price assumption for backtest")
    parser.add_argument("--limit-tplus", action="store_true", help="Enable next-open limit order + T+2 settlement")
    parser.add_argument("--recorder-id", default=None, help="Specify recorder id from training experiment; if omitted, use the latest recorder.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    combos = resolve_combos(args.combo)

    init_qlib()
    for combo_name in combos:
        spec = COMBO_CONFIGS[combo_name]
        logging.info("=== Backtest combo: %s (handler=%s, model=%s) ===", combo_name, spec["handler"], spec["model"])
        backtest_combo(
            combo_name,
            spec["handler"],
            spec["model"],
            spec.get("max_instruments"),
            spec.get("infer_processors"),
            n_drop_override=args.n_drop,
            topk_override=args.topk,
            rebalance=args.rebalance,
            strategy_choice=args.strategy,
            deal_price=args.deal_price,
            limit_tplus=args.limit_tplus,
            recorder_override=args.recorder_id,
        )


if __name__ == "__main__":
    main()
