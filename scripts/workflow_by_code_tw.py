#!/usr/bin/env python3
"""End-to-end Taiwan Qlib workflow without notebooks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from tw_workflow.runner import init_qlib, run_combo
from tw_workflow.settings import COMBO_CONFIGS, combo_choices, resolve_combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taiwan Qlib workflow")
    parser.add_argument(
        "--combo",
        choices=combo_choices(),
        nargs="+",
        help="Specify one or more handler/model combos to run (default alpha158_lgb). Use 'all' to run every combo.",
    )
    parser.add_argument("--n-drop", type=int, default=None, help="Override TopkDropout n_drop (outputs under <combo>_ndrop<N>)")
    parser.add_argument("--topk", type=int, default=None, help="Override TopkDropout topk (outputs under <combo>_topk<K>)")
    parser.add_argument("--rebalance", choices=["day", "week"], default="day", help="Rebalance frequency for backtest/exchange")
    parser.add_argument("--strategy", choices=["bucket", "equal"], default="bucket", help="bucketed weights or equal-weight TopkDropout")
    parser.add_argument("--deal-price", choices=["close", "open"], default="close", help="Execution price assumption for backtest")
    parser.add_argument("--simulate-limit", action="store_true", help="Enable simplified limit-order simulation")
    parser.add_argument("--limit-slippage", type=float, default=0.01, help="Limit price offset relative to base_price")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    combos = resolve_combos(args.combo)

    init_qlib()
    for combo_name in combos:
        spec = COMBO_CONFIGS[combo_name]
        run_combo(
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
            simulate_limit=args.simulate_limit,
            limit_slippage=args.limit_slippage,
        )


if __name__ == "__main__":
    main()
