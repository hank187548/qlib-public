#!/usr/bin/env python3
"""End-to-end Taiwan Qlib workflow without notebooks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.research.runner import backtest_combo, init_qlib, train_combo
from qlib_tw.research.search_results import extract_model_kwargs, load_search_result_row
from qlib_tw.research.settings import COMBO_CONFIGS, MODEL_CONFIGS, combo_choices, resolve_combos


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
    parser.add_argument("--from-search", type=Path, default=None, help="Load model params from an auto search results CSV")
    parser.add_argument("--run-index", type=int, default=None, help="Select a specific run_index from --from-search")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output/experiment combo name when using --from-search. Default: <combo>_searchrun<run_index>",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()

    if args.from_search is not None and args.run_index is None:
        raise SystemExit("--run-index is required when using --from-search")

    if args.from_search is not None and args.combo is not None and len(args.combo) > 1:
        raise SystemExit("--from-search only supports a single --combo at a time")

    search_row = None
    search_combo = None
    model_kwargs_override = None
    runtime_name_override = None

    if args.from_search is not None:
        search_row = load_search_result_row(args.from_search, args.run_index)
        search_combo = str(search_row.get("combo")) if "combo" in search_row.index else None
        combos = [args.combo[0]] if args.combo else [search_combo]
        if combos[0] is None:
            raise SystemExit("Unable to infer combo from search CSV; please pass --combo explicitly")
        if search_combo and combos[0] != search_combo:
            logging.warning(
                "Search row combo=%s but selected combo=%s; applying search params onto selected combo",
                search_combo,
                combos[0],
            )
        runtime_name_override = args.run_name or f"{combos[0]}_searchrun{args.run_index}"
    else:
        combos = resolve_combos(args.combo)

    init_qlib()
    for combo_name in combos:
        spec = COMBO_CONFIGS[combo_name]
        runtime_combo_name = combo_name
        model_kwargs_override = None
        if search_row is not None:
            allowed_keys = MODEL_CONFIGS[spec["model"]]["kwargs"].keys()
            model_kwargs_override = extract_model_kwargs(search_row, allowed_keys)
            runtime_combo_name = runtime_name_override
        model_rid = train_combo(
            runtime_combo_name,
            spec["handler"],
            spec["model"],
            spec.get("max_instruments"),
            spec.get("infer_processors"),
            model_kwargs_override=model_kwargs_override,
        )
        backtest_combo(
            runtime_combo_name,
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
            recorder_override=model_rid,
            model_kwargs_override=model_kwargs_override,
        )


if __name__ == "__main__":
    main()
