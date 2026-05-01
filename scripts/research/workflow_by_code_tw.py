#!/usr/bin/env python3
"""Taiwan Qlib research workflow with explicit train/backtest stages."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.research.runner import backtest_combo, init_qlib, train_combo
from qlib_tw.research.builders import MODEL_FIT_KWARG_KEYS
from qlib_tw.research.search_results import extract_model_kwargs, load_search_result_row
from qlib_tw.research.settings import COMBO_CONFIGS, MODEL_CONFIGS, combo_choices, resolve_combos


def _add_combo_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--combo",
        choices=combo_choices(),
        nargs="+",
        help="Specify one or more handler/model combos to run (default alpha158_lgb). Use 'all' to run every combo.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional runtime name for experiment/output folders. Supports a single combo only.",
    )
    parser.add_argument("--from-search", type=Path, default=None, help="Load model params from a model-search results CSV")
    parser.add_argument("--run-index", type=int, default=None, help="Select a specific run_index from --from-search")


def _add_train_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Limit training threads (lightgbm/catboost/xgboost)",
    )


def _add_backtest_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--recorder-id", default=None, help="Use a specific training recorder id instead of the latest run")
    parser.add_argument("--n-drop", type=int, default=None, help="Override TopkDropout n_drop (outputs under <combo>_ndrop<N>)")
    parser.add_argument("--topk", type=int, default=None, help="Override TopkDropout topk (outputs under <combo>_topk<K>)")
    parser.add_argument("--rebalance", choices=["day", "week"], default="day", help="Rebalance frequency for backtest/exchange")
    parser.add_argument("--strategy", choices=["bucket", "equal"], default="bucket", help="bucketed weights or equal-weight TopkDropout")
    parser.add_argument("--deal-price", choices=["close", "open"], default="close", help="Execution price assumption for backtest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taiwan Qlib research workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train model only and export model analysis under outputs/models")
    _add_combo_args(train_parser)
    _add_train_options(train_parser)

    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Run strategy backtest only from an existing trained model and export under outputs/backtest",
    )
    _add_combo_args(backtest_parser)
    _add_backtest_options(backtest_parser)

    full_parser = subparsers.add_parser("full", help="Train model and then backtest in one command")
    _add_combo_args(full_parser)
    _add_train_options(full_parser)
    _add_backtest_options(full_parser)

    return parser.parse_args()


def _resolve_runtime_specs(args: argparse.Namespace, *, include_search_params: bool) -> list[dict]:
    if args.from_search is not None and args.run_index is None:
        raise SystemExit("--run-index is required when using --from-search")

    if args.from_search is not None and args.combo is not None and len(args.combo) > 1:
        raise SystemExit("--from-search only supports a single --combo at a time")

    if args.from_search is not None:
        search_row = load_search_result_row(args.from_search, args.run_index)
        search_combo = str(search_row.get("combo")) if "combo" in search_row.index else None
        combo_names = [args.combo[0]] if args.combo else [search_combo]
        if combo_names[0] is None:
            raise SystemExit("Unable to infer combo from search CSV; please pass --combo explicitly")
        if search_combo and combo_names[0] != search_combo:
            logging.warning(
                "Search row combo=%s but selected combo=%s; applying search params onto selected combo",
                search_combo,
                combo_names[0],
            )
        runtime_name = args.run_name or f"{combo_names[0]}_searchrun{args.run_index}"
        spec = COMBO_CONFIGS[combo_names[0]]
        model_kwargs_override = None
        if include_search_params:
            allowed_keys = set(MODEL_CONFIGS[spec["model"]]["kwargs"].keys()) | MODEL_FIT_KWARG_KEYS
            model_kwargs_override = extract_model_kwargs(search_row, allowed_keys)
        return [
            {
                "combo_name": combo_names[0],
                "runtime_name": runtime_name,
                "spec": spec,
                "model_kwargs_override": model_kwargs_override,
            }
        ]

    combo_names = resolve_combos(args.combo)
    if args.run_name is not None:
        if len(combo_names) != 1:
            raise SystemExit("--run-name only supports a single combo")
        return [
            {
                "combo_name": combo_names[0],
                "runtime_name": args.run_name,
                "spec": COMBO_CONFIGS[combo_names[0]],
                "model_kwargs_override": None,
            }
        ]

    return [
        {
            "combo_name": combo_name,
            "runtime_name": combo_name,
            "spec": COMBO_CONFIGS[combo_name],
            "model_kwargs_override": None,
        }
        for combo_name in combo_names
    ]


def _run_train(args: argparse.Namespace, specs: list[dict]) -> list[tuple[str, str]]:
    init_qlib()
    results = []
    for item in specs:
        spec = item["spec"]
        model_rid = train_combo(
            item["runtime_name"],
            spec["handler"],
            spec["model"],
            spec.get("max_instruments"),
            spec.get("infer_processors"),
            threads=getattr(args, "threads", None),
            model_kwargs_override=item["model_kwargs_override"],
        )
        results.append((item["runtime_name"], model_rid))
    return results


def _run_backtest(args: argparse.Namespace, specs: list[dict], recorder_ids: dict[str, str] | None = None) -> None:
    init_qlib()
    for item in specs:
        spec = item["spec"]
        recorder_override = None
        if recorder_ids is not None:
            recorder_override = recorder_ids.get(item["runtime_name"])
        if getattr(args, "recorder_id", None):
            recorder_override = args.recorder_id
        backtest_combo(
            item["runtime_name"],
            spec["handler"],
            spec["model"],
            spec.get("max_instruments"),
            spec.get("infer_processors"),
            n_drop_override=args.n_drop,
            topk_override=args.topk,
            rebalance=args.rebalance,
            strategy_choice=args.strategy,
            deal_price=args.deal_price,
            recorder_override=recorder_override,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()

    if args.command == "train":
        specs = _resolve_runtime_specs(args, include_search_params=True)
        _run_train(args, specs)
        return

    if args.command == "backtest":
        specs = _resolve_runtime_specs(args, include_search_params=False)
        _run_backtest(args, specs)
        return

    if args.command == "full":
        specs = _resolve_runtime_specs(args, include_search_params=True)
        train_results = _run_train(args, specs)
        recorder_ids = {runtime_name: recorder_id for runtime_name, recorder_id in train_results}
        _run_backtest(args, specs, recorder_ids=recorder_ids)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
