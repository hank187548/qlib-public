#!/usr/bin/env python3
"""Taiwan Qlib research workflow with explicit train/backtest stages."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any


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
    parser.add_argument(
        "--from-strategy-search",
        type=Path,
        default=None,
        help="Load best topk/n_drop/strategy/deal-price from a backtest_search output dir or best_result.json",
    )
    parser.add_argument("--n-drop", type=int, default=None, help="Override TopkDropout n_drop (outputs under <combo>_ndrop<N>)")
    parser.add_argument("--topk", type=int, default=None, help="Override TopkDropout topk (outputs under <combo>_topk<K>)")
    parser.add_argument("--rebalance", choices=["day", "week"], default=None, help="Rebalance frequency for backtest/exchange")
    parser.add_argument("--strategy", choices=["bucket", "equal"], default=None, help="bucketed weights or equal-weight TopkDropout")
    parser.add_argument("--deal-price", choices=["close", "open"], default=None, help="Execution price assumption for backtest")
    parser.add_argument("--account", type=float, default=None, help="Override initial account cash")
    parser.add_argument("--backtest-end", default=None, help="Override final backtest end date")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_search_paths(path: Path) -> tuple[Path, Path | None]:
    if path.is_dir():
        return path / "best_result.json", path / "backtest_search_summary.json"
    summary_path = path.with_name("backtest_search_summary.json")
    return path, summary_path if summary_path.exists() else None


def _infer_run_name(train_experiment: str | None) -> str | None:
    prefix = "tw_train_model_"
    if train_experiment and train_experiment.startswith(prefix):
        return train_experiment[len(prefix) :]
    return None


def _load_strategy_search(path: Path) -> dict[str, Any]:
    best_path, summary_path = _strategy_search_paths(path)
    if not best_path.exists():
        raise SystemExit(f"Strategy search best result not found: {best_path}")

    best_payload = _read_json(best_path)
    summary_payload = _read_json(summary_path) if summary_path and summary_path.exists() else {}
    strategy_params = best_payload.get("strategy_params") or summary_payload.get("best_strategy_params")
    if not strategy_params:
        raise SystemExit(f"Strategy search result has no strategy_params: {best_path}")

    train_experiment = summary_payload.get("train_experiment")
    return {
        "combo": summary_payload.get("combo"),
        "run_name": _infer_run_name(train_experiment),
        "train_recorder_id": summary_payload.get("train_recorder_id"),
        "strategy_params": strategy_params,
    }


def _apply_strategy_search_defaults(args: argparse.Namespace) -> None:
    if getattr(args, "from_strategy_search", None) is None:
        return

    search = _load_strategy_search(args.from_strategy_search)
    strategy_params = search["strategy_params"]

    if args.combo is None and search.get("combo"):
        args.combo = [search["combo"]]
    if args.run_name is None and search.get("run_name"):
        args.run_name = search["run_name"]
    if args.recorder_id is None and search.get("train_recorder_id"):
        args.recorder_id = search["train_recorder_id"]
    if args.topk is None and strategy_params.get("topk") is not None:
        args.topk = int(strategy_params["topk"])
    if args.n_drop is None and strategy_params.get("n_drop") is not None:
        args.n_drop = int(strategy_params["n_drop"])
    if args.strategy is None and strategy_params.get("strategy") is not None:
        args.strategy = str(strategy_params["strategy"])
    if args.deal_price is None and strategy_params.get("deal_price") is not None:
        args.deal_price = str(strategy_params["deal_price"])
    if args.rebalance is None and strategy_params.get("rebalance") is not None:
        args.rebalance = str(strategy_params["rebalance"])


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
            rebalance=args.rebalance or "day",
            strategy_choice=args.strategy or "bucket",
            deal_price=args.deal_price or "close",
            recorder_override=recorder_override,
            account_override=args.account,
            backtest_end_time=args.backtest_end,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()

    if args.command == "train":
        specs = _resolve_runtime_specs(args, include_search_params=True)
        _run_train(args, specs)
        return

    if args.command == "backtest":
        _apply_strategy_search_defaults(args)
        specs = _resolve_runtime_specs(args, include_search_params=False)
        _run_backtest(args, specs)
        return

    if args.command == "full":
        _apply_strategy_search_defaults(args)
        specs = _resolve_runtime_specs(args, include_search_params=True)
        train_results = _run_train(args, specs)
        recorder_ids = {runtime_name: recorder_id for runtime_name, recorder_id in train_results}
        _run_backtest(args, specs, recorder_ids=recorder_ids)
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
