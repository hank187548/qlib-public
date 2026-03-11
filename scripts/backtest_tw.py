#!/usr/bin/env python3
"""Backtest-only pipeline using trained models."""

from __future__ import annotations

import argparse
import logging
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord

from scripts.workflow_by_code_tw import (
    COMBO_CONFIGS,
    UNIVERSE,
    PROVIDER_URI,
    DEFAULT_COMBO,
    build_task_config,
    build_port_analysis_config,
    set_output_dirs,
    dump_report_frames,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest with trained models")
    combo_choices = sorted(list(COMBO_CONFIGS.keys()) + ["all"])
    parser.add_argument(
        "--combo",
        choices=combo_choices,
        nargs="+",
        help="Combos to backtest (default alpha158_lgb). Use all to run all combos.",
    )
    parser.add_argument("--n-drop", type=int, default=None, help="Override TopkDropout n_drop")
    parser.add_argument("--topk", type=int, default=None, help="Override TopkDropout topk")
    parser.add_argument("--rebalance", choices=["day", "week"], default="day", help="Backtest rebalance frequency")
    parser.add_argument("--strategy", choices=["bucket", "equal"], default="bucket", help="bucket weighting / equal weighting")
    parser.add_argument(
        "--deal-price",
        choices=["close", "open"],
        default="close",
        help="Execution price assumption for backtest; default close, optional open",
    )
    parser.add_argument(
        "--limit-tplus",
        action="store_true",
        help="Enable next-open limit order + T+2 settlement (base=open, slippage=1%%). Default is immediate settlement.",
    )
    parser.add_argument(
        "--recorder-id",
        default=None,
        help="Specify recorder id from training experiment; if omitted, use the latest recorder.",
    )
    return parser.parse_args()


def resolve_combos(requested: List[str] | None) -> List[str]:
    if not requested:
        return [DEFAULT_COMBO]
    if "all" in requested:
        return list(COMBO_CONFIGS.keys())
    return requested


def latest_recorder_id(experiment: str) -> str:
    recs = R.list_recorders(experiment_name=experiment)
    if not recs:
        raise RuntimeError(f"No recorder found for experiment {experiment}; train model first")
    def _ts(rid: str) -> float:
        info = recs[rid].info
        ts = info.get("start_time")
        try:
            from datetime import datetime
            return datetime.fromisoformat(ts).timestamp()
        except Exception:
            return 0.0
    return sorted(recs.keys(), key=_ts, reverse=True)[0]


def backtest_combo(
    combo_name: str,
    handler_key: str,
    model_key: str,
    max_instruments: int | None,
    infer_processors: List[Dict[str, object]] | None,
    n_drop_override: int | None,
    topk_override: int | None,
    rebalance: str,
    strategy_choice: str,
    deal_price: str,
    limit_tplus: bool,
    recorder_override: str | None,
) -> None:
    effective_name = combo_name
    if n_drop_override is not None:
        effective_name = f"{effective_name}_ndrop{n_drop_override}"
    if topk_override is not None:
        effective_name = f"{effective_name}_topk{topk_override}"
    if rebalance != "day":
        effective_name = f"{effective_name}_{rebalance}"
    if strategy_choice != "bucket":
        effective_name = f"{effective_name}_{strategy_choice}"
    set_output_dirs(effective_name)

    task_config = build_task_config(handler_key, model_key, UNIVERSE, max_instruments, infer_processors)
    port_config = build_port_analysis_config()
    if strategy_choice == "equal":
        port_config["strategy"]["class"] = "TopkDropoutStrategy"
        port_config["strategy"]["module_path"] = "qlib.contrib.strategy.signal_strategy"
    if n_drop_override is not None:
        port_config["strategy"]["kwargs"]["n_drop"] = n_drop_override
    if topk_override is not None:
        port_config["strategy"]["kwargs"]["topk"] = topk_override
    if rebalance != "day":
        port_config["executor"]["kwargs"]["time_per_step"] = rebalance
        port_config["backtest"]["exchange_kwargs"]["freq"] = rebalance

    port_config["backtest"]["exchange_kwargs"]["deal_price"] = deal_price
    if limit_tplus:
        base_ex_kwargs = deepcopy(port_config["backtest"]["exchange_kwargs"])
        # Next-open limit order ±1% + T+2 settlement
        base_ex_kwargs["deal_price"] = "open"
        exchange_cfg = {
            "class": "TPlusLimitExchange",
            "module_path": "scripts.custom_exchange",
            "kwargs": {
                **base_ex_kwargs,
                "start_time": port_config["backtest"]["start_time"],
                "end_time": port_config["backtest"]["end_time"],
                "codes": task_config["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"],
                "limit_slippage": 0.01,
                "settlement_lag": 2,
            },
        }
        port_config["backtest"]["exchange_kwargs"] = {"exchange": exchange_cfg}

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    train_exp = f"tw_train_model_{combo_name}"
    recorder_id = recorder_override or latest_recorder_id(train_exp)
    logging.info("Use training recorder: %s (experiment %s)", recorder_id, train_exp)
    train_recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=train_exp)
    trained_model = train_recorder.load_object("trained_model")

    backtest_exp = f"tw_backtest_{combo_name}"
    logging.info("Start backtest - experiment %s", backtest_exp)
    with R.start(experiment_name=backtest_exp):
        current_recorder = R.get_recorder()
        signal_rec = SignalRecord(trained_model, dataset, current_recorder)
        signal_rec.generate()

        strategy_kwargs = port_config["strategy"]["kwargs"]
        strategy_kwargs["model"] = trained_model
        strategy_kwargs["dataset"] = dataset

        port_rec = PortAnaRecord(current_recorder, port_config, "day")
        port_rec.generate()
        backtest_rid = current_recorder.id

    logging.info("Backtest complete, recorder id = %s", backtest_rid)
    recorder = R.get_recorder(recorder_id=backtest_rid, experiment_name=backtest_exp)
    handler_kwargs = task_config["dataset"]["kwargs"]["handler"]["kwargs"]
    segments = task_config["dataset"]["kwargs"]["segments"]
    dump_report_frames(
        recorder,
        dataset,
        universe=handler_kwargs["instruments"],
        data_handler_config=handler_kwargs,
        segments=segments,
        port_config=port_config,
    )
    logging.info("Combo %s completed. Output root: %s", effective_name, recorder.list_tags().get("output", "outputs"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    combos = resolve_combos(args.combo)

    logging.info("Initialize Qlib, provider uri: %s", PROVIDER_URI)
    qlib.init(provider_uri=str(PROVIDER_URI), region="tw")
    logging.info("Universe size: %d symbols", len(UNIVERSE))

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
