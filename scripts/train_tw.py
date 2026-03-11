#!/usr/bin/env python3
"""Train-only pipeline (no backtest) for Taiwan Qlib combos."""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R

from scripts.workflow_by_code_tw import (
    COMBO_CONFIGS,
    UNIVERSE,
    PROVIDER_URI,
    DEFAULT_COMBO,
    build_task_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models (no backtest)")
    combo_choices = sorted(list(COMBO_CONFIGS.keys()) + ["all"])
    parser.add_argument(
        "--combo",
        choices=combo_choices,
        nargs="+",
        help="Combos to train (default alpha158_lgb). Use all to run all combos.",
    )
    parser.add_argument("--threads", type=int, default=4, help="Limit training threads (catboost/lgb)")
    return parser.parse_args()


def resolve_combos(requested: List[str] | None) -> List[str]:
    if not requested:
        return [DEFAULT_COMBO]
    if "all" in requested:
        return list(COMBO_CONFIGS.keys())
    return requested


def train_combo(combo_name: str, handler_key: str, model_key: str, max_instruments, infer_processors, threads: int):
    task_config = build_task_config(handler_key, model_key, UNIVERSE, max_instruments, infer_processors)
    # override thread_count if present
    if "thread_count" in task_config["model"]["kwargs"]:
        task_config["model"]["kwargs"]["thread_count"] = threads
    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    train_exp = f"tw_train_model_{combo_name}"
    logging.info("Start model training - experiment %s", train_exp)
    with R.start(experiment_name=train_exp):
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        model_rid = R.get_recorder().id
    logging.info("Model training complete, recorder id = %s", model_rid)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    combos = resolve_combos(args.combo)

    logging.info("Initialize Qlib, provider uri: %s", PROVIDER_URI)
    qlib.init(provider_uri=str(PROVIDER_URI), region="tw")
    logging.info("Universe size: %d symbols", len(UNIVERSE))

    for combo_name in combos:
        spec = COMBO_CONFIGS[combo_name]
        logging.info("=== Training combo: %s (handler=%s, model=%s) ===", combo_name, spec["handler"], spec["model"])
        train_combo(
            combo_name,
            spec["handler"],
            spec["model"],
            spec.get("max_instruments"),
            spec.get("infer_processors"),
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
