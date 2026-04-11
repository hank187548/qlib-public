#!/usr/bin/env python3
"""Train-only pipeline (no backtest) for Taiwan Qlib combos."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from research.runner import init_qlib, train_combo
from research.settings import COMBO_CONFIGS, combo_choices, resolve_combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train models (no backtest)")
    parser.add_argument("--combo", choices=combo_choices(), nargs="+", help="Combos to train (default alpha158_lgb). Use all to run all combos.")
    parser.add_argument("--threads", type=int, default=4, help="Limit training threads (catboost/lgb)")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = parse_args()
    combos = resolve_combos(args.combo)

    init_qlib()
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
