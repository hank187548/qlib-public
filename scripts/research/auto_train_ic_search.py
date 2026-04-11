#!/usr/bin/env python3
"""
GPU-first IC search helper for selected Taiwan workflow combos.

Examples:
- Search CatBoost on GPU:
    python3 scripts/auto_train_ic_search.py --combo alpha158_cat --trials 100
- Search LightGBM:
    python3 scripts/auto_train_ic_search.py --combo alpha158_lgb --trials 100
- Search both and score on valid:
    python3 scripts/auto_train_ic_search.py --combo alpha158_lgb alpha158_cat --segment valid
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import qlib
from qlib.utils import init_instance_by_config

from research.builders import build_task_config
from research.ic import calc_daily_ic
from research.settings import COMBO_CONFIGS, PROVIDER_URI, UNIVERSE


ComboSampler = Callable[[random.Random, argparse.Namespace], Dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC screening helper for alpha158_lgb / alpha158_cat")
    parser.add_argument(
        "--combo",
        choices=["alpha158_lgb", "alpha158_cat"],
        nargs="+",
        default=["alpha158_cat"],
        help="One or more combos to screen. Default alpha158_cat.",
    )
    parser.add_argument("--trials", type=int, default=400, help="Number of random trials per combo")
    parser.add_argument(
        "--segment",
        choices=["train", "valid", "test"],
        default="valid",
        help="Dataset segment used for IC scoring. Default valid.",
    )
    parser.add_argument("--seed", type=int, default=20231203, help="Random seed")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "outputs",
        help="Root folder for auto search outputs",
    )
    parser.add_argument(
        "--cat-task-type",
        choices=["GPU", "CPU"],
        default="GPU",
        help="CatBoost task type. Default GPU.",
    )
    parser.add_argument(
        "--cat-devices",
        default=None,
        help="Optional CatBoost devices string, e.g. 0 or 0:1. Only used for GPU mode.",
    )
    return parser.parse_args()


def sample_params_lgb(rng: random.Random, _: argparse.Namespace) -> Dict[str, float | int]:
    return {
        "learning_rate": rng.choice([0.02, 0.03, 0.04, 0.05]),
        "max_depth": rng.choice([5, 6, 7]),
        "num_leaves": rng.choice([64, 96, 128]),
        "min_child_samples": rng.choice([20, 50, 80]),
        "colsample_bytree": rng.choice([0.6, 0.7, 0.75, 0.85]),
        "subsample": rng.choice([0.7, 0.8, 0.9]),
        "lambda_l1": rng.choice([0.0, 10.0, 50.0, 100.0]),
        "lambda_l2": rng.choice([50.0, 150.0, 300.0]),
        "num_threads": os.cpu_count() or 8,
    }


def sample_params_cat(rng: random.Random, args: argparse.Namespace) -> Dict[str, float | int | str]:
    params: Dict[str, float | int | str] = {
        "learning_rate": rng.choice([0.02, 0.03, 0.05]),
        "depth": rng.choice([6, 8, 10]),
        "l2_leaf_reg": rng.choice([1.0, 3.0, 5.0, 7.0]),
        "subsample": rng.choice([0.7, 0.8, 0.9, 1.0]),
        "random_strength": rng.choice([0.5, 0.8, 1.0]),
        "iterations": rng.choice([500, 800, 1000]),
        "leaf_estimation_iterations": rng.choice([2, 5, 8]),
        "bootstrap_type": "Bernoulli",
        "task_type": args.cat_task_type,
    }
    if args.cat_task_type == "GPU" and args.cat_devices:
        params["devices"] = args.cat_devices
    if args.cat_task_type == "CPU":
        params["thread_count"] = min(4, os.cpu_count() or 4)
    return params


COMBO_SAMPLERS: Dict[str, ComboSampler] = {
    "alpha158_lgb": sample_params_lgb,
    "alpha158_cat": sample_params_cat,
}


def run_combo_trials(
    combo_name: str,
    *,
    trials: int,
    segment: str,
    rng: random.Random,
    args: argparse.Namespace,
) -> pd.DataFrame:
    combo_spec = COMBO_CONFIGS[combo_name]
    base_task = build_task_config(
        handler_key=combo_spec["handler"],
        model_key=combo_spec["model"],
        instruments=UNIVERSE,
        max_instruments=combo_spec.get("max_instruments"),
        infer_processors=combo_spec.get("infer_processors"),
    )
    sampler = COMBO_SAMPLERS[combo_name]

    result_rows: List[Dict[str, object]] = []
    success_count = 0

    for idx in range(trials):
        params = sampler(rng, args)
        task_cfg = deepcopy(base_task)
        task_cfg["model"]["kwargs"].update(params)
        model = init_instance_by_config(task_cfg["model"])
        dataset = init_instance_by_config(task_cfg["dataset"])

        print(f"[{combo_name} {idx + 1}/{trials}] params: {params}")
        try:
            model.fit(dataset)
            pred = model.predict(dataset, segment=segment)
            label_df = dataset.prepare(segment, col_set="label")
            label_df.columns = ["label"]
            mean_ic, std_ic, n_ic = calc_daily_ic(pred, label_df["label"])
            success_count += 1
        except Exception as err:
            mean_ic = float("nan")
            std_ic = float("nan")
            n_ic = 0
            print(f"    [warn] {combo_name} run {idx + 1} failed: {err}")

        row = {
            "run_index": idx + 1,
            "combo": combo_name,
            "segment": segment,
            **params,
            "ic_mean": mean_ic,
            "ic_std": std_ic,
            "ic_days": n_ic,
        }
        result_rows.append(row)
        if pd.isna(mean_ic):
            print("    IC invalid (nan)")
        else:
            print(f"    IC mean={mean_ic:.4f}, std={std_ic:.4f}, days={n_ic}")

    if success_count == 0:
        raise SystemExit(
            f"All {trials} trials failed for {combo_name}. "
            f"If using CatBoost GPU, check GPU availability or pass --cat-task-type CPU explicitly."
        )

    return pd.DataFrame(result_rows).sort_values("ic_mean", ascending=False, na_position="last")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Initialize Qlib: {PROVIDER_URI}")
    qlib.init(provider_uri=str(PROVIDER_URI), region="tw")

    for combo_name in args.combo:
        out_dir = args.output_root / f"auto_search_{combo_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        df = run_combo_trials(
            combo_name,
            trials=args.trials,
            segment=args.segment,
            rng=rng,
            args=args,
        )
        out_path = out_dir / "results.csv"
        df.to_csv(out_path, index=False)
        print(f"\n{combo_name} results written to: {out_path}")
        print(df.head())


if __name__ == "__main__":
    main()
