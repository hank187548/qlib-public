from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List


WORK_DIR = Path(__file__).resolve().parents[2]
PROVIDER_URI = WORK_DIR / "Data" / "tw_data"
REGION = "tw"
BENCHMARK = "^TWII"
DEFAULT_COMBO = "alpha158_lgb"

BASE_DATA_HANDLER_CONFIG: Dict[str, object] = {
    "start_time": "2018-01-01",
    "end_time": "2026-04-10",
    "fit_start_time": "2018-01-01",
    "fit_end_time": "2024-12-31",
}

SEGMENTS: Dict[str, tuple[str, str]] = {
    "train": ("2018-01-01", "2024-12-31"),
    "valid": ("2025-01-01", "2025-06-30"),
    "test": ("2025-07-01", "2026-04-09"),
}

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "lgb": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.75,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "lambda_l1": 50.0,
            "lambda_l2": 200.0,
            "max_depth": 6,
            "num_leaves": 96,
            "min_child_samples": 50,
            "num_threads": os.cpu_count() or 8,
        },
    },
    "lgb_run11": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.75,
            "learning_rate": 0.02,
            "subsample": 0.7,
            "lambda_l1": 50.0,
            "lambda_l2": 150.0,
            "max_depth": 7,
            "num_leaves": 128,
            "min_child_samples": 80,
            "num_threads": os.cpu_count() or 8,
        },
    },
    "xgb": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "eval_metric": "rmse",
            "colsample_bytree": 0.8879,
            "eta": 0.0421,
            "max_depth": 8,
            "n_estimators": 647,
            "subsample": 0.8789,
            "nthread": os.cpu_count() or 8,
        },
    },
    "cat": {
        "class": "CatBoostModel",
        "module_path": "qlib.contrib.model.catboost_model",
        "kwargs": {
            "loss_function": "RMSE",
            "iterations": 800,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 3,
            "subsample": 0.9,
            "bootstrap_type": "Bernoulli",
            "random_strength": 0.8,
            "leaf_estimation_iterations": 5,
            "task_type": "CPU",
            "thread_count": min(4, os.cpu_count() or 4),
        },
    },
}

HANDLER_CONFIGS: Dict[str, Dict[str, str]] = {
    "alpha158": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"},
    "alpha360": {"class": "Alpha360", "module_path": "qlib.contrib.data.handler"},
    "alpha191": {"class": "Alpha191", "module_path": "qlib.contrib.data.handler"},
}

ALPHA158_INFER_PIPELINE = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]

COMBO_CONFIGS = {
    "alpha158_lgb": {"handler": "alpha158", "model": "lgb", "max_instruments": None, "infer_processors": []},
    "alpha158_lgb_run11": {
        "handler": "alpha158",
        "model": "lgb_run11",
        "max_instruments": None,
        "infer_processors": [],
    },
    "alpha158_lgb_pro_fil": {
        "handler": "alpha158",
        "model": "lgb",
        "max_instruments": None,
        "infer_processors": ALPHA158_INFER_PIPELINE,
    },
    "alpha360_lgb": {"handler": "alpha360", "model": "lgb", "max_instruments": None, "infer_processors": []},
    "alpha191_lgb": {"handler": "alpha191", "model": "lgb", "max_instruments": None, "infer_processors": []},
    "alpha158_xgb": {
        "handler": "alpha158",
        "model": "xgb",
        "max_instruments": None,
        "infer_processors": ALPHA158_INFER_PIPELINE,
    },
    "alpha360_xgb": {"handler": "alpha360", "model": "xgb", "max_instruments": None, "infer_processors": []},
    "alpha191_xgb": {"handler": "alpha191", "model": "xgb", "max_instruments": None, "infer_processors": []},
    "alpha158_cat": {"handler": "alpha158", "model": "cat", "max_instruments": None, "infer_processors": []},
    "alpha360_cat": {"handler": "alpha360", "model": "cat", "max_instruments": None, "infer_processors": []},
}

BASE_PORT_ANALYSIS_CONFIG: Dict[str, object] = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "BucketWeightTopkDropout",
        "module_path": "qlib_tw.trade.custom_strategy",
        "kwargs": {
            "model": None,
            "dataset": None,
            "topk": 50,
            "n_drop": 5,
            "forbid_all_trade_at_limit": False,
        },
    },
    "backtest": {
        "start_time": SEGMENTS["test"][0],
        "end_time": SEGMENTS["test"][1],
        "account": 100_000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "freq": "day",
            "limit_threshold": (
                "$change >= 0.095 * Ref($close, 1)",
                "$change <= -0.095 * Ref($close, 1)",
            ),
            "deal_price": "close",
            "open_cost": 0.00092625,
            "close_cost": 0.00242625,
            "min_cost": 1,
            "trade_unit": 1,
        },
    },
}


def load_full_universe(provider_uri: Path, benchmark: str = BENCHMARK) -> list[str]:
    inst_path = provider_uri / "instruments" / "all.txt"
    symbols = set()
    if inst_path.exists():
        for line in inst_path.read_text().splitlines():
            parts = line.strip().split()
            if parts:
                symbols.add(parts[0].upper())
    features_dir = provider_uri / "features"
    if features_dir.exists():
        for entry in features_dir.iterdir():
            if entry.is_dir():
                symbols.add(entry.name.upper())
    if benchmark:
        symbols.add(benchmark.upper())
    if not symbols:
        raise RuntimeError(f"No instruments found under {provider_uri}")
    return sorted(symbols)


ALL_CODES = load_full_universe(PROVIDER_URI)
UNIVERSE = [code for code in ALL_CODES if not code.startswith("^")]
if not UNIVERSE:
    raise RuntimeError(f"No equity instruments found under {PROVIDER_URI}")


def combo_choices() -> list[str]:
    return sorted(list(COMBO_CONFIGS.keys()) + ["all"])


def resolve_combos(requested: List[str] | None) -> List[str]:
    if not requested:
        return [DEFAULT_COMBO]
    if "all" in requested:
        return list(COMBO_CONFIGS.keys())
    return requested
