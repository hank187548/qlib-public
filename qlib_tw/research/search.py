from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config

from qlib_tw.data_layout import build_exp_manager_config
from qlib_tw.research.builders import build_task_config
from qlib_tw.research.ic import calc_daily_ic
from qlib_tw.research.settings import COMBO_CONFIGS, PROVIDER_URI, REGION, UNIVERSE, WORK_DIR


LOGGER = logging.getLogger("qlib_tw.research.model_search")

DEFAULT_OUTPUT_ROOT = WORK_DIR / "outputs" / "model_search"
DEFAULT_SCREEN_METRIC = "ic_mean"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model search: screen model parameters by IC and export the highest-ranked candidate models"
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config file")
    parser.add_argument("--combo", default=None, choices=sorted(COMBO_CONFIGS.keys()))
    parser.add_argument("--n-trials", type=int, default=None, help="Number of model parameter trials for IC screening")
    parser.add_argument("--top-n", type=int, default=None, help="Number of highest-ranked models to export as top candidates")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for parameter sampling")
    parser.add_argument("--run-tag", default=None, help="Identifier for this model-search run; defaults to timestamp")
    parser.add_argument("--screen-segment", default=None, choices=["train", "valid", "test"], help="Dataset segment used for IC screening; default valid")
    parser.add_argument("--screen-ranking-metric", default=None, help="Metric used to rank screening trials; default ic_mean")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root for pipeline artifacts")
    return parser


def load_config(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise SystemExit("YAML config requires pyyaml to be installed") from exc
        data = yaml.safe_load(text)
        return data or {}
    raise SystemExit(f"Unsupported config format: {path}")


def resolve_setting(cli_value: Any, config: Dict[str, Any], key: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if key in config:
        return config[key]
    return default


def normalize_search_space(space: Dict[str, Any]) -> Dict[str, List[Any]]:
    normalized: Dict[str, List[Any]] = {}
    for key, value in space.items():
        if isinstance(value, (list, tuple)):
            if not value:
                raise SystemExit(f"Search space for {key} is empty")
            normalized[key] = list(value)
        else:
            normalized[key] = [value]
    return normalized


def default_model_search_space(model_key: str) -> Dict[str, List[Any]]:
    cpu_count = os.cpu_count() or 8
    screen_threads = min(8, cpu_count)
    if model_key == "lgb":
        return {
            "learning_rate": [0.02, 0.03, 0.05],
            "max_depth": [5, 6, 7],
            "num_leaves": [64, 96, 128],
            "min_child_samples": [20, 50, 80],
            "colsample_bytree": [0.6, 0.75, 0.85],
            "subsample": [0.7, 0.8, 0.9],
            "lambda_l1": [0.0, 10.0, 50.0, 100.0],
            "lambda_l2": [50.0, 150.0, 300.0],
            "num_threads": [screen_threads],
        }
    if model_key == "cat":
        return {
            "learning_rate": [0.02, 0.03, 0.05],
            "depth": [6, 8, 10],
            "l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
            "subsample": [0.7, 0.8, 0.9],
            "random_strength": [0.5, 0.8, 1.0],
            "iterations": [500, 800, 1000],
            "leaf_estimation_iterations": [2, 5, 8],
            "bootstrap_type": ["Bernoulli"],
            "task_type": ["CPU"],
            "thread_count": [min(4, cpu_count)],
        }
    if model_key == "xgb":
        return {
            "eta": [0.02, 0.04, 0.06],
            "max_depth": [5, 6, 8],
            "num_boost_round": [300, 500, 700],
            "colsample_bytree": [0.6, 0.8, 0.9],
            "subsample": [0.7, 0.8, 0.9],
            "nthread": [screen_threads],
        }
    raise SystemExit(f"Unsupported model key for search: {model_key}")


def sample_from_search_space(space: Dict[str, List[Any]], rng: random.Random) -> Dict[str, Any]:
    return {key: rng.choice(values) for key, values in space.items()}


def generate_model_trials(space: Dict[str, List[Any]], n_trials: int, rng: random.Random) -> List[Dict[str, Any]]:
    trials: List[Dict[str, Any]] = []
    seen = set()
    max_attempts = max(n_trials * 20, 50)
    attempts = 0
    while len(trials) < n_trials and attempts < max_attempts:
        attempts += 1
        params = sample_from_search_space(space, rng)
        signature = json.dumps(params, sort_keys=True, default=str)
        if signature in seen:
            continue
        seen.add(signature)
        trials.append(params)
    if len(trials) < n_trials:
        LOGGER.warning("Only generated %d unique model trials out of requested %d", len(trials), n_trials)
    return trials


def build_combo_task_config(combo_name: str, model_params: Dict[str, Any]) -> Dict[str, Any]:
    spec = COMBO_CONFIGS[combo_name]
    return build_task_config(
        spec["handler"],
        spec["model"],
        UNIVERSE,
        spec.get("max_instruments"),
        spec.get("infer_processors"),
        model_kwargs_override=model_params,
    )


def flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, Path):
            flat[key] = str(value)
        else:
            flat[key] = value
    return flat


def save_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame([flatten_row(row) for row in rows]).to_csv(path, index=False)


def save_json(data: Any, path: Path) -> None:
    def default_serializer(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value)} is not JSON serializable")

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default_serializer), encoding="utf-8")


def ensure_no_active_mlflow_run() -> None:
    while mlflow.active_run() is not None:
        mlflow.end_run()


def run_screen_trial(combo_name: str, model_params: Dict[str, Any], screen_segment: str) -> Dict[str, Any]:
    task_cfg = build_combo_task_config(combo_name, model_params)
    model = init_instance_by_config(task_cfg["model"])
    dataset = init_instance_by_config(task_cfg["dataset"])
    try:
        model.fit(dataset, **task_cfg.get("model_fit_kwargs", {}))
        pred = model.predict(dataset, segment=screen_segment)
        label_df = dataset.prepare(screen_segment, col_set="label")
        label_df.columns = ["label"]
        ic_mean, ic_std, ic_days = calc_daily_ic(pred, label_df["label"])
    finally:
        ensure_no_active_mlflow_run()
        del model
        del dataset
        gc.collect()
    ic_ir = float("nan")
    if pd.notna(ic_mean) and pd.notna(ic_std) and ic_std not in (0, 0.0):
        ic_ir = float(ic_mean / ic_std)
    return {
        "screen_segment": screen_segment,
        "ic_mean": ic_mean,
        "ic_std": ic_std,
        "ic_days": ic_days,
        "ic_ir": ic_ir,
        **dict(model_params),
        "model_params": dict(model_params),
    }


def _extract_metric(row: Dict[str, Any], metric_name: str) -> float:
    value = row.get(metric_name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def sort_rows(rows: List[Dict[str, Any]], metric_name: str) -> List[Dict[str, Any]]:
    valid = [row for row in rows if not pd.isna(_extract_metric(row, metric_name))]
    invalid = [row for row in rows if pd.isna(_extract_metric(row, metric_name))]
    valid.sort(key=lambda row: _extract_metric(row, metric_name), reverse=True)
    return valid + invalid


def resolved_settings(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    settings["combo"] = resolve_setting(args.combo, config, "combo", None)
    settings["n_trials"] = int(resolve_setting(args.n_trials, config, "n_trials", 20))
    settings["top_n"] = int(resolve_setting(args.top_n, config, "top_n", 5))
    settings["seed"] = int(resolve_setting(args.seed, config, "seed", 20260409))
    settings["run_tag"] = resolve_setting(args.run_tag, config, "run_tag", datetime.now().strftime("modelsearch%Y%m%d_%H%M%S"))
    settings["screen_segment"] = resolve_setting(args.screen_segment, config, "screen_segment", "valid")
    settings["screen_ranking_metric"] = resolve_setting(args.screen_ranking_metric, config, "screen_ranking_metric", DEFAULT_SCREEN_METRIC)
    settings["output_root"] = Path(resolve_setting(args.output_root, config, "output_root", DEFAULT_OUTPUT_ROOT))
    if not settings["combo"]:
        raise SystemExit("combo is required, either via CLI or config")
    model_key = COMBO_CONFIGS[settings["combo"]]["model"]
    settings["model_search_space"] = normalize_search_space(resolve_setting(None, config, "model_search_space", default_model_search_space(model_key)))
    return settings


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = load_config(args.config)
    settings = resolved_settings(args, config)

    output_dir = settings["output_root"] / settings["run_tag"]
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(settings, output_dir / "resolved_config.json")

    LOGGER.info("Initialize Qlib, provider uri: %s", PROVIDER_URI)
    qlib.init(provider_uri=str(PROVIDER_URI), region=REGION, exp_manager=build_exp_manager_config())
    LOGGER.info("Universe size: %d symbols", len(UNIVERSE))

    rng = random.Random(settings["seed"])
    model_trials = generate_model_trials(settings["model_search_space"], settings["n_trials"], rng)
    LOGGER.info("Generated %d model trials for combo %s", len(model_trials), settings["combo"])

    screen_results: List[Dict[str, Any]] = []
    for index, model_params in enumerate(model_trials, start=1):
        LOGGER.info("Screen trial %d/%d with params %s", index, len(model_trials), model_params)
        row = run_screen_trial(settings["combo"], model_params, settings["screen_segment"])
        row["run_index"] = index
        screen_results.append(row)

    screen_results = sort_rows(screen_results, settings["screen_ranking_metric"])
    save_rows_csv(screen_results, output_dir / "model_search_results.csv")
    save_json(screen_results, output_dir / "model_search_results.json")

    top_models = screen_results[: min(settings["top_n"], len(screen_results))]
    save_rows_csv(top_models, output_dir / "top_model_candidates.csv")
    if not screen_results:
        raise SystemExit("No model-search results were produced")

    best_model = screen_results[0]
    save_json(best_model, output_dir / "best_model.json")
    LOGGER.info(
        "Best model: %s = %.6f (run_index=%s)",
        settings["screen_ranking_metric"],
        _extract_metric(best_model, settings["screen_ranking_metric"]),
        best_model["run_index"],
    )

    return 0
