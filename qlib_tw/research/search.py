from __future__ import annotations

import argparse
import gc
import itertools
import json
import logging
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mlflow
import pandas as pd
import qlib
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from qlib_tw.research.builders import apply_strategy_overrides, build_port_analysis_config, build_task_config
from qlib_tw.research.ic import calc_daily_ic
from qlib_tw.research.paths import set_output_dirs
from qlib_tw.research.publish import promote_output
from qlib_tw.research.reports import dump_report_frames
from qlib_tw.research.settings import COMBO_CONFIGS, PROVIDER_URI, REGION, UNIVERSE, WORK_DIR


LOGGER = logging.getLogger("auto_search_pipeline")

DEFAULT_OUTPUT_ROOT = WORK_DIR / "outputs" / "auto_pipeline"
DEFAULT_RANKING_METRIC = "risk.excess_return_with_cost.annualized_return"
DEFAULT_SCREEN_METRIC = "ic_mean"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search model params, backtest top candidates, optionally promote best run")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON/YAML config file")
    parser.add_argument("--combo", default=None, choices=sorted(COMBO_CONFIGS.keys()))
    parser.add_argument("--n-trials", type=int, default=None, help="Number of model parameter trials for IC screening")
    parser.add_argument("--top-n-backtest", type=int, default=None, help="Number of screened models to run full backtests for")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for parameter sampling")
    parser.add_argument("--run-tag", default=None, help="Identifier for this pipeline run; defaults to timestamp")
    parser.add_argument("--screen-segment", default=None, choices=["train", "valid", "test"], help="Dataset segment used for IC screening; default valid")
    parser.add_argument("--screen-ranking-metric", default=None, help="Metric used to rank screening trials; default ic_mean")
    parser.add_argument("--ranking-metric", default=None, help="Metric used to rank full backtests")
    parser.add_argument("--topk-values", type=int, nargs="+", default=None, help="Strategy grid values for topk")
    parser.add_argument("--n-drop-values", type=int, nargs="+", default=None, help="Strategy grid values for n_drop")
    parser.add_argument("--strategy-values", nargs="+", default=None, choices=["bucket", "equal"], help="Strategy grid values for weighting rule")
    parser.add_argument("--deal-price-values", nargs="+", default=None, choices=["close", "open"], help="Strategy grid values for deal price assumption")
    parser.add_argument("--rebalance-values", nargs="+", default=None, choices=["day", "week"], help="Strategy grid values for rebalance frequency")
    parser.add_argument("--limit-tplus-values", nargs="+", default=None, help="Strategy grid values for T+2 limit mode, e.g. false true")
    parser.add_argument("--output-root", type=Path, default=None, help="Output root for pipeline artifacts")
    parser.add_argument("--promote-best", action=argparse.BooleanOptionalAction, default=None, help="Promote the best backtest into outputs/best_run")
    parser.add_argument("--clean-best-run", action=argparse.BooleanOptionalAction, default=None, help="Clean outputs/best_run/reports and figures before promoting")
    parser.add_argument("--screen-only", action=argparse.BooleanOptionalAction, default=None, help="Stop after IC screening and do not run full backtests")
    parser.add_argument("--minimize-metric", action=argparse.BooleanOptionalAction, default=None, help="Rank backtests by lower-is-better instead of higher-is-better")
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


def parse_bool_token(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean token: {value}")


def resolve_bool_setting(cli_value: Any, config: Dict[str, Any], key: str, default: bool) -> bool:
    return parse_bool_token(resolve_setting(cli_value, config, key, default))


def normalize_bool_list(values: Iterable[Any]) -> List[bool]:
    return [parse_bool_token(value) for value in values]


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
            "n_estimators": [300, 500, 700],
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


def build_strategy_variants(settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    variants = []
    for topk, n_drop, strategy, deal_price, rebalance, limit_tplus in itertools.product(
        settings["topk_values"],
        settings["n_drop_values"],
        settings["strategy_values"],
        settings["deal_price_values"],
        settings["rebalance_values"],
        settings["limit_tplus_values"],
    ):
        variants.append(
            {
                "topk": int(topk),
                "n_drop": int(n_drop),
                "strategy": str(strategy),
                "deal_price": str(deal_price),
                "rebalance": str(rebalance),
                "limit_tplus": bool(limit_tplus),
            }
        )
    return variants


def build_combo_task_config(combo_name: str, model_params: Dict[str, Any]) -> Dict[str, Any]:
    spec = COMBO_CONFIGS[combo_name]
    task_cfg = build_task_config(spec["handler"], spec["model"], UNIVERSE, spec.get("max_instruments"), spec.get("infer_processors"))
    task_cfg["model"]["kwargs"].update(deepcopy(model_params))
    return task_cfg


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
        model.fit(dataset)
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
        "model_params": deepcopy(model_params),
    }


def build_output_name(combo_name: str, run_tag: str, model_rank: int, strategy_cfg: Dict[str, Any]) -> str:
    name = (
        f"{combo_name}__{run_tag}__m{model_rank:02d}"
        f"_topk{strategy_cfg['topk']}"
        f"_ndrop{strategy_cfg['n_drop']}"
        f"_{strategy_cfg['strategy']}"
        f"_{strategy_cfg['rebalance']}"
        f"_{strategy_cfg['deal_price']}"
    )
    if strategy_cfg["limit_tplus"]:
        name += "_tplus"
    return name


def parse_summary_metrics(path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value.endswith("%"):
            try:
                metrics[key] = float(value[:-1]) / 100.0
                continue
            except ValueError:
                pass
        try:
            metrics[key] = float(value)
            continue
        except ValueError:
            metrics[key] = value
    return metrics


def extract_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def run_backtest_candidate(
    combo_name: str,
    run_tag: str,
    model_rank: int,
    model_params: Dict[str, Any],
    strategy_cfg: Dict[str, Any],
    ranking_metric: str,
) -> Dict[str, Any]:
    task_cfg = build_combo_task_config(combo_name, model_params)
    output_name = build_output_name(combo_name, run_tag, model_rank, strategy_cfg)
    paths = set_output_dirs(output_name)

    model = init_instance_by_config(task_cfg["model"])
    dataset = init_instance_by_config(task_cfg["dataset"])
    train_exp = f"auto_train_{run_tag}_m{model_rank:02d}"
    backtest_exp = f"auto_backtest_{run_tag}_m{model_rank:02d}_{strategy_cfg['strategy']}_{strategy_cfg['topk']}_{strategy_cfg['n_drop']}"
    try:
        ensure_no_active_mlflow_run()
        with R.start(experiment_name=train_exp):
            R.log_params(**flatten_dict(task_cfg))
            model.fit(dataset)
            R.save_objects(trained_model=model)
            train_rid = R.get_recorder().id

        port_config = build_port_analysis_config()
        apply_strategy_overrides(
            port_config,
            task_cfg,
            n_drop_override=strategy_cfg["n_drop"],
            topk_override=strategy_cfg["topk"],
            rebalance=strategy_cfg["rebalance"],
            strategy_choice=strategy_cfg["strategy"],
            deal_price=strategy_cfg["deal_price"],
            limit_tplus=strategy_cfg["limit_tplus"],
        )

        ensure_no_active_mlflow_run()
        with R.start(experiment_name=backtest_exp):
            current_recorder = R.get_recorder()
            current_recorder.log_params(**strategy_cfg)
            current_recorder.set_tags(output=output_name, pipeline_run=run_tag)
            signal_rec = SignalRecord(model, dataset, current_recorder)
            signal_rec.generate()

            strategy_kwargs = port_config["strategy"]["kwargs"]
            strategy_kwargs["model"] = model
            strategy_kwargs["dataset"] = dataset

            port_rec = PortAnaRecord(current_recorder, port_config, "day")
            port_rec.generate()
            backtest_rid = current_recorder.id

        recorder = R.get_recorder(recorder_id=backtest_rid, experiment_name=backtest_exp)
        handler_kwargs = task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]
        segments = task_cfg["dataset"]["kwargs"]["segments"]
        dump_report_frames(
            recorder,
            dataset,
            universe=handler_kwargs["instruments"],
            data_handler_config=handler_kwargs,
            segments=segments,
            port_config=port_config,
            paths=paths,
        )
        summary_path = paths.report_dir / "summary.txt"
        summary_metrics = parse_summary_metrics(summary_path)
        ranking_value = extract_metric(summary_metrics, ranking_metric)
        return {
            "combo": combo_name,
            "run_tag": run_tag,
            "model_rank": model_rank,
            "output_name": output_name,
            "output_dir": paths.output_root,
            "train_experiment": train_exp,
            "train_recorder_id": train_rid,
            "backtest_experiment": backtest_exp,
            "backtest_recorder_id": backtest_rid,
            "ranking_metric": ranking_metric,
            "ranking_value": ranking_value,
            "strategy_params": deepcopy(strategy_cfg),
            "model_params": deepcopy(model_params),
            "summary_metrics": summary_metrics,
        }
    finally:
        ensure_no_active_mlflow_run()
        del model
        del dataset
        gc.collect()


def promote_best_output(output_name: str, clean_best_run: bool) -> None:
    promote_output(output_name, clean=clean_best_run, translate_summary_file=True)
    LOGGER.info("Promoted best output %s into outputs/best_run", output_name)


def sort_rows(rows: List[Dict[str, Any]], metric_name: str, minimize: bool = False) -> List[Dict[str, Any]]:
    valid = [row for row in rows if not pd.isna(extract_metric(row, metric_name))]
    invalid = [row for row in rows if pd.isna(extract_metric(row, metric_name))]
    valid.sort(key=lambda row: extract_metric(row, metric_name), reverse=not minimize)
    return valid + invalid


def sort_backtest_rows(rows: List[Dict[str, Any]], metric_name: str, minimize: bool = False) -> List[Dict[str, Any]]:
    valid = [row for row in rows if not pd.isna(row.get("ranking_value", float("nan")))]
    invalid = [row for row in rows if pd.isna(row.get("ranking_value", float("nan")))]
    valid.sort(key=lambda row: float(row["ranking_value"]), reverse=not minimize)
    return valid + invalid


def resolved_settings(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    settings: Dict[str, Any] = {}
    settings["combo"] = resolve_setting(args.combo, config, "combo", None)
    settings["n_trials"] = int(resolve_setting(args.n_trials, config, "n_trials", 20))
    settings["top_n_backtest"] = int(resolve_setting(args.top_n_backtest, config, "top_n_backtest", 5))
    settings["seed"] = int(resolve_setting(args.seed, config, "seed", 20260409))
    settings["run_tag"] = resolve_setting(args.run_tag, config, "run_tag", datetime.now().strftime("auto%Y%m%d_%H%M%S"))
    settings["screen_segment"] = resolve_setting(args.screen_segment, config, "screen_segment", "valid")
    settings["screen_ranking_metric"] = resolve_setting(args.screen_ranking_metric, config, "screen_ranking_metric", DEFAULT_SCREEN_METRIC)
    settings["ranking_metric"] = resolve_setting(args.ranking_metric, config, "ranking_metric", DEFAULT_RANKING_METRIC)
    settings["topk_values"] = list(resolve_setting(args.topk_values, config, "topk_values", [50]))
    settings["n_drop_values"] = list(resolve_setting(args.n_drop_values, config, "n_drop_values", [5]))
    settings["strategy_values"] = list(resolve_setting(args.strategy_values, config, "strategy_values", ["bucket"]))
    settings["deal_price_values"] = list(resolve_setting(args.deal_price_values, config, "deal_price_values", ["close"]))
    settings["rebalance_values"] = list(resolve_setting(args.rebalance_values, config, "rebalance_values", ["day"]))
    settings["limit_tplus_values"] = normalize_bool_list(resolve_setting(args.limit_tplus_values, config, "limit_tplus_values", [False]))
    settings["output_root"] = Path(resolve_setting(args.output_root, config, "output_root", DEFAULT_OUTPUT_ROOT))
    settings["promote_best"] = resolve_bool_setting(args.promote_best, config, "promote_best", False)
    settings["clean_best_run"] = resolve_bool_setting(args.clean_best_run, config, "clean_best_run", False)
    settings["screen_only"] = resolve_bool_setting(args.screen_only, config, "screen_only", False)
    settings["minimize_metric"] = resolve_bool_setting(args.minimize_metric, config, "minimize_metric", False)
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
    qlib.init(provider_uri=str(PROVIDER_URI), region=REGION)
    LOGGER.info("Universe size: %d symbols", len(UNIVERSE))

    rng = random.Random(settings["seed"])
    model_trials = generate_model_trials(settings["model_search_space"], settings["n_trials"], rng)
    LOGGER.info("Generated %d model trials for combo %s", len(model_trials), settings["combo"])

    screen_results: List[Dict[str, Any]] = []
    for index, model_params in enumerate(model_trials, start=1):
        LOGGER.info("Screen trial %d/%d with params %s", index, len(model_trials), model_params)
        row = run_screen_trial(settings["combo"], model_params, settings["screen_segment"])
        row["trial_index"] = index
        screen_results.append(row)

    screen_results = sort_rows(screen_results, settings["screen_ranking_metric"], minimize=False)
    save_rows_csv(screen_results, output_dir / "screen_trials.csv")
    save_json(screen_results, output_dir / "screen_trials.json")

    if settings["screen_only"] or settings["top_n_backtest"] <= 0:
        LOGGER.info("Screen-only mode enabled; skipping full backtests")
        return 0

    top_models = screen_results[: min(settings["top_n_backtest"], len(screen_results))]
    save_rows_csv(top_models, output_dir / "top_screen_models.csv")
    strategy_variants = build_strategy_variants(settings)
    save_json(strategy_variants, output_dir / "strategy_variants.json")
    LOGGER.info("Running full backtests for %d screened models across %d strategy variants", len(top_models), len(strategy_variants))

    backtest_results: List[Dict[str, Any]] = []
    for model_rank, screen_row in enumerate(top_models, start=1):
        model_params = screen_row["model_params"]
        for strategy_cfg in strategy_variants:
            LOGGER.info(
                "Backtest model rank %d with strategy topk=%s n_drop=%s strategy=%s rebalance=%s deal_price=%s limit_tplus=%s",
                model_rank,
                strategy_cfg["topk"],
                strategy_cfg["n_drop"],
                strategy_cfg["strategy"],
                strategy_cfg["rebalance"],
                strategy_cfg["deal_price"],
                strategy_cfg["limit_tplus"],
            )
            result = run_backtest_candidate(
                combo_name=settings["combo"],
                run_tag=settings["run_tag"],
                model_rank=model_rank,
                model_params=model_params,
                strategy_cfg=strategy_cfg,
                ranking_metric=settings["ranking_metric"],
            )
            result["screen_metrics"] = {
                "trial_index": screen_row["trial_index"],
                "ic_mean": screen_row["ic_mean"],
                "ic_std": screen_row["ic_std"],
                "ic_days": screen_row["ic_days"],
                "ic_ir": screen_row["ic_ir"],
            }
            backtest_results.append(result)

    backtest_results = sort_backtest_rows(backtest_results, settings["ranking_metric"], minimize=settings["minimize_metric"])
    save_rows_csv(backtest_results, output_dir / "backtest_results.csv")
    save_json(backtest_results, output_dir / "backtest_results.json")

    if not backtest_results:
        raise SystemExit("No backtest results were produced")

    best_result = backtest_results[0]
    save_json(best_result, output_dir / "best_result.json")
    LOGGER.info("Best result: %s = %.6f (%s)", settings["ranking_metric"], best_result["ranking_value"], best_result["output_name"])

    if settings["promote_best"]:
        promote_best_output(best_result["output_name"], clean_best_run=settings["clean_best_run"])

    return 0
