from __future__ import annotations

import itertools
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import mlflow
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from qlib_tw.data_layout import build_exp_manager_config, resolve_provider_uri, resolve_raw_data_dir, resolve_workspace_path
from qlib_tw.research.builders import apply_strategy_overrides, build_port_analysis_config, build_task_config
from qlib_tw.research.get_data_tai import run_collect
from qlib_tw.research.settings import BENCHMARK, COMBO_CONFIGS, PROVIDER_URI, REGION, WORK_DIR, load_full_universe


LOGGER = logging.getLogger("qlib_tw.research.backtest_search")


def _resolve_path(value: str | Path | None) -> Path | None:
    return resolve_workspace_path(value)


def _save_json(data: Any, path: Path) -> None:
    def default_serializer(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Object of type {type(value)} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default_serializer), encoding="utf-8")


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
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


def _save_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    pd.DataFrame([_flatten_row(row) for row in rows]).to_csv(path, index=False)


def _ensure_no_active_mlflow_run() -> None:
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _read_calendar_dates(provider_uri: Path) -> list[pd.Timestamp]:
    day_file = provider_uri / "calendars" / "day.txt"
    lines = [line.strip() for line in day_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Calendar file is empty: {day_file}")
    return [pd.Timestamp(line).normalize() for line in lines]


def _latest_calendar_date(provider_uri: Path) -> pd.Timestamp:
    return _read_calendar_dates(provider_uri)[-1]


def _latest_trade_date_on_or_before(provider_uri: Path, target_date: pd.Timestamp) -> pd.Timestamp:
    target_date = pd.Timestamp(target_date).normalize()
    eligible_dates = [dt for dt in _read_calendar_dates(provider_uri) if dt <= target_date]
    if not eligible_dates:
        raise RuntimeError(
            f"No available trade date on or before {target_date.strftime('%Y-%m-%d')} under {provider_uri}"
        )
    return eligible_dates[-1]


def _build_calendar_overlay_provider(base_provider_uri: Path, overlay_uri: Path) -> Path:
    base_calendar = _read_calendar_dates(base_provider_uri)
    next_business_day = (base_calendar[-1] + pd.tseries.offsets.BDay(1)).normalize()
    overlay_uri.mkdir(parents=True, exist_ok=True)

    for dirname in ("features", "instruments"):
        source = (base_provider_uri / dirname).resolve()
        target = overlay_uri / dirname
        if target.exists() or target.is_symlink():
            if target.is_symlink() and target.resolve() == source:
                continue
            _reset_path(target)
        target.symlink_to(source, target_is_directory=True)

    calendars_dir = overlay_uri / "calendars"
    calendars_dir.mkdir(parents=True, exist_ok=True)
    overlay_lines = [dt.strftime("%Y-%m-%d") for dt in base_calendar]
    next_label = next_business_day.strftime("%Y-%m-%d")
    if overlay_lines[-1] != next_label:
        overlay_lines.append(next_label)
    (calendars_dir / "day.txt").write_text("\n".join(overlay_lines) + "\n", encoding="utf-8")
    return overlay_uri


def _extract_summary_metrics(recorder) -> Dict[str, Any]:
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    try:
        indicator_summary_df = recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
    except (FileNotFoundError, KeyError):
        indicator_summary_df = None
    try:
        indicator_daily_df = recorder.load_object("portfolio_analysis/indicators_normal_1day.pkl")
    except (FileNotFoundError, KeyError):
        indicator_daily_df = None

    metrics: Dict[str, Any] = {}
    metrics["Strategy cumulative return"] = float((1 + report_df["return"]).prod() - 1)
    metrics["Benchmark cumulative return"] = float((1 + report_df["bench"]).prod() - 1)
    metrics["Trading days"] = int((report_df["total_turnover"] > 0).sum()) if "total_turnover" in report_df.columns else 0

    if isinstance(analysis_df, pd.DataFrame) and "risk" in analysis_df.columns:
        risk_series = analysis_df["risk"]
        if isinstance(risk_series, pd.Series):
            for (category, metric), value in risk_series.items():
                if pd.notna(value):
                    metrics[f"risk.{category}.{metric}"] = float(value)
    if indicator_summary_df is not None and "value" in indicator_summary_df.columns:
        for idx, value in indicator_summary_df["value"].items():
            if pd.notna(value):
                metrics[f"indicator_summary.{idx}"] = float(value)
    if indicator_daily_df is not None:
        numeric_cols = indicator_daily_df.select_dtypes(include="number")
        if not numeric_cols.empty:
            stats = numeric_cols.mean(numeric_only=True)
            for col, value in stats.items():
                if pd.notna(value):
                    metrics[f"indicator_daily_mean.{col}"] = float(value)
    return metrics


def _extract_metric(metrics: Dict[str, Any], metric_name: str) -> float:
    value = metrics.get(metric_name)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _sort_results(rows: List[Dict[str, Any]], metric_name: str, minimize: bool = False) -> List[Dict[str, Any]]:
    valid = [row for row in rows if not pd.isna(row.get("ranking_value", float("nan")))]
    invalid = [row for row in rows if pd.isna(row.get("ranking_value", float("nan")))]
    valid.sort(key=lambda row: float(row["ranking_value"]), reverse=not minimize)
    return valid + invalid


def _risk_degree_slug(value: float) -> str:
    return str(value).replace(".", "p")


@dataclass(frozen=True)
class BacktestSearchConfig:
    name: str
    combo: str
    train_experiment: str
    train_recorder_id: str
    target_date: str
    train_metadata: Path | None = None
    backtest_start: str = "2025-07-01"
    data_start: str = "2018-01-01"
    data_refresh_start: str = "2018-01-01"
    provider_uri: Path = PROVIDER_URI.resolve()
    output_root: Path = WORK_DIR / "outputs" / "backtest_search"
    ranking_metric: str = "risk.excess_return_with_cost.annualized_return"
    minimize_metric: bool = False
    refresh_data: bool = False
    account: float = 1_000_000.0
    open_cost: float = 0.00092625
    close_cost: float = 0.00392625
    min_cost: float = 20.0
    odd_lot_min_cost: float = 1.0
    board_lot_size: int = 1000
    trade_unit: int = 1
    topk_values: List[int] | None = None
    n_drop_values: List[int] | None = None
    strategy_values: List[str] | None = None
    deal_price_values: List[str] | None = None
    rebalance_values: List[str] | None = None
    risk_degree_values: List[float] | None = None
    settlement_lag_values: List[int] | None = None
    region: str = REGION

    @classmethod
    def from_json(cls, path: str | Path) -> "BacktestSearchConfig":
        config_path = _resolve_path(path)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(f"Backtest search config not found: {path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        train_metadata = _resolve_path(payload.get("train_metadata"))
        metadata_payload = {}
        if train_metadata is not None:
            if not train_metadata.exists():
                raise FileNotFoundError(f"Train metadata file not found: {train_metadata}")
            metadata_payload = json.loads(train_metadata.read_text(encoding="utf-8"))
        train_experiment = str(payload.get("train_experiment") or metadata_payload.get("train_experiment") or "")
        train_recorder_id = str(payload.get("train_recorder_id") or metadata_payload.get("train_recorder_id") or "")
        if not train_experiment or not train_recorder_id:
            raise ValueError("Backtest search config requires train_experiment/train_recorder_id or train_metadata")
        provider_uri = resolve_provider_uri(_resolve_path(payload.get("provider_uri")) or PROVIDER_URI.resolve())
        output_root = _resolve_path(payload.get("output_root")) or (WORK_DIR / "outputs" / "backtest_search")
        combo = str(payload["combo"])
        if combo not in COMBO_CONFIGS:
            raise ValueError(f"Unsupported combo in backtest search config: {combo}")
        return cls(
            name=str(payload["name"]),
            combo=combo,
            train_experiment=train_experiment,
            train_recorder_id=train_recorder_id,
            train_metadata=train_metadata,
            target_date=str(payload["target_date"]),
            backtest_start=str(payload.get("backtest_start", "2025-07-01")),
            data_start=str(payload.get("data_start", "2018-01-01")),
            data_refresh_start=str(payload.get("data_refresh_start", payload.get("data_start", "2018-01-01"))),
            provider_uri=provider_uri.resolve(),
            output_root=output_root,
            ranking_metric=str(payload.get("ranking_metric", "risk.excess_return_with_cost.annualized_return")),
            minimize_metric=bool(payload.get("minimize_metric", False)),
            refresh_data=bool(payload.get("refresh_data", False)),
            account=float(payload.get("account", 1_000_000.0)),
            open_cost=float(payload.get("open_cost", 0.00092625)),
            close_cost=float(payload.get("close_cost", 0.00392625)),
            min_cost=float(payload.get("min_cost", 20.0)),
            odd_lot_min_cost=float(payload.get("odd_lot_min_cost", 1.0)),
            board_lot_size=int(payload.get("board_lot_size", 1000)),
            trade_unit=int(payload.get("trade_unit", 1)),
            topk_values=[int(v) for v in payload.get("topk_values", [])] or None,
            n_drop_values=[int(v) for v in payload.get("n_drop_values", [])] or None,
            strategy_values=[str(v) for v in payload.get("strategy_values", [])] or None,
            deal_price_values=[str(v) for v in payload.get("deal_price_values", [])] or None,
            rebalance_values=[str(v) for v in payload.get("rebalance_values", [])] or None,
            risk_degree_values=[float(v) for v in payload.get("risk_degree_values", [])] or None,
            settlement_lag_values=[int(v) for v in payload.get("settlement_lag_values", [])] or None,
            region=str(payload.get("region", REGION)),
        )

    @property
    def combo_spec(self) -> Dict[str, object]:
        return COMBO_CONFIGS[self.combo]

    @property
    def resolved_provider_uri(self) -> Path:
        return self.provider_uri.resolve()

    @property
    def resolved_output_root(self) -> Path:
        return (self.output_root / self.name).resolve()

    @property
    def resolved_raw_dir(self) -> Path:
        return resolve_raw_data_dir()


def build_strategy_variants(config: BacktestSearchConfig) -> List[Dict[str, Any]]:
    topk_values = config.topk_values or [50]
    n_drop_values = config.n_drop_values or [5]
    strategy_values = config.strategy_values or ["bucket"]
    deal_price_values = config.deal_price_values or ["close"]
    rebalance_values = config.rebalance_values or ["day"]
    risk_degree_values = config.risk_degree_values or [0.95]
    settlement_lag_values = config.settlement_lag_values or [2]

    variants: List[Dict[str, Any]] = []
    for topk, n_drop, strategy, deal_price, rebalance, risk_degree, settlement_lag in itertools.product(
        topk_values,
        n_drop_values,
        strategy_values,
        deal_price_values,
        rebalance_values,
        risk_degree_values,
        settlement_lag_values,
    ):
        variants.append(
            {
                "topk": int(topk),
                "n_drop": int(n_drop),
                "strategy": str(strategy),
                "deal_price": str(deal_price),
                "rebalance": str(rebalance),
                "risk_degree": float(risk_degree),
                "settlement_lag": int(settlement_lag),
            }
        )
    return variants


def _variant_slug(variant: Dict[str, Any]) -> str:
    return "_".join(
        [
            str(variant["strategy"]),
            f"topk{variant['topk']}",
            f"ndrop{variant['n_drop']}",
            f"risk{_risk_degree_slug(float(variant['risk_degree']))}",
            str(variant["rebalance"]),
            str(variant["deal_price"]),
            f"tplus{variant['settlement_lag']}",
        ]
    )


def _refresh_provider_data(config: BacktestSearchConfig, target_date: pd.Timestamp) -> int:
    LOGGER.info(
        "Refreshing shared provider %s from %s to %s",
        config.resolved_provider_uri,
        config.data_refresh_start,
        target_date.strftime("%Y-%m-%d"),
    )
    return run_collect(
        start=config.data_refresh_start,
        end=target_date.strftime("%Y-%m-%d"),
        target_dir=config.resolved_provider_uri,
        raw_dir=config.resolved_raw_dir,
    )


def _init_qlib(provider_uri: Path, region: str) -> None:
    LOGGER.info("Initialize Qlib for backtest search, provider uri: %s", provider_uri)
    qlib.init(provider_uri=str(provider_uri), region=region, exp_manager=build_exp_manager_config())


def _build_task_config(config: BacktestSearchConfig, provider_uri: Path, effective_end: pd.Timestamp) -> dict:
    spec = config.combo_spec
    task_cfg = build_task_config(
        handler_key=spec["handler"],
        model_key=spec["model"],
        instruments=[],
        max_instruments=spec.get("max_instruments"),
        infer_processors=spec.get("infer_processors"),
    )
    all_codes = load_full_universe(provider_uri, benchmark=BENCHMARK)
    universe = [code for code in all_codes if not code.startswith("^")]
    task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = universe
    handler_kwargs = task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs["start_time"] = config.data_start
    handler_kwargs["end_time"] = effective_end.strftime("%Y-%m-%d")
    fit_end_time = pd.Timestamp(handler_kwargs["fit_end_time"])
    if fit_end_time > effective_end:
        handler_kwargs["fit_end_time"] = effective_end.strftime("%Y-%m-%d")
    task_cfg["dataset"]["kwargs"]["segments"]["test"] = (config.backtest_start, effective_end.strftime("%Y-%m-%d"))
    return task_cfg


def _build_port_config(config: BacktestSearchConfig, task_cfg: dict, effective_end: pd.Timestamp, variant: Dict[str, Any]) -> dict:
    port_config = build_port_analysis_config()
    port_config["backtest"]["start_time"] = config.backtest_start
    port_config["backtest"]["end_time"] = effective_end.strftime("%Y-%m-%d")
    port_config["backtest"]["account"] = config.account
    port_config["backtest"]["exchange_kwargs"]["open_cost"] = config.open_cost
    port_config["backtest"]["exchange_kwargs"]["close_cost"] = config.close_cost
    port_config["backtest"]["exchange_kwargs"]["min_cost"] = config.min_cost
    port_config["backtest"]["exchange_kwargs"]["trade_unit"] = config.trade_unit
    apply_strategy_overrides(
        port_config,
        task_cfg,
        n_drop_override=int(variant["n_drop"]),
        topk_override=int(variant["topk"]),
        rebalance=str(variant["rebalance"]),
        strategy_choice=str(variant["strategy"]),
        deal_price=str(variant["deal_price"]),
    )
    port_config["strategy"]["kwargs"]["risk_degree"] = float(variant["risk_degree"])
    exchange_kwargs = port_config["backtest"]["exchange_kwargs"]["exchange"]["kwargs"]
    exchange_kwargs["odd_lot_min_cost"] = config.odd_lot_min_cost
    exchange_kwargs["board_lot_size"] = config.board_lot_size
    exchange_kwargs["settlement_lag"] = int(variant["settlement_lag"])
    return port_config


def run_backtest_search(config: BacktestSearchConfig) -> Dict[str, Any]:
    target_date = pd.Timestamp(config.target_date).normalize()
    output_root = config.resolved_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if config.refresh_data:
        exit_code = _refresh_provider_data(config, target_date)
        if exit_code != 0:
            raise RuntimeError(f"Data refresh failed with exit code {exit_code}")

    base_provider_uri = config.resolved_provider_uri
    effective_end = _latest_trade_date_on_or_before(base_provider_uri, target_date)
    active_provider_uri = base_provider_uri
    if effective_end == _latest_calendar_date(base_provider_uri):
        overlay_uri = output_root / "_provider_overlay"
        active_provider_uri = _build_calendar_overlay_provider(base_provider_uri, overlay_uri)
        LOGGER.info(
            "Using calendar overlay provider %s to expose next-session boundary after %s",
            active_provider_uri,
            effective_end.strftime("%Y-%m-%d"),
        )

    _init_qlib(active_provider_uri, config.region)
    task_cfg = _build_task_config(config, active_provider_uri, effective_end)
    dataset = init_instance_by_config(task_cfg["dataset"])
    train_recorder = R.get_recorder(experiment_name=config.train_experiment, recorder_id=config.train_recorder_id)
    trained_model = train_recorder.load_object("trained_model")
    pred_df = trained_model.predict(dataset, segment="test")
    if isinstance(pred_df, pd.Series):
        pred_df = pred_df.to_frame("score")
    label_df = SignalRecord.generate_label(dataset)

    variants = build_strategy_variants(config)
    _save_json(
        {
            "config": _flatten_row(asdict(config)),
            "active_provider_uri": str(active_provider_uri),
            "effective_end": effective_end.strftime("%Y-%m-%d"),
            "variant_count": len(variants),
        },
        output_root / "resolved_config.json",
    )
    _save_json(variants, output_root / "strategy_variants.json")

    experiment_name = f"backtest_search_{config.name}"
    results: List[Dict[str, Any]] = []

    for index, variant in enumerate(variants, start=1):
        variant_slug = _variant_slug(variant)
        LOGGER.info("[%d/%d] Backtest %s", index, len(variants), variant_slug)
        port_config = _build_port_config(config, task_cfg, effective_end, variant)
        strategy_kwargs = port_config["strategy"]["kwargs"]
        strategy_kwargs.pop("model", None)
        strategy_kwargs.pop("dataset", None)
        strategy_kwargs["signal"] = "<PRED>"

        _ensure_no_active_mlflow_run()
        with R.start(experiment_name=experiment_name):
            current_recorder = R.get_recorder()
            current_recorder.set_tags(
                search_name=config.name,
                variant_slug=variant_slug,
                requested_target_date=target_date.strftime("%Y-%m-%d"),
            )
            current_recorder.log_params(
                combo=config.combo,
                train_experiment=config.train_experiment,
                train_recorder_id=config.train_recorder_id,
                backtest_start=config.backtest_start,
                effective_end=effective_end.strftime("%Y-%m-%d"),
                account=config.account,
                open_cost=config.open_cost,
                close_cost=config.close_cost,
                min_cost=config.min_cost,
                odd_lot_min_cost=config.odd_lot_min_cost,
                board_lot_size=config.board_lot_size,
                trade_unit=config.trade_unit,
                **variant,
            )
            current_recorder.save_objects(**{"pred.pkl": pred_df, "label.pkl": label_df})
            port_rec = PortAnaRecord(current_recorder, port_config, "day")
            port_rec.generate()
            recorder_id = current_recorder.id

        recorder = R.get_recorder(experiment_name=experiment_name, recorder_id=recorder_id)
        summary_metrics = _extract_summary_metrics(recorder)
        ranking_value = _extract_metric(summary_metrics, config.ranking_metric)
        results.append(
            {
                "variant_index": index,
                "variant_slug": variant_slug,
                "backtest_experiment": experiment_name,
                "backtest_recorder_id": recorder_id,
                "ranking_metric": config.ranking_metric,
                "ranking_value": ranking_value,
                "strategy_params": dict(variant),
                "summary_metrics": summary_metrics,
            }
        )

    _ensure_no_active_mlflow_run()
    results = _sort_results(results, config.ranking_metric, minimize=config.minimize_metric)
    _save_rows_csv(results, output_root / "backtest_search_results.csv")
    _save_json(results, output_root / "backtest_search_results.json")

    if not results:
        raise RuntimeError("No backtest-search results were produced")

    best_result = results[0]
    _save_json(best_result, output_root / "best_result.json")
    _save_json(
        {
            "name": config.name,
            "combo": config.combo,
            "train_experiment": config.train_experiment,
            "train_recorder_id": config.train_recorder_id,
            "target_date": target_date.strftime("%Y-%m-%d"),
            "effective_end": effective_end.strftime("%Y-%m-%d"),
            "active_provider_uri": str(active_provider_uri),
            "variant_count": len(results),
            "ranking_metric": config.ranking_metric,
            "best_variant_slug": best_result["variant_slug"],
            "best_ranking_value": best_result["ranking_value"],
            "best_strategy_params": best_result["strategy_params"],
            "best_summary_metrics": best_result["summary_metrics"],
        },
        output_root / "backtest_search_summary.json",
    )
    return {
        "output_root": output_root,
        "best_result": best_result,
    }
