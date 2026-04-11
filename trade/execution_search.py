from __future__ import annotations

import itertools
import json
import logging
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List

import mlflow
import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from research.settings import WORK_DIR
from scripts.research.Get_data_Tai import run_collect
from trade.config import PaperTradingProfile
from trade.replay import _dynamic_port_config, _dynamic_task_config, latest_calendar_date, latest_trade_date_on_or_before


LOGGER = logging.getLogger("paper_trading.execution_search")


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = WORK_DIR / path
    return path.resolve()


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


def _parse_bool_token(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unsupported boolean token: {value}")


def _ensure_no_active_mlflow_run() -> None:
    while mlflow.active_run() is not None:
        mlflow.end_run()


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _build_calendar_overlay_provider(base_provider_uri: Path, overlay_uri: Path) -> Path:
    day_file = base_provider_uri / "calendars" / "day.txt"
    lines = [line.strip() for line in day_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Calendar file is empty: {day_file}")
    base_calendar = [pd.Timestamp(line).normalize() for line in lines]
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
    if overlay_lines[-1] != next_business_day.strftime("%Y-%m-%d"):
        overlay_lines.append(next_business_day.strftime("%Y-%m-%d"))
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


def _profile_payload(profile: PaperTradingProfile) -> Dict[str, Any]:
    payload = asdict(profile)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return payload


@dataclass(frozen=True)
class ExecutionGridConfig:
    name: str
    paper_profile: Path
    target_date: str
    output_root: Path = WORK_DIR / "outputs" / "execution_grid"
    ranking_metric: str = "risk.excess_return_with_cost.annualized_return"
    minimize_metric: bool = False
    refresh_data: bool = False
    topk_values: List[int] | None = None
    n_drop_values: List[int] | None = None
    strategy_values: List[str] | None = None
    deal_price_values: List[str] | None = None
    rebalance_values: List[str] | None = None
    limit_tplus_values: List[bool] | None = None
    risk_degree_values: List[float] | None = None
    limit_slippage_values: List[float] | None = None
    settlement_lag_values: List[int] | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExecutionGridConfig":
        config_path = _resolve_path(path)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(f"Execution grid config not found: {path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        paper_profile = _resolve_path(payload.get("paper_profile"))
        if paper_profile is None:
            raise ValueError("paper_profile is required")
        output_root = _resolve_path(payload.get("output_root")) or (WORK_DIR / "outputs" / "execution_grid")
        return cls(
            name=str(payload["name"]),
            paper_profile=paper_profile,
            target_date=str(payload["target_date"]),
            output_root=output_root,
            ranking_metric=str(payload.get("ranking_metric", "risk.excess_return_with_cost.annualized_return")),
            minimize_metric=bool(payload.get("minimize_metric", False)),
            refresh_data=bool(payload.get("refresh_data", False)),
            topk_values=[int(v) for v in payload.get("topk_values", [])] or None,
            n_drop_values=[int(v) for v in payload.get("n_drop_values", [])] or None,
            strategy_values=[str(v) for v in payload.get("strategy_values", [])] or None,
            deal_price_values=[str(v) for v in payload.get("deal_price_values", [])] or None,
            rebalance_values=[str(v) for v in payload.get("rebalance_values", [])] or None,
            limit_tplus_values=[_parse_bool_token(v) for v in payload.get("limit_tplus_values", [])] or None,
            risk_degree_values=[float(v) for v in payload.get("risk_degree_values", [])] or None,
            limit_slippage_values=[float(v) for v in payload.get("limit_slippage_values", [])] or None,
            settlement_lag_values=[int(v) for v in payload.get("settlement_lag_values", [])] or None,
        )

    @property
    def resolved_output_root(self) -> Path:
        return (self.output_root / self.name).resolve()


def build_strategy_variants(config: ExecutionGridConfig, base_profile: PaperTradingProfile) -> List[Dict[str, Any]]:
    topk_values = config.topk_values or [base_profile.topk]
    n_drop_values = config.n_drop_values or [base_profile.n_drop]
    strategy_values = config.strategy_values or [base_profile.strategy]
    deal_price_values = config.deal_price_values or [base_profile.deal_price]
    rebalance_values = config.rebalance_values or [base_profile.rebalance]
    limit_tplus_values = config.limit_tplus_values or [base_profile.limit_tplus]
    risk_degree_values = config.risk_degree_values or [base_profile.risk_degree]
    limit_slippage_values = config.limit_slippage_values or [base_profile.limit_slippage]
    settlement_lag_values = config.settlement_lag_values or [base_profile.settlement_lag]

    variants: List[Dict[str, Any]] = []
    for topk, n_drop, strategy, deal_price, rebalance, limit_tplus, risk_degree, limit_slippage, settlement_lag in itertools.product(
        topk_values,
        n_drop_values,
        strategy_values,
        deal_price_values,
        rebalance_values,
        limit_tplus_values,
        risk_degree_values,
        limit_slippage_values,
        settlement_lag_values,
    ):
        variants.append(
            {
                "topk": int(topk),
                "n_drop": int(n_drop),
                "strategy": str(strategy),
                "deal_price": str(deal_price),
                "rebalance": str(rebalance),
                "limit_tplus": bool(limit_tplus),
                "risk_degree": float(risk_degree),
                "limit_slippage": float(limit_slippage),
                "settlement_lag": int(settlement_lag),
            }
        )
    return variants


def _build_variant_profile(base_profile: PaperTradingProfile, variant: Dict[str, Any]) -> PaperTradingProfile:
    return replace(
        base_profile,
        topk=int(variant["topk"]),
        n_drop=int(variant["n_drop"]),
        strategy=str(variant["strategy"]),
        deal_price=str(variant["deal_price"]),
        rebalance=str(variant["rebalance"]),
        limit_tplus=bool(variant["limit_tplus"]),
        risk_degree=float(variant["risk_degree"]),
        limit_slippage=float(variant["limit_slippage"]),
        settlement_lag=int(variant["settlement_lag"]),
    )


def _variant_slug(variant: Dict[str, Any]) -> str:
    parts = [
        str(variant["strategy"]),
        f"topk{variant['topk']}",
        f"ndrop{variant['n_drop']}",
        f"risk{_risk_degree_slug(float(variant['risk_degree']))}",
        str(variant["rebalance"]),
        str(variant["deal_price"]),
    ]
    if variant["limit_tplus"]:
        parts.append("tplus")
    return "_".join(parts)


def _refresh_provider_data(profile: PaperTradingProfile, target_date: pd.Timestamp) -> int:
    LOGGER.info(
        "Refreshing shared provider %s from %s to %s",
        profile.resolved_provider_uri,
        profile.data_refresh_start,
        target_date.strftime("%Y-%m-%d"),
    )
    return run_collect(
        start=profile.data_refresh_start,
        end=target_date.strftime("%Y-%m-%d"),
        target_dir=profile.resolved_provider_uri,
        tmp_dir=profile.resolved_raw_dir,
    )


def _init_qlib(provider_uri: Path, region: str) -> None:
    LOGGER.info("Initialize Qlib for execution grid, provider uri: %s", provider_uri)
    qlib.init(provider_uri=str(provider_uri), region=region)


def run_execution_grid(config: ExecutionGridConfig) -> Dict[str, Any]:
    base_profile = PaperTradingProfile.from_json(config.paper_profile)
    target_date = pd.Timestamp(config.target_date).normalize()
    output_root = config.resolved_output_root
    output_root.mkdir(parents=True, exist_ok=True)

    if config.refresh_data:
        exit_code = _refresh_provider_data(base_profile, target_date)
        if exit_code != 0:
            raise RuntimeError(f"Data refresh failed with exit code {exit_code}")

    base_provider_uri = base_profile.resolved_provider_uri
    effective_end = latest_trade_date_on_or_before(base_provider_uri, target_date)
    active_provider_uri = base_provider_uri
    if effective_end == latest_calendar_date(base_provider_uri):
        overlay_uri = output_root / "_provider_overlay"
        active_provider_uri = _build_calendar_overlay_provider(base_provider_uri, overlay_uri)
        LOGGER.info(
            "Using calendar overlay provider %s to expose next-session boundary after %s",
            active_provider_uri,
            effective_end.strftime("%Y-%m-%d"),
        )

    _init_qlib(active_provider_uri, base_profile.region)
    task_cfg = _dynamic_task_config(base_profile, active_provider_uri, effective_end)
    dataset = init_instance_by_config(task_cfg["dataset"])
    train_recorder = R.get_recorder(experiment_name=base_profile.train_experiment, recorder_id=base_profile.train_recorder_id)
    trained_model = train_recorder.load_object("trained_model")
    pred_df = trained_model.predict(dataset, segment="test")
    if isinstance(pred_df, pd.Series):
        pred_df = pred_df.to_frame("score")
    label_df = SignalRecord.generate_label(dataset)

    variants = build_strategy_variants(config, base_profile)
    _save_json(
        {
            "config": _flatten_row(asdict(config)),
            "base_profile": _profile_payload(base_profile),
            "active_provider_uri": str(active_provider_uri),
            "effective_end": effective_end.strftime("%Y-%m-%d"),
            "variant_count": len(variants),
        },
        output_root / "resolved_config.json",
    )
    _save_json(variants, output_root / "strategy_variants.json")

    experiment_name = f"execution_grid_{config.name}"
    results: List[Dict[str, Any]] = []

    for index, variant in enumerate(variants, start=1):
        variant_profile = _build_variant_profile(base_profile, variant)
        variant_slug = _variant_slug(variant)
        LOGGER.info(
            "[%d/%d] Backtest %s",
            index,
            len(variants),
            variant_slug,
        )
        port_config = _dynamic_port_config(variant_profile, task_cfg, effective_end)
        strategy_kwargs = port_config["strategy"]["kwargs"]
        strategy_kwargs.pop("model", None)
        strategy_kwargs.pop("dataset", None)
        strategy_kwargs["signal"] = "<PRED>"

        _ensure_no_active_mlflow_run()
        with R.start(experiment_name=experiment_name):
            current_recorder = R.get_recorder()
            current_recorder.set_tags(
                grid_name=config.name,
                variant_slug=variant_slug,
                requested_target_date=target_date.strftime("%Y-%m-%d"),
            )
            current_recorder.log_params(
                combo=base_profile.combo,
                train_experiment=base_profile.train_experiment,
                train_recorder_id=base_profile.train_recorder_id,
                backtest_start=variant_profile.backtest_start,
                effective_end=effective_end.strftime("%Y-%m-%d"),
                account=variant_profile.account,
                open_cost=variant_profile.open_cost,
                close_cost=variant_profile.close_cost,
                min_cost=variant_profile.min_cost,
                odd_lot_min_cost=variant_profile.odd_lot_min_cost,
                board_lot_size=variant_profile.board_lot_size,
                trade_unit=variant_profile.trade_unit,
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
    _save_rows_csv(results, output_root / "grid_results_ranked.csv")
    _save_json(results, output_root / "grid_results_ranked.json")

    if not results:
        raise RuntimeError("No execution-grid results were produced")

    best_result = results[0]
    best_profile = _build_variant_profile(base_profile, best_result["strategy_params"])
    _save_json(best_result, output_root / "best_result.json")
    _save_json(_profile_payload(best_profile), output_root / "best_paper_profile.json")
    _save_json(
        {
            "name": config.name,
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
        output_root / "scan_summary.json",
    )
    return {
        "output_root": output_root,
        "best_result": best_result,
        "best_profile": best_profile,
    }
