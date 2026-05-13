from __future__ import annotations

import bisect
import itertools
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from qlib_tw.data_layout import build_exp_manager_config, resolve_provider_uri, resolve_workspace_path
from qlib_tw.research.builders import apply_strategy_overrides, build_port_analysis_config, build_task_config
from qlib_tw.research.ic import calc_ic_metrics
from qlib_tw.research.paths import WorkflowPaths
from qlib_tw.research.settings import BENCHMARK, COMBO_CONFIGS, PROVIDER_URI, REGION, WORK_DIR, load_full_universe


LOGGER = logging.getLogger("qlib_tw.research.rolling")


@dataclass(frozen=True)
class RollingSplit:
    round_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    trade_start: pd.Timestamp
    trade_end: pd.Timestamp

    def segments(self) -> dict[str, tuple[str, str]]:
        return {
            "train": (_date_str(self.train_start), _date_str(self.train_end)),
            "valid": (_date_str(self.valid_start), _date_str(self.valid_end)),
            "test": (_date_str(self.trade_start), _date_str(self.trade_end)),
        }

    def to_record(self) -> dict[str, Any]:
        return {
            "round_id": self.round_id,
            "train_start": _date_str(self.train_start),
            "train_end": _date_str(self.train_end),
            "valid_start": _date_str(self.valid_start),
            "valid_end": _date_str(self.valid_end),
            "trade_start": _date_str(self.trade_start),
            "trade_end": _date_str(self.trade_end),
        }


@dataclass(frozen=True)
class RollingSplitPlan:
    splits: list[RollingSplit]
    skipped_rounds: list[dict[str, Any]]


@dataclass(frozen=True)
class RollingWalkForwardConfig:
    name: str
    combo: str
    start_date: str
    end_date: str
    train_years: int = 3
    valid_quarters: int = 2
    trade_quarters: int = 1
    step_quarters: int = 1
    embargo_days: int = 2
    provider_uri: Path = PROVIDER_URI.resolve()
    output_root: Path = WORK_DIR / "outputs" / "rolling_walk_forward"
    topk: int = 50
    n_drop: int = 5
    strategy: str = "bucket"
    deal_price: str = "close"
    rebalance: str = "day"
    risk_degree: float = 0.95
    settlement_lag: int = 2
    account: float = 10_000_000.0
    open_cost: float = 0.00092625
    close_cost: float = 0.00392625
    min_cost: float = 20.0
    odd_lot_min_cost: float = 1.0
    board_lot_size: int = 1000
    trade_unit: int = 1
    max_instruments: int | None = None
    model_kwargs: Dict[str, Any] | None = None
    strategy_mode: str = "fixed"
    strategy_search: Dict[str, Any] | None = None
    region: str = REGION

    @classmethod
    def from_json(cls, path: str | Path) -> "RollingWalkForwardConfig":
        config_path = resolve_workspace_path(path)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(f"Rolling walk-forward config not found: {path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        provider_uri = resolve_provider_uri(resolve_workspace_path(payload.get("provider_uri")) or PROVIDER_URI.resolve())
        output_root = resolve_workspace_path(payload.get("output_root")) or (WORK_DIR / "outputs" / "rolling_walk_forward")
        combo = str(payload["combo"])
        if combo not in COMBO_CONFIGS:
            raise ValueError(f"Unsupported combo in rolling config: {combo}")
        return cls(
            name=str(payload["name"]),
            combo=combo,
            start_date=str(payload["start_date"]),
            end_date=str(payload["end_date"]),
            train_years=int(payload.get("train_years", 3)),
            valid_quarters=int(payload.get("valid_quarters", 2)),
            trade_quarters=int(payload.get("trade_quarters", 1)),
            step_quarters=int(payload.get("step_quarters", 1)),
            embargo_days=int(payload.get("embargo_days", 2)),
            provider_uri=provider_uri.resolve(),
            output_root=output_root,
            topk=int(payload.get("topk", 50)),
            n_drop=int(payload.get("n_drop", 5)),
            strategy=str(payload.get("strategy", "bucket")),
            deal_price=str(payload.get("deal_price", "close")),
            rebalance=str(payload.get("rebalance", "day")),
            risk_degree=float(payload.get("risk_degree", 0.95)),
            settlement_lag=int(payload.get("settlement_lag", 2)),
            account=float(payload.get("account", 10_000_000.0)),
            open_cost=float(payload.get("open_cost", 0.00092625)),
            close_cost=float(payload.get("close_cost", 0.00392625)),
            min_cost=float(payload.get("min_cost", 20.0)),
            odd_lot_min_cost=float(payload.get("odd_lot_min_cost", 1.0)),
            board_lot_size=int(payload.get("board_lot_size", 1000)),
            trade_unit=int(payload.get("trade_unit", 1)),
            max_instruments=int(payload["max_instruments"]) if payload.get("max_instruments") is not None else None,
            model_kwargs=dict(payload.get("model_kwargs", {})) or None,
            strategy_mode=str(payload.get("strategy_mode", "fixed")),
            strategy_search=dict(payload.get("strategy_search", {})) or None,
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


def _date_str(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return _date_str(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _flatten_row(row: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat[f"{key}.{sub_key}"] = sub_value
        elif isinstance(value, Path):
            flat[key] = str(value)
        elif isinstance(value, pd.Timestamp):
            flat[key] = _date_str(value)
        else:
            flat[key] = value
    return flat


def _save_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([_flatten_row(row) for row in rows]).to_csv(path, index=False)


def read_calendar_dates(provider_uri: Path) -> list[pd.Timestamp]:
    day_file = provider_uri / "calendars" / "day.txt"
    lines = [line.strip() for line in day_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Calendar file is empty: {day_file}")
    return sorted({pd.Timestamp(line).normalize() for line in lines})


def _normalize_calendar(calendar: Iterable[pd.Timestamp | str]) -> list[pd.Timestamp]:
    normalized = sorted({pd.Timestamp(value).normalize() for value in calendar})
    if not normalized:
        raise ValueError("Trading calendar is empty")
    return normalized


def _add_quarters(value: pd.Timestamp, quarters: int) -> pd.Timestamp:
    return (pd.Timestamp(value).normalize() + pd.DateOffset(months=quarters * 3)).normalize()


def _first_on_or_after(calendar: list[pd.Timestamp], value: pd.Timestamp) -> pd.Timestamp | None:
    idx = bisect.bisect_left(calendar, pd.Timestamp(value).normalize())
    if idx >= len(calendar):
        return None
    return calendar[idx]


def _last_on_or_before(calendar: list[pd.Timestamp], value: pd.Timestamp) -> pd.Timestamp | None:
    idx = bisect.bisect_right(calendar, pd.Timestamp(value).normalize()) - 1
    if idx < 0:
        return None
    return calendar[idx]


def _calendar_index(calendar: list[pd.Timestamp], value: pd.Timestamp) -> int:
    idx = bisect.bisect_left(calendar, pd.Timestamp(value).normalize())
    if idx >= len(calendar) or calendar[idx] != pd.Timestamp(value).normalize():
        raise ValueError(f"Date is not in trading calendar: {_date_str(value)}")
    return idx


def _embargo_end_before(calendar: list[pd.Timestamp], next_start: pd.Timestamp, embargo_days: int) -> pd.Timestamp | None:
    idx = _calendar_index(calendar, next_start)
    end_idx = idx - int(embargo_days) - 1
    if end_idx < 0:
        return None
    return calendar[end_idx]


def _same_quarter(value: pd.Timestamp, anchor: pd.Timestamp) -> bool:
    return pd.Timestamp(value).to_period("Q") == pd.Timestamp(anchor).to_period("Q")


def _skip_record(round_id: int, train_anchor: pd.Timestamp, reason: str) -> dict[str, Any]:
    return {
        "round_id": round_id,
        "train_anchor": _date_str(train_anchor),
        "reason": reason,
    }


def _assert_split(split: RollingSplit, calendar: list[pd.Timestamp], embargo_days: int) -> None:
    index = {dt: idx for idx, dt in enumerate(calendar)}
    for field_name in (
        "train_start",
        "train_end",
        "valid_start",
        "valid_end",
        "trade_start",
        "trade_end",
    ):
        value = getattr(split, field_name)
        assert value in index, f"{field_name} is not a trading-calendar date: {_date_str(value)}"

    assert split.train_start <= split.train_end, "train segment is empty"
    assert split.valid_start <= split.valid_end, "validation segment is empty"
    assert split.trade_start <= split.trade_end, "trade segment is empty"
    assert split.train_end < split.valid_start, "train_end must be before valid_start"
    assert split.valid_end < split.trade_start, "valid_end must be before trade_start"
    assert split.trade_start > split.valid_end, "trade_start must be after valid_end"

    train_gap = index[split.valid_start] - index[split.train_end] - 1
    valid_gap = index[split.trade_start] - index[split.valid_end] - 1
    assert train_gap >= embargo_days, f"train/valid embargo is {train_gap}, expected >= {embargo_days}"
    assert valid_gap >= embargo_days, f"valid/trade embargo is {valid_gap}, expected >= {embargo_days}"
    assert split.trade_start > split.train_end and split.trade_end > split.valid_end, "trade period overlaps train/valid"


def generate_rolling_splits(
    calendar: Iterable[pd.Timestamp | str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    train_years: int = 3,
    valid_quarters: int = 2,
    trade_quarters: int = 1,
    step_quarters: int = 1,
    embargo_days: int = 2,
) -> RollingSplitPlan:
    if train_years <= 0:
        raise ValueError("train_years must be positive")
    if valid_quarters <= 0 or trade_quarters <= 0 or step_quarters <= 0:
        raise ValueError("valid_quarters/trade_quarters/step_quarters must be positive")
    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")

    trading_calendar = _normalize_calendar(calendar)
    requested_start = pd.Timestamp(start_date).normalize()
    requested_end = pd.Timestamp(end_date).normalize()
    if requested_start > requested_end:
        raise ValueError("start_date must be on or before end_date")

    train_quarters = int(train_years) * 4
    splits: list[RollingSplit] = []
    skipped: list[dict[str, Any]] = []
    offset_quarters = 0
    round_id = 1

    while True:
        train_anchor = _add_quarters(requested_start, offset_quarters)
        valid_anchor = _add_quarters(train_anchor, train_quarters)
        trade_anchor = _add_quarters(valid_anchor, valid_quarters)
        trade_end_anchor = _add_quarters(trade_anchor, trade_quarters) - pd.Timedelta(days=1)

        if trade_anchor > requested_end:
            break
        if trade_end_anchor > requested_end:
            skipped.append(_skip_record(round_id, train_anchor, "trade period is incomplete under requested end_date"))
            break
        if trade_end_anchor > trading_calendar[-1]:
            skipped.append(_skip_record(round_id, train_anchor, "trade period is incomplete under provider calendar"))
            break

        train_start = _first_on_or_after(trading_calendar, train_anchor)
        valid_start = _first_on_or_after(trading_calendar, valid_anchor)
        trade_start = _first_on_or_after(trading_calendar, trade_anchor)
        trade_end = _last_on_or_before(trading_calendar, trade_end_anchor)

        if train_start is None or not _same_quarter(train_start, train_anchor):
            skipped.append(_skip_record(round_id, train_anchor, "train window start is outside available calendar quarter"))
            offset_quarters += step_quarters
            round_id += 1
            continue
        if valid_start is None or not _same_quarter(valid_start, valid_anchor):
            skipped.append(_skip_record(round_id, train_anchor, "validation window start is outside available calendar quarter"))
            break
        if trade_start is None or not _same_quarter(trade_start, trade_anchor):
            skipped.append(_skip_record(round_id, train_anchor, "trade window start is outside available calendar quarter"))
            break
        if trade_end is None or trade_end < trade_start or not _same_quarter(trade_end, trade_anchor):
            skipped.append(_skip_record(round_id, train_anchor, "trade window has no complete trading dates"))
            break

        train_end = _embargo_end_before(trading_calendar, valid_start, embargo_days)
        valid_end = _embargo_end_before(trading_calendar, trade_start, embargo_days)
        if train_end is None or train_start > train_end:
            skipped.append(_skip_record(round_id, train_anchor, "train window is empty after embargo"))
            offset_quarters += step_quarters
            round_id += 1
            continue
        if valid_end is None or valid_start > valid_end:
            skipped.append(_skip_record(round_id, train_anchor, "validation window is empty after embargo"))
            offset_quarters += step_quarters
            round_id += 1
            continue

        split = RollingSplit(
            round_id=round_id,
            train_start=train_start,
            train_end=train_end,
            valid_start=valid_start,
            valid_end=valid_end,
            trade_start=trade_start,
            trade_end=trade_end,
        )
        _assert_split(split, trading_calendar, embargo_days)
        splits.append(split)

        offset_quarters += step_quarters
        round_id += 1

    return RollingSplitPlan(splits=splits, skipped_rounds=skipped)


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _build_calendar_overlay_provider(base_provider_uri: Path, overlay_uri: Path) -> Path:
    base_calendar = read_calendar_dates(base_provider_uri)
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
    overlay_lines = [_date_str(dt) for dt in base_calendar]
    next_label = _date_str(next_business_day)
    if overlay_lines[-1] != next_label:
        overlay_lines.append(next_label)
    (calendars_dir / "day.txt").write_text("\n".join(overlay_lines) + "\n", encoding="utf-8")
    return overlay_uri


def _make_paths(output_root: Path) -> WorkflowPaths:
    report_dir = output_root / "reports"
    fig_dir = output_root / "figures"
    for directory in (output_root, report_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return WorkflowPaths(output_root=output_root, report_dir=report_dir, fig_dir=fig_dir)


def _init_qlib(provider_uri: Path, region: str) -> None:
    import qlib

    LOGGER.info("Initialize Qlib for rolling walk-forward, provider uri: %s", provider_uri)
    qlib.init(provider_uri=str(provider_uri), region=region, exp_manager=build_exp_manager_config())


def _ensure_no_active_mlflow_run() -> None:
    import mlflow

    while mlflow.active_run() is not None:
        mlflow.end_run()


def _prediction_frame(pred: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(pred, pd.Series):
        return pred.to_frame("score")
    if isinstance(pred, pd.DataFrame):
        if len(pred.columns) == 1:
            pred = pred.copy()
            pred.columns = ["score"]
        return pred
    raise TypeError(f"Unsupported prediction type: {type(pred)}")


def _label_frame(label: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(label, pd.Series):
        return label.to_frame("label")
    if isinstance(label, pd.DataFrame):
        label = label.copy()
        if len(label.columns) == 1:
            label.columns = ["label"]
        return label
    raise TypeError(f"Unsupported label type: {type(label)}")


def _build_round_task_config(
    config: RollingWalkForwardConfig,
    split: RollingSplit,
    universe: list[str],
) -> dict:
    spec = config.combo_spec
    max_instruments = config.max_instruments if config.max_instruments is not None else spec.get("max_instruments")
    return build_task_config(
        handler_key=str(spec["handler"]),
        model_key=str(spec["model"]),
        instruments=universe,
        max_instruments=max_instruments,
        infer_processors=spec.get("infer_processors"),  # type: ignore[arg-type]
        model_kwargs_override=config.model_kwargs,
        segments_override=split.segments(),
        handler_kwargs_override={
            "start_time": _date_str(split.train_start),
            "end_time": _date_str(split.trade_end),
            "fit_start_time": _date_str(split.train_start),
            "fit_end_time": _date_str(split.train_end),
        },
    )


def _build_backtest_task_config(
    config: RollingWalkForwardConfig,
    universe: list[str],
    first_split: RollingSplit,
    last_split: RollingSplit,
) -> dict:
    spec = config.combo_spec
    max_instruments = config.max_instruments if config.max_instruments is not None else spec.get("max_instruments")
    return build_task_config(
        handler_key=str(spec["handler"]),
        model_key=str(spec["model"]),
        instruments=universe,
        max_instruments=max_instruments,
        infer_processors=spec.get("infer_processors"),  # type: ignore[arg-type]
        model_kwargs_override=config.model_kwargs,
        segments_override={
            "train": (_date_str(first_split.train_start), _date_str(first_split.train_end)),
            "valid": (_date_str(first_split.valid_start), _date_str(first_split.valid_end)),
            "test": (_date_str(first_split.trade_start), _date_str(last_split.trade_end)),
        },
        handler_kwargs_override={
            "start_time": _date_str(first_split.train_start),
            "end_time": _date_str(last_split.trade_end),
            "fit_start_time": _date_str(first_split.train_start),
            "fit_end_time": _date_str(first_split.train_end),
        },
    )


def _base_strategy_params(config: RollingWalkForwardConfig) -> dict[str, Any]:
    return {
        "topk": int(config.topk),
        "n_drop": int(config.n_drop),
        "strategy": str(config.strategy),
        "deal_price": str(config.deal_price),
        "rebalance": str(config.rebalance),
        "risk_degree": float(config.risk_degree),
        "settlement_lag": int(config.settlement_lag),
    }


def _strategy_search_config(config: RollingWalkForwardConfig) -> dict[str, Any]:
    search_cfg = dict(config.strategy_search or {})
    return {
        "ranking_metric": str(search_cfg.get("ranking_metric", "risk.excess_return_with_cost.annualized_return")),
        "ranking_order": str(search_cfg.get("ranking_order", "desc")),
        "topk_values": [int(v) for v in search_cfg.get("topk_values", [config.topk])],
        "n_drop_values": [int(v) for v in search_cfg.get("n_drop_values", [config.n_drop])],
        "risk_degree_values": [float(v) for v in search_cfg.get("risk_degree_values", [config.risk_degree])],
        "tie_break": list(
            search_cfg.get(
                "tie_break",
                [
                    {"metric": "risk.excess_return_with_cost.max_drawdown", "order": "desc"},
                    {"metric": "avg_turnover", "order": "asc"},
                    {"metric": "topk", "order": "desc"},
                    {"metric": "n_drop", "order": "asc"},
                ],
            )
        ),
    }


def _validate_strategy_mode(config: RollingWalkForwardConfig) -> None:
    if config.strategy_mode not in {"fixed", "validation_search"}:
        raise ValueError("strategy_mode must be 'fixed' or 'validation_search'")
    if config.strategy_mode == "validation_search" and config.strategy != "bucket":
        raise ValueError("validation_search currently supports strategy='bucket' only")


def _build_strategy_variants(config: RollingWalkForwardConfig) -> list[dict[str, Any]]:
    search_cfg = _strategy_search_config(config)
    variants = []
    for topk, n_drop, risk_degree in itertools.product(
        search_cfg["topk_values"],
        search_cfg["n_drop_values"],
        search_cfg["risk_degree_values"],
    ):
        if int(n_drop) > int(topk):
            continue
        variant = _base_strategy_params(config)
        variant.update(
            {
                "topk": int(topk),
                "n_drop": int(n_drop),
                "risk_degree": float(risk_degree),
            }
        )
        variants.append(variant)
    if not variants:
        raise ValueError("strategy_search produced no valid strategy variants")
    return variants


def _variant_slug(variant: dict[str, Any]) -> str:
    risk = str(variant["risk_degree"]).replace(".", "p")
    return f"{variant['strategy']}_topk{variant['topk']}_ndrop{variant['n_drop']}_risk{risk}"


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _compare_metric(left: dict[str, Any], right: dict[str, Any], metric: str, order: str) -> int:
    left_value = _metric_value(left, metric)
    right_value = _metric_value(right, metric)
    left_nan = pd.isna(left_value)
    right_nan = pd.isna(right_value)
    if left_nan and right_nan:
        return 0
    if left_nan:
        return 1
    if right_nan:
        return -1
    if left_value == right_value:
        return 0
    if order == "asc":
        return -1 if left_value < right_value else 1
    return -1 if left_value > right_value else 1


def _sort_strategy_search_rows(rows: list[dict[str, Any]], search_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    ranking_metric = str(search_cfg["ranking_metric"])
    ranking_order = str(search_cfg["ranking_order"])
    tie_break = list(search_cfg["tie_break"])

    def _compare(left: dict[str, Any], right: dict[str, Any]) -> int:
        result = _compare_metric(left, right, ranking_metric, ranking_order)
        if result != 0:
            return result
        for item in tie_break:
            result = _compare_metric(left, right, str(item["metric"]), str(item.get("order", "desc")))
            if result != 0:
                return result
        return 0

    return sorted(rows, key=cmp_to_key(_compare))


def _build_strategy_schedule(splits: list[RollingSplit], selected_params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    schedule = []
    for split, params in zip(splits, selected_params):
        schedule.append(
            {
                "round_id": int(split.round_id),
                "start_time": _date_str(split.trade_start),
                "end_time": _date_str(split.trade_end),
                "topk": int(params["topk"]),
                "n_drop": int(params["n_drop"]),
                "risk_degree": float(params["risk_degree"]),
            }
        )
    return schedule


def _build_port_config(
    config: RollingWalkForwardConfig,
    task_cfg: dict,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    strategy_params: dict[str, Any],
    *,
    strategy_schedule: list[dict[str, Any]] | None = None,
) -> dict:
    port_config = build_port_analysis_config()
    port_config["backtest"]["start_time"] = _date_str(start_time)
    port_config["backtest"]["end_time"] = _date_str(end_time)
    port_config["backtest"]["account"] = config.account
    port_config["backtest"]["exchange_kwargs"]["open_cost"] = config.open_cost
    port_config["backtest"]["exchange_kwargs"]["close_cost"] = config.close_cost
    port_config["backtest"]["exchange_kwargs"]["min_cost"] = config.min_cost
    port_config["backtest"]["exchange_kwargs"]["trade_unit"] = config.trade_unit
    apply_strategy_overrides(
        port_config,
        task_cfg,
        n_drop_override=int(strategy_params["n_drop"]),
        topk_override=int(strategy_params["topk"]),
        rebalance=str(strategy_params["rebalance"]),
        strategy_choice=str(strategy_params["strategy"]),
        deal_price=str(strategy_params["deal_price"]),
    )
    strategy_kwargs = port_config["strategy"]["kwargs"]
    strategy_kwargs["risk_degree"] = float(strategy_params["risk_degree"])
    strategy_kwargs.pop("model", None)
    strategy_kwargs.pop("dataset", None)
    strategy_kwargs["signal"] = "<PRED>"
    if strategy_schedule is not None:
        port_config["strategy"]["class"] = "RollingScheduledBucketWeightTopkDropout"
        port_config["strategy"]["module_path"] = "qlib_tw.trade.custom_strategy"
        strategy_kwargs["strategy_schedule"] = strategy_schedule

    exchange_kwargs = port_config["backtest"]["exchange_kwargs"]["exchange"]["kwargs"]
    exchange_kwargs["odd_lot_min_cost"] = config.odd_lot_min_cost
    exchange_kwargs["board_lot_size"] = config.board_lot_size
    exchange_kwargs["settlement_lag"] = int(strategy_params["settlement_lag"])
    return port_config


def _slice_report_metrics(report_df: pd.DataFrame, split: RollingSplit) -> dict[str, Any]:
    start = pd.Timestamp(split.trade_start)
    end = pd.Timestamp(split.trade_end)
    sliced = report_df[(pd.to_datetime(report_df.index) >= start) & (pd.to_datetime(report_df.index) <= end)]
    if sliced.empty:
        return {
            "trade_cumulative_return": float("nan"),
            "net_cumulative_return": float("nan"),
            "benchmark_cumulative_return": float("nan"),
            "total_cost": float("nan"),
            "total_turnover": float("nan"),
            "avg_turnover": float("nan"),
            "trading_days": 0,
        }
    turnover = sliced["total_turnover"] if "total_turnover" in sliced.columns else pd.Series(dtype="float64")
    net_return = sliced["return"] - sliced["cost"] if "cost" in sliced.columns else None
    return {
        "trade_cumulative_return": float((1 + sliced["return"]).prod() - 1),
        "net_cumulative_return": float((1 + net_return).prod() - 1) if net_return is not None else float("nan"),
        "benchmark_cumulative_return": float((1 + sliced["bench"]).prod() - 1),
        "total_cost": float(sliced["cost"].sum()) if "cost" in sliced.columns else float("nan"),
        "total_turnover": float(turnover.sum()) if not turnover.empty else float("nan"),
        "avg_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
        "trading_days": int((turnover > 0).sum()) if not turnover.empty else int(len(sliced)),
    }


def _extract_summary_metrics(recorder) -> Dict[str, Any]:
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    net_return = report_df["return"] - report_df["cost"] if "cost" in report_df.columns else None
    metrics: Dict[str, Any] = {
        "strategy_cumulative_return": float((1 + report_df["return"]).prod() - 1),
        "net_cumulative_return": float((1 + net_return).prod() - 1) if net_return is not None else float("nan"),
        "benchmark_cumulative_return": float((1 + report_df["bench"]).prod() - 1),
        "total_cost": float(report_df["cost"].sum()) if "cost" in report_df.columns else float("nan"),
        "avg_turnover": float(report_df["total_turnover"].mean()) if "total_turnover" in report_df.columns else float("nan"),
    }
    if isinstance(analysis_df, pd.DataFrame) and "risk" in analysis_df.columns:
        risk_series = analysis_df["risk"]
        if isinstance(risk_series, pd.Series):
            for (category, metric), value in risk_series.items():
                if pd.notna(value):
                    metrics[f"risk.{category}.{metric}"] = float(value)
    return metrics


def _run_validation_strategy_search(
    config: RollingWalkForwardConfig,
    task_cfg: dict,
    split: RollingSplit,
    valid_pred: pd.DataFrame,
    valid_label: pd.DataFrame,
    output_root: Path,
    variants: list[dict[str, Any]],
    search_cfg: dict[str, Any],
    *,
    recorder_api,
    port_record_cls,
) -> dict[str, Any]:
    experiment_name = f"rolling_strategy_search_{config.name}"
    rows: list[dict[str, Any]] = []
    search_dir = output_root / "strategy_search"
    search_dir.mkdir(parents=True, exist_ok=True)

    for index, variant in enumerate(variants, start=1):
        variant_slug = _variant_slug(variant)
        LOGGER.info("Round %d validation strategy search %d/%d: %s", split.round_id, index, len(variants), variant_slug)
        port_config = _build_port_config(
            config,
            task_cfg,
            split.valid_start,
            split.valid_end,
            variant,
        )
        _ensure_no_active_mlflow_run()
        with recorder_api.start(experiment_name=experiment_name):
            current_recorder = recorder_api.get_recorder()
            current_recorder.set_tags(
                rolling_name=config.name,
                round_id=str(split.round_id),
                variant_slug=variant_slug,
            )
            current_recorder.log_params(
                strategy_mode=config.strategy_mode,
                round_id=split.round_id,
                validation_start=_date_str(split.valid_start),
                validation_end=_date_str(split.valid_end),
                **variant,
            )
            current_recorder.save_objects(**{"pred.pkl": valid_pred, "label.pkl": valid_label})
            port_rec = port_record_cls(current_recorder, port_config, "day")
            port_rec.generate()
            recorder_id = current_recorder.id

        recorder = recorder_api.get_recorder(experiment_name=experiment_name, recorder_id=recorder_id)
        summary_metrics = _extract_summary_metrics(recorder)
        row = {
            "round_id": split.round_id,
            "variant_index": index,
            "variant_slug": variant_slug,
            "strategy_search_experiment": experiment_name,
            "strategy_search_recorder_id": recorder_id,
            **variant,
            **summary_metrics,
        }
        row["ranking_metric"] = search_cfg["ranking_metric"]
        row["ranking_value"] = _metric_value(row, str(search_cfg["ranking_metric"]))
        rows.append(row)

    rows = _sort_strategy_search_rows(rows, search_cfg)
    _save_rows_csv(rows, search_dir / f"round_{split.round_id:03d}_strategy_search.csv")
    _save_json(rows, search_dir / f"round_{split.round_id:03d}_strategy_search.json")
    if not rows:
        raise RuntimeError(f"No validation strategy search rows were produced for round {split.round_id}")
    best = rows[0]
    _save_json(best, search_dir / f"round_{split.round_id:03d}_best_strategy.json")
    return best


def _write_split_outputs(plan: RollingSplitPlan, output_root: Path) -> None:
    _save_rows_csv([split.to_record() for split in plan.splits], output_root / "splits.csv")
    _save_json([split.to_record() for split in plan.splits], output_root / "splits.json")
    _save_rows_csv(plan.skipped_rounds, output_root / "skipped_rounds.csv")
    _save_json(plan.skipped_rounds, output_root / "skipped_rounds.json")


def run_rolling_walk_forward(config: RollingWalkForwardConfig, *, dry_run: bool = False) -> Dict[str, Any]:
    _validate_strategy_mode(config)
    output_root = config.resolved_output_root
    output_root.mkdir(parents=True, exist_ok=True)
    paths = _make_paths(output_root)
    calendar = read_calendar_dates(config.resolved_provider_uri)
    split_plan = generate_rolling_splits(
        calendar,
        start_date=config.start_date,
        end_date=config.end_date,
        train_years=config.train_years,
        valid_quarters=config.valid_quarters,
        trade_quarters=config.trade_quarters,
        step_quarters=config.step_quarters,
        embargo_days=config.embargo_days,
    )
    _write_split_outputs(split_plan, output_root)
    _save_json(
        {
            "config": asdict(config),
            "provider_uri": str(config.resolved_provider_uri),
            "split_count": len(split_plan.splits),
            "skipped_round_count": len(split_plan.skipped_rounds),
            "dry_run": dry_run,
        },
        output_root / "resolved_config.json",
    )

    if dry_run:
        LOGGER.info("Dry-run generated %d rolling splits under %s", len(split_plan.splits), output_root)
        return {"output_root": output_root, "splits": split_plan.splits, "skipped_rounds": split_plan.skipped_rounds}
    if not split_plan.splits:
        raise RuntimeError("No complete rolling walk-forward splits were produced")

    from qlib.utils import flatten_dict, init_instance_by_config
    from qlib.workflow import R
    from qlib.workflow.record_temp import PortAnaRecord

    from qlib_tw.research.reports import dump_report_frames

    active_provider_uri = config.resolved_provider_uri
    if split_plan.splits[-1].trade_end == calendar[-1]:
        active_provider_uri = _build_calendar_overlay_provider(config.resolved_provider_uri, output_root / "_provider_overlay")
        LOGGER.info("Using calendar overlay provider %s", active_provider_uri)

    _init_qlib(active_provider_uri, config.region)
    all_codes = load_full_universe(active_provider_uri, benchmark=BENCHMARK)
    universe = [code for code in all_codes if not code.startswith("^")]
    if not universe:
        raise RuntimeError(f"No equity instruments found under {active_provider_uri}")

    predictions_dir = output_root / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    train_experiment = f"rolling_train_{config.name}"
    round_rows: list[dict[str, Any]] = []
    oos_predictions: list[pd.DataFrame] = []
    oos_labels: list[pd.DataFrame] = []
    selected_strategy_params: list[dict[str, Any]] = []
    strategy_search_rows: list[dict[str, Any]] = []
    search_cfg = _strategy_search_config(config)
    strategy_variants = _build_strategy_variants(config) if config.strategy_mode == "validation_search" else []

    for split in split_plan.splits:
        LOGGER.info(
            "Round %d train=%s~%s valid=%s~%s trade=%s~%s",
            split.round_id,
            _date_str(split.train_start),
            _date_str(split.train_end),
            _date_str(split.valid_start),
            _date_str(split.valid_end),
            _date_str(split.trade_start),
            _date_str(split.trade_end),
        )
        task_cfg = _build_round_task_config(config, split, universe)
        model = init_instance_by_config(task_cfg["model"])
        dataset = init_instance_by_config(task_cfg["dataset"])
        _ensure_no_active_mlflow_run()
        with R.start(experiment_name=train_experiment):
            current_recorder = R.get_recorder()
            current_recorder.set_tags(rolling_name=config.name, round_id=str(split.round_id))
            current_recorder.log_params(**flatten_dict(task_cfg))
            current_recorder.log_params(**{f"rolling.{key}": value for key, value in split.to_record().items()})
            model.fit(dataset, **task_cfg.get("model_fit_kwargs", {}))
            current_recorder.save_objects(trained_model=model)
            train_recorder_id = current_recorder.id

        valid_pred = _prediction_frame(model.predict(dataset, segment="valid"))
        trade_pred = _prediction_frame(model.predict(dataset, segment="test"))
        valid_label = _label_frame(dataset.prepare("valid", col_set="label"))
        trade_label = _label_frame(dataset.prepare("test", col_set="label"))
        valid_metrics = calc_ic_metrics(valid_pred, valid_label)
        trade_metrics = calc_ic_metrics(trade_pred, trade_label)

        selected_params = _base_strategy_params(config)
        best_strategy_row: dict[str, Any] | None = None
        if config.strategy_mode == "validation_search":
            best_strategy_row = _run_validation_strategy_search(
                config,
                task_cfg,
                split,
                valid_pred,
                valid_label,
                output_root,
                strategy_variants,
                search_cfg,
                recorder_api=R,
                port_record_cls=PortAnaRecord,
            )
            selected_params = {
                key: best_strategy_row[key]
                for key in ("topk", "n_drop", "strategy", "deal_price", "rebalance", "risk_degree", "settlement_lag")
            }
            strategy_search_rows.append(best_strategy_row)
        selected_strategy_params.append(selected_params)

        if trade_pred.index.duplicated().any():
            raise RuntimeError(f"Duplicate trade prediction index in round {split.round_id}")
        trade_pred.to_pickle(predictions_dir / f"round_{split.round_id:03d}_pred.pkl")
        trade_pred.reset_index().to_csv(predictions_dir / f"round_{split.round_id:03d}_pred.csv", index=False)
        oos_predictions.append(trade_pred)
        oos_labels.append(trade_label)

        round_rows.append(
            {
                **split.to_record(),
                "train_experiment": train_experiment,
                "train_recorder_id": train_recorder_id,
                "strategy_mode": config.strategy_mode,
                "selected_topk": int(selected_params["topk"]),
                "selected_n_drop": int(selected_params["n_drop"]),
                "selected_strategy": str(selected_params["strategy"]),
                "selected_deal_price": str(selected_params["deal_price"]),
                "selected_rebalance": str(selected_params["rebalance"]),
                "selected_risk_degree": float(selected_params["risk_degree"]),
                "selected_settlement_lag": int(selected_params["settlement_lag"]),
                "strategy_search_ranking_metric": search_cfg["ranking_metric"] if best_strategy_row else None,
                "strategy_search_ranking_value": best_strategy_row["ranking_value"] if best_strategy_row else None,
                "strategy_search_recorder_id": best_strategy_row["strategy_search_recorder_id"] if best_strategy_row else None,
                **{f"valid_{key}": value for key, value in valid_metrics.items()},
                **{f"trade_{key}": value for key, value in trade_metrics.items()},
            }
        )

    rolling_pred = pd.concat(oos_predictions).sort_index()
    rolling_label = pd.concat(oos_labels).sort_index()
    if rolling_pred.index.duplicated().any():
        dup_count = int(rolling_pred.index.duplicated().sum())
        raise RuntimeError(f"Rolling OOS predictions contain {dup_count} duplicate rows")
    rolling_pred.to_pickle(output_root / "rolling_oos_pred.pkl")
    rolling_pred.reset_index().to_csv(output_root / "rolling_oos_pred.csv", index=False)
    rolling_label.to_pickle(output_root / "rolling_oos_label.pkl")

    first_split = split_plan.splits[0]
    last_split = split_plan.splits[-1]
    backtest_task_cfg = _build_backtest_task_config(config, universe, first_split, last_split)
    backtest_dataset = init_instance_by_config(backtest_task_cfg["dataset"])
    strategy_schedule = None
    final_strategy_params = _base_strategy_params(config)
    if config.strategy_mode == "validation_search":
        strategy_schedule = _build_strategy_schedule(split_plan.splits, selected_strategy_params)
        final_strategy_params = selected_strategy_params[0]
        _save_rows_csv(strategy_schedule, output_root / "selected_strategy_schedule.csv")
        _save_json(strategy_schedule, output_root / "selected_strategy_schedule.json")
        _save_rows_csv(strategy_search_rows, output_root / "selected_strategy_results.csv")
        _save_json(strategy_search_rows, output_root / "selected_strategy_results.json")
    port_config = _build_port_config(
        config,
        backtest_task_cfg,
        first_split.trade_start,
        last_split.trade_end,
        final_strategy_params,
        strategy_schedule=strategy_schedule,
    )
    backtest_experiment = f"rolling_backtest_{config.name}"
    _ensure_no_active_mlflow_run()
    with R.start(experiment_name=backtest_experiment):
        current_recorder = R.get_recorder()
        current_recorder.set_tags(rolling_name=config.name, split_count=str(len(split_plan.splits)))
        current_recorder.log_params(
            name=config.name,
            combo=config.combo,
            backtest_start=_date_str(first_split.trade_start),
            backtest_end=_date_str(last_split.trade_end),
            strategy_mode=config.strategy_mode,
            topk=final_strategy_params["topk"],
            n_drop=final_strategy_params["n_drop"],
            strategy=final_strategy_params["strategy"],
            deal_price=final_strategy_params["deal_price"],
            rebalance=final_strategy_params["rebalance"],
            risk_degree=final_strategy_params["risk_degree"],
            settlement_lag=final_strategy_params["settlement_lag"],
        )
        current_recorder.save_objects(**{"pred.pkl": rolling_pred, "label.pkl": rolling_label})
        port_rec = PortAnaRecord(current_recorder, port_config, "day")
        port_rec.generate()
        backtest_recorder_id = current_recorder.id

    recorder = R.get_recorder(experiment_name=backtest_experiment, recorder_id=backtest_recorder_id)
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    for row, split in zip(round_rows, split_plan.splits):
        row.update(_slice_report_metrics(report_df, split))

    summary_metrics = _extract_summary_metrics(recorder)
    _save_rows_csv(round_rows, output_root / "round_results.csv")
    _save_json(round_rows, output_root / "round_results.json")
    _save_json(
        {
            "name": config.name,
            "combo": config.combo,
            "split_count": len(split_plan.splits),
            "train_experiment": train_experiment,
            "backtest_experiment": backtest_experiment,
            "backtest_recorder_id": backtest_recorder_id,
            "backtest_start": _date_str(first_split.trade_start),
            "backtest_end": _date_str(last_split.trade_end),
            "strategy_mode": config.strategy_mode,
            "strategy_search_config": search_cfg if config.strategy_mode == "validation_search" else None,
            "summary_metrics": summary_metrics,
        },
        output_root / "rolling_backtest_summary.json",
    )

    handler_kwargs = backtest_task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]
    segments = backtest_task_cfg["dataset"]["kwargs"]["segments"]
    dump_report_frames(
        recorder,
        backtest_dataset,
        universe=handler_kwargs["instruments"],
        data_handler_config=handler_kwargs,
        segments=segments,
        port_config=port_config,
        paths=paths,
    )
    _ensure_no_active_mlflow_run()
    return {
        "output_root": output_root,
        "splits": split_plan.splits,
        "round_results": round_rows,
        "backtest_experiment": backtest_experiment,
        "backtest_recorder_id": backtest_recorder_id,
        "summary_metrics": summary_metrics,
    }
