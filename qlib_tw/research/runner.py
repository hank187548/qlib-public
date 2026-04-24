from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
from typing import Dict, List

import pandas as pd
import qlib
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from qlib_tw.data_layout import build_exp_manager_config
from qlib_tw.research.builders import apply_strategy_overrides, build_port_analysis_config, build_task_config
from qlib_tw.research.paths import build_effective_name, set_backtest_output_dirs, set_model_output_dirs
from qlib_tw.research.reports import dump_model_frames, dump_report_frames
from qlib_tw.research.settings import PROVIDER_URI, REGION, UNIVERSE


LOGGER = logging.getLogger("qlib_tw.research.runner")


def init_qlib(provider_uri: Path | None = None) -> None:
    active_provider_uri = provider_uri.resolve() if provider_uri is not None else PROVIDER_URI.resolve()
    LOGGER.info("Initialize Qlib, provider uri: %s", active_provider_uri)
    qlib.init(provider_uri=str(active_provider_uri), region=REGION, exp_manager=build_exp_manager_config())
    LOGGER.info("Universe size: %d symbols", len(UNIVERSE))


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


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _build_calendar_overlay_provider(output_root: Path, provider_uri: Path) -> Path:
    base_calendar = _read_calendar_dates(provider_uri)
    next_business_day = (base_calendar[-1] + pd.tseries.offsets.BDay(1)).normalize()
    overlay_uri = (output_root / "_provider_overlay").resolve()
    overlay_uri.mkdir(parents=True, exist_ok=True)

    for dirname in ("features", "instruments"):
        source = (provider_uri / dirname).resolve()
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
    (calendars_dir / "day.txt").write_text("\n".join(overlay_lines) + "\n")
    return overlay_uri


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


def train_combo(
    combo_name: str,
    handler_key: str,
    model_key: str,
    max_instruments,
    infer_processors,
    *,
    threads: int | None = None,
    model_kwargs_override: Dict[str, object] | None = None,
) -> str:
    paths = set_model_output_dirs(combo_name)
    task_config = build_task_config(
        handler_key,
        model_key,
        UNIVERSE,
        max_instruments,
        infer_processors,
        model_kwargs_override=model_kwargs_override,
    )
    if threads is not None:
        if "thread_count" in task_config["model"]["kwargs"]:
            task_config["model"]["kwargs"]["thread_count"] = threads
        if "num_threads" in task_config["model"]["kwargs"]:
            task_config["model"]["kwargs"]["num_threads"] = threads
        if "nthread" in task_config["model"]["kwargs"]:
            task_config["model"]["kwargs"]["nthread"] = threads

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])
    train_exp = f"tw_train_model_{combo_name}"
    LOGGER.info("Start model training - experiment %s", train_exp)
    with R.start(experiment_name=train_exp):
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        signal_rec = SignalRecord(model, dataset, R.get_recorder())
        signal_rec.generate()
        model_rid = R.get_recorder().id
    recorder = R.get_recorder(recorder_id=model_rid, experiment_name=train_exp)
    handler_kwargs = task_config["dataset"]["kwargs"]["handler"]["kwargs"]
    segments = task_config["dataset"]["kwargs"]["segments"]
    dump_model_frames(
        recorder,
        dataset,
        universe=handler_kwargs["instruments"],
        data_handler_config=handler_kwargs,
        segments=segments,
        paths=paths,
    )
    metadata = {
        "run_name": combo_name,
        "train_experiment": train_exp,
        "train_recorder_id": model_rid,
        "handler_key": handler_key,
        "model_key": model_key,
        "output_root": str(paths.output_root),
    }
    (paths.output_root / "train_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Model training complete, recorder id = %s", model_rid)
    return model_rid


def backtest_combo(
    combo_name: str,
    handler_key: str,
    model_key: str,
    max_instruments: int | None,
    infer_processors: List[Dict[str, object]] | None,
    *,
    n_drop_override: int | None = None,
    topk_override: int | None = None,
    rebalance: str = "day",
    strategy_choice: str = "bucket",
    deal_price: str = "close",
    recorder_override: str | None = None,
    model_kwargs_override: Dict[str, object] | None = None,
    account_override: float | None = None,
    risk_degree_override: float | None = None,
    adjust_prices_for_backtest: bool = False,
    backtest_end_time: str | None = None,
) -> None:
    effective_name = build_effective_name(
        combo_name,
        n_drop_override=n_drop_override,
        topk_override=topk_override,
        rebalance=rebalance,
        strategy_choice=strategy_choice,
        deal_price=deal_price,
        adjust_prices_for_backtest=adjust_prices_for_backtest,
    )
    paths = set_backtest_output_dirs(effective_name)
    base_provider_uri = PROVIDER_URI.resolve()

    task_config = build_task_config(
        handler_key,
        model_key,
        UNIVERSE,
        max_instruments,
        infer_processors,
        model_kwargs_override=model_kwargs_override,
    )
    port_config = build_port_analysis_config()
    requested_end_time = backtest_end_time or str(port_config["backtest"]["end_time"])
    requested_end = pd.Timestamp(requested_end_time).normalize()
    effective_end = _latest_trade_date_on_or_before(base_provider_uri, requested_end)
    effective_end_time = effective_end.strftime("%Y-%m-%d")
    if effective_end != requested_end:
        LOGGER.warning(
            "Requested backtest end %s is beyond the latest available trade date. Capping to %s.",
            requested_end.strftime("%Y-%m-%d"),
            effective_end_time,
        )
    dataset_kwargs = task_config["dataset"]["kwargs"]
    test_start, _ = dataset_kwargs["segments"]["test"]
    dataset_kwargs["segments"]["test"] = (test_start, effective_end_time)
    handler_kwargs = dataset_kwargs["handler"]["kwargs"]
    handler_kwargs["end_time"] = max(str(handler_kwargs["end_time"]), effective_end_time)
    port_config["backtest"]["end_time"] = effective_end_time
    if account_override is not None:
        port_config["backtest"]["account"] = account_override
    if risk_degree_override is not None:
        port_config["strategy"]["kwargs"]["risk_degree"] = risk_degree_override
    apply_strategy_overrides(
        port_config,
        task_config,
        n_drop_override=n_drop_override,
        topk_override=topk_override,
        rebalance=rebalance,
        strategy_choice=strategy_choice,
        deal_price=deal_price,
        adjust_prices_for_backtest=adjust_prices_for_backtest,
    )
    active_provider_uri = base_provider_uri
    if effective_end == _latest_calendar_date(base_provider_uri):
        active_provider_uri = _build_calendar_overlay_provider(paths.output_root, base_provider_uri)
        LOGGER.info(
            "Using calendar overlay provider %s to expose next-session boundary after %s",
            active_provider_uri,
            effective_end_time,
        )
    init_qlib(active_provider_uri)

    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])
    train_exp = f"tw_train_model_{combo_name}"
    recorder_id = recorder_override or latest_recorder_id(train_exp)
    LOGGER.info("Use training recorder: %s (experiment %s)", recorder_id, train_exp)
    train_recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=train_exp)
    trained_model = train_recorder.load_object("trained_model")

    backtest_exp = f"tw_backtest_{combo_name}"
    LOGGER.info("Start backtest - experiment %s", backtest_exp)
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

    LOGGER.info("Backtest complete, recorder id = %s", backtest_rid)
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
        paths=paths,
    )
    LOGGER.info("Combo %s completed. Output root: %s", effective_name, paths.output_root)
