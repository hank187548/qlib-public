from __future__ import annotations

import logging
from pathlib import Path
import shutil

import pandas as pd
import qlib
from qlib.utils import init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from paper_trading.config import PaperTradingProfile
from paper_trading.extract import extract_outputs, write_outputs
from paper_trading.paths import build_paths
from scripts.Get_data_Tai import run_collect
from tw_workflow.builders import apply_strategy_overrides, build_port_analysis_config, build_task_config


LOGGER = logging.getLogger("paper_trading")


def read_calendar_dates(provider_uri: Path) -> list[pd.Timestamp]:
    day_file = provider_uri / "calendars" / "day.txt"
    if not day_file.exists():
        raise FileNotFoundError(f"Calendar file not found under provider: {day_file}")
    lines = [line.strip() for line in day_file.read_text().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Calendar file is empty: {day_file}")
    return [pd.Timestamp(line).normalize() for line in lines]


def latest_calendar_date(provider_uri: Path) -> pd.Timestamp:
    return read_calendar_dates(provider_uri)[-1]


def latest_trade_date_on_or_before(provider_uri: Path, target_date: pd.Timestamp) -> pd.Timestamp:
    target_date = pd.Timestamp(target_date).normalize()
    eligible_dates = [dt for dt in read_calendar_dates(provider_uri) if dt <= target_date]
    if not eligible_dates:
        raise RuntimeError(f"No available trade date on or before {target_date.strftime('%Y-%m-%d')} under {provider_uri}")
    return eligible_dates[-1]


def _reset_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def build_calendar_overlay_provider(profile: PaperTradingProfile, provider_uri: Path) -> Path:
    base_calendar = read_calendar_dates(provider_uri)
    next_business_day = (base_calendar[-1] + pd.tseries.offsets.BDay(1)).normalize()
    overlay_uri = (profile.resolved_output_root / "_provider_overlay").resolve()
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
    if overlay_lines[-1] != next_business_day.strftime("%Y-%m-%d"):
        overlay_lines.append(next_business_day.strftime("%Y-%m-%d"))
    (calendars_dir / "day.txt").write_text("\n".join(overlay_lines) + "\n")
    return overlay_uri


def refresh_provider_data(profile: PaperTradingProfile, target_date: pd.Timestamp) -> int:
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


def init_qlib(provider_uri: Path, region: str) -> None:
    LOGGER.info("Initialize Qlib for paper trading, provider uri: %s", provider_uri)
    qlib.init(provider_uri=str(provider_uri), region=region)


def _dynamic_task_config(profile: PaperTradingProfile, provider_uri: Path, effective_end: pd.Timestamp) -> dict:
    spec = profile.combo_spec
    task_cfg = build_task_config(
        handler_key=spec["handler"],
        model_key=spec["model"],
        instruments=[],
        max_instruments=spec.get("max_instruments"),
        infer_processors=spec.get("infer_processors"),
        model_kwargs_override=profile.resolved_model_kwargs(),
    )
    from tw_workflow.settings import BENCHMARK, load_full_universe

    all_codes = load_full_universe(provider_uri, benchmark=BENCHMARK)
    universe = [code for code in all_codes if not code.startswith("^")]
    task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"] = universe
    handler_kwargs = task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]
    handler_kwargs["start_time"] = profile.data_start
    handler_kwargs["end_time"] = effective_end.strftime("%Y-%m-%d")
    fit_end_time = pd.Timestamp(handler_kwargs["fit_end_time"])
    if fit_end_time > effective_end:
        handler_kwargs["fit_end_time"] = effective_end.strftime("%Y-%m-%d")
    segments = task_cfg["dataset"]["kwargs"]["segments"]
    segments["test"] = (profile.backtest_start, effective_end.strftime("%Y-%m-%d"))
    return task_cfg


def _dynamic_port_config(profile: PaperTradingProfile, task_cfg: dict, effective_end: pd.Timestamp) -> dict:
    port_config = build_port_analysis_config()
    port_config["backtest"]["start_time"] = profile.backtest_start
    port_config["backtest"]["end_time"] = effective_end.strftime("%Y-%m-%d")
    port_config["backtest"]["account"] = profile.account
    exchange_kwargs = port_config["backtest"]["exchange_kwargs"]
    exchange_kwargs["open_cost"] = profile.open_cost
    exchange_kwargs["close_cost"] = profile.close_cost
    exchange_kwargs["min_cost"] = profile.min_cost
    exchange_kwargs["odd_lot_min_cost"] = profile.odd_lot_min_cost
    exchange_kwargs["board_lot_size"] = profile.board_lot_size
    exchange_kwargs["trade_unit"] = profile.trade_unit
    apply_strategy_overrides(
        port_config,
        task_cfg,
        n_drop_override=profile.n_drop,
        topk_override=profile.topk,
        rebalance=profile.rebalance,
        strategy_choice=profile.strategy,
        deal_price=profile.deal_price,
        limit_tplus=profile.limit_tplus,
        limit_slippage=profile.limit_slippage,
    )
    port_config["strategy"]["kwargs"]["risk_degree"] = profile.risk_degree
    if profile.limit_tplus:
        tplus_exchange_kwargs = port_config["backtest"]["exchange_kwargs"]["exchange"]["kwargs"]
        tplus_exchange_kwargs["limit_slippage"] = profile.limit_slippage
        tplus_exchange_kwargs["settlement_lag"] = profile.settlement_lag
    return port_config


def run_daily_replay(
    profile: PaperTradingProfile,
    target_date: pd.Timestamp,
    provider_uri: Path,
    effective_end: pd.Timestamp,
) -> tuple[str, str, pd.Timestamp]:
    task_cfg = _dynamic_task_config(profile, provider_uri, effective_end)
    port_config = _dynamic_port_config(profile, task_cfg, effective_end)
    dataset = init_instance_by_config(task_cfg["dataset"])
    train_recorder = R.get_recorder(experiment_name=profile.train_experiment, recorder_id=profile.train_recorder_id)
    trained_model = train_recorder.load_object("trained_model")
    experiment_name = f"paper_replay_{profile.name}"
    LOGGER.info("Start paper replay for %s up to %s", profile.name, effective_end.strftime("%Y-%m-%d"))
    with R.start(experiment_name=experiment_name):
        current_recorder = R.get_recorder()
        current_recorder.set_tags(profile=profile.name, requested_target_date=target_date.strftime("%Y-%m-%d"))
        current_recorder.log_params(
            combo=profile.combo,
            strategy=profile.strategy,
            topk=profile.topk,
            n_drop=profile.n_drop,
            requested_deal_price=profile.deal_price,
            effective_deal_price=profile.effective_deal_price,
            limit_tplus=profile.limit_tplus,
            settlement_lag=profile.settlement_lag,
            effective_end=effective_end.strftime("%Y-%m-%d"),
        )
        signal_rec = SignalRecord(trained_model, dataset, current_recorder)
        signal_rec.generate()
        strategy_kwargs = port_config["strategy"]["kwargs"]
        strategy_kwargs["model"] = trained_model
        strategy_kwargs["dataset"] = dataset
        port_rec = PortAnaRecord(current_recorder, port_config, "day")
        port_rec.generate()
        recorder_id = current_recorder.id
    return experiment_name, recorder_id, effective_end


def export_replay_outputs(
    profile: PaperTradingProfile,
    experiment_name: str,
    recorder_id: str,
    requested_target_date: pd.Timestamp,
    calendar_dates: list[pd.Timestamp],
    active_provider_uri: Path,
) -> dict:
    recorder = R.get_recorder(experiment_name=experiment_name, recorder_id=recorder_id)
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions_dict = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    indicator_obj = recorder.load_object("portfolio_analysis/indicators_normal_1day_obj.pkl")
    pred_df = recorder.load_object("pred.pkl")
    metadata = profile.to_metadata()
    metadata.update(
        {
            "requested_target_date": requested_target_date.strftime("%Y-%m-%d"),
            "paper_experiment": experiment_name,
            "paper_recorder_id": recorder_id,
            "active_provider_uri": str(active_provider_uri),
        }
    )
    outputs = extract_outputs(
        profile=profile,
        report_df=report_df,
        positions_dict=positions_dict,
        indicator_obj=indicator_obj,
        pred_df=pred_df,
        calendar_dates=calendar_dates,
        metadata=metadata,
    )
    return write_outputs(build_paths(profile.resolved_output_root), outputs, metadata)


def run_paper_trading_cycle(profile: PaperTradingProfile, target_date: pd.Timestamp, refresh_data: bool = True) -> dict:
    if refresh_data:
        exit_code = refresh_provider_data(profile, target_date)
        if exit_code != 0:
            raise RuntimeError(f"Data refresh failed with exit code {exit_code}")
    base_provider_uri = profile.resolved_provider_uri
    effective_end = latest_trade_date_on_or_before(base_provider_uri, target_date)
    run_provider_uri = base_provider_uri
    if effective_end == latest_calendar_date(base_provider_uri):
        run_provider_uri = build_calendar_overlay_provider(profile, base_provider_uri)
        LOGGER.info(
            "Using calendar overlay provider %s to expose next-session boundary after %s",
            run_provider_uri,
            effective_end.strftime("%Y-%m-%d"),
        )
    init_qlib(run_provider_uri, profile.region)
    experiment_name, recorder_id, _effective_end = run_daily_replay(
        profile,
        target_date,
        run_provider_uri,
        effective_end,
    )
    return export_replay_outputs(
        profile,
        experiment_name,
        recorder_id,
        target_date,
        calendar_dates=read_calendar_dates(run_provider_uri),
        active_provider_uri=run_provider_uri,
    )
