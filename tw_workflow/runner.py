from __future__ import annotations

import logging
from typing import Dict, List

import qlib
from qlib.utils import flatten_dict, init_instance_by_config
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SignalRecord

from tw_workflow.builders import apply_strategy_overrides, build_port_analysis_config, build_task_config
from tw_workflow.paths import build_effective_name, set_output_dirs
from tw_workflow.reports import dump_report_frames
from tw_workflow.settings import PROVIDER_URI, REGION, UNIVERSE


LOGGER = logging.getLogger("tw_workflow")


def init_qlib() -> None:
    LOGGER.info("Initialize Qlib, provider uri: %s", PROVIDER_URI)
    qlib.init(provider_uri=str(PROVIDER_URI), region=REGION)
    LOGGER.info("Universe size: %d symbols", len(UNIVERSE))


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
) -> str:
    task_config = build_task_config(handler_key, model_key, UNIVERSE, max_instruments, infer_processors)
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
        model_rid = R.get_recorder().id
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
    limit_tplus: bool = False,
    recorder_override: str | None = None,
) -> None:
    effective_name = build_effective_name(
        combo_name,
        n_drop_override=n_drop_override,
        topk_override=topk_override,
        rebalance=rebalance,
        strategy_choice=strategy_choice,
        deal_price=deal_price,
        limit_tplus=limit_tplus,
    )
    paths = set_output_dirs(effective_name)

    task_config = build_task_config(handler_key, model_key, UNIVERSE, max_instruments, infer_processors)
    port_config = build_port_analysis_config()
    apply_strategy_overrides(
        port_config,
        task_config,
        n_drop_override=n_drop_override,
        topk_override=topk_override,
        rebalance=rebalance,
        strategy_choice=strategy_choice,
        deal_price=deal_price,
        limit_tplus=limit_tplus,
    )

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


def run_combo(
    combo_name: str,
    handler_key: str,
    model_key: str,
    max_instruments: int | None,
    infer_processors: List[Dict[str, object]] | None = None,
    *,
    n_drop_override: int | None = None,
    topk_override: int | None = None,
    rebalance: str = "day",
    strategy_choice: str = "bucket",
    deal_price: str = "close",
    simulate_limit: bool = False,
    limit_slippage: float = 0.01,
) -> None:
    available = len(UNIVERSE)
    if max_instruments is None:
        LOGGER.info(
            "=== Running combo: %s (handler=%s, model=%s, instruments=%d) ===",
            combo_name,
            handler_key,
            model_key,
            available,
        )
    else:
        LOGGER.info(
            "=== Running combo: %s (handler=%s, model=%s, instruments=%d/%d) ===",
            combo_name,
            handler_key,
            model_key,
            min(available, max_instruments),
            available,
        )

    effective_name = build_effective_name(
        combo_name,
        n_drop_override=n_drop_override,
        topk_override=topk_override,
        rebalance=rebalance,
        strategy_choice=strategy_choice,
        deal_price=deal_price,
        simulate_limit=simulate_limit,
        limit_slippage=limit_slippage,
    )
    paths = set_output_dirs(effective_name)

    task_config = build_task_config(handler_key, model_key, UNIVERSE, max_instruments, infer_processors)
    port_config = build_port_analysis_config()
    apply_strategy_overrides(
        port_config,
        task_config,
        n_drop_override=n_drop_override,
        topk_override=topk_override,
        rebalance=rebalance,
        strategy_choice=strategy_choice,
        deal_price=deal_price,
        simulate_limit=simulate_limit,
        limit_slippage=limit_slippage,
    )

    LOGGER.info("Build model and dataset configuration")
    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    train_exp = f"tw_train_model_{combo_name}"
    LOGGER.info("Start model training - experiment %s", train_exp)
    with R.start(experiment_name=train_exp):
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        model_rid = R.get_recorder().id
    LOGGER.info("Model training complete, recorder id = %s", model_rid)

    backtest_exp = f"tw_backtest_{combo_name}"
    LOGGER.info("Start signal generation and backtest - experiment %s", backtest_exp)
    with R.start(experiment_name=backtest_exp):
        train_recorder = R.get_recorder(recorder_id=model_rid, experiment_name=train_exp)
        trained_model = train_recorder.load_object("trained_model")

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
