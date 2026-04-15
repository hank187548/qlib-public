from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

from qlib_tw.research.settings import (
    BASE_DATA_HANDLER_CONFIG,
    BASE_PORT_ANALYSIS_CONFIG,
    HANDLER_CONFIGS,
    MODEL_CONFIGS,
    SEGMENTS,
)


def build_task_config(
    handler_key: str,
    model_key: str,
    instruments: List[str],
    max_instruments: int | None = None,
    infer_processors: List[Dict[str, object]] | None = None,
    model_kwargs_override: Dict[str, object] | None = None,
) -> Dict[str, object]:
    handler_spec = HANDLER_CONFIGS[handler_key]
    model_spec = deepcopy(MODEL_CONFIGS[model_key])
    if model_kwargs_override:
        model_spec["kwargs"].update(deepcopy(model_kwargs_override))
    handler_kwargs = deepcopy(BASE_DATA_HANDLER_CONFIG)
    selected_instruments = instruments if max_instruments is None else instruments[:max_instruments]
    handler_kwargs["instruments"] = selected_instruments
    if infer_processors is not None:
        handler_kwargs["infer_processors"] = deepcopy(infer_processors)
    dataset_cfg = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": handler_spec["class"],
                "module_path": handler_spec["module_path"],
                "kwargs": handler_kwargs,
            },
            "segments": deepcopy(SEGMENTS),
        },
    }
    return {"model": model_spec, "dataset": dataset_cfg}


def build_port_analysis_config() -> Dict[str, object]:
    return deepcopy(BASE_PORT_ANALYSIS_CONFIG)


def apply_strategy_overrides(
    port_config: Dict[str, Any],
    task_cfg: Dict[str, Any],
    *,
    n_drop_override: int | None = None,
    topk_override: int | None = None,
    rebalance: str = "day",
    strategy_choice: str = "bucket",
    deal_price: str = "close",
    simulate_limit: bool = False,
    limit_slippage: float = 0.01,
    limit_tplus: bool = False,
    adjust_prices_for_backtest: bool = False,
) -> Dict[str, Any]:
    if strategy_choice == "equal":
        port_config["strategy"]["class"] = "TopkDropoutStrategy"
        port_config["strategy"]["module_path"] = "qlib.contrib.strategy.signal_strategy"
    if n_drop_override is not None:
        port_config["strategy"]["kwargs"]["n_drop"] = n_drop_override
    if topk_override is not None:
        port_config["strategy"]["kwargs"]["topk"] = topk_override
    if rebalance != "day":
        port_config["executor"]["kwargs"]["time_per_step"] = rebalance
        port_config["backtest"]["exchange_kwargs"]["freq"] = rebalance

    port_config["backtest"]["exchange_kwargs"]["deal_price"] = deal_price

    if simulate_limit and limit_tplus:
        raise ValueError("simulate_limit and limit_tplus cannot be enabled together")
    if adjust_prices_for_backtest and not (simulate_limit or limit_tplus):
        raise ValueError("adjust_prices_for_backtest currently requires simulate_limit or limit_tplus")

    if simulate_limit:
        base_ex_kwargs = deepcopy(port_config["backtest"]["exchange_kwargs"])
        exchange_cfg = {
            "class": "TWLimitExchange",
            "module_path": "qlib_tw.trade.custom_exchange",
            "kwargs": {
                **base_ex_kwargs,
                "start_time": port_config["backtest"]["start_time"],
                "end_time": port_config["backtest"]["end_time"],
                "codes": task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"],
                "limit_slippage": limit_slippage,
                "adjust_prices_for_backtest": adjust_prices_for_backtest,
            },
        }
        port_config["backtest"]["exchange_kwargs"] = {"exchange": exchange_cfg}

    if limit_tplus:
        base_ex_kwargs = deepcopy(port_config["backtest"]["exchange_kwargs"])
        base_ex_kwargs["deal_price"] = "open"
        exchange_cfg = {
            "class": "TPlusLimitExchange",
            "module_path": "qlib_tw.trade.custom_exchange",
            "kwargs": {
                **base_ex_kwargs,
                "start_time": port_config["backtest"]["start_time"],
                "end_time": port_config["backtest"]["end_time"],
                "codes": task_cfg["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"],
                "settlement_lag": 2,
                "adjust_prices_for_backtest": adjust_prices_for_backtest,
            },
        }
        port_config["backtest"]["exchange_kwargs"] = {"exchange": exchange_cfg}

    return port_config
