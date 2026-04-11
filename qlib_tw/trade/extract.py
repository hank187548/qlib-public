from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
from qlib.data import D

from qlib_tw.research.builders import build_port_analysis_config
from qlib_tw.trade.config import PaperTradingProfile
from qlib_tw.trade.state import snapshot_from_cash, snapshot_from_position


def _single_data_to_series(metric) -> pd.Series:
    if hasattr(metric, "to_dict"):
        return pd.Series(metric.to_dict(), dtype="float64")
    if hasattr(metric, "items"):
        return pd.Series(dict(metric.items()), dtype="float64")
    return pd.Series(metric, dtype="float64")


def default_bucket_weights(topk: int) -> list[float]:
    weights = [0.04] * 10 + [0.02] * 20 + [0.01] * 20
    if len(weights) < topk:
        weights += [1.0 / topk] * (topk - len(weights))
    return weights[:topk]


def fetch_close_prices(codes: Iterable[str], dt: pd.Timestamp) -> Dict[str, float]:
    codes = sorted({str(code) for code in codes})
    if not codes:
        return {}
    df = D.features(codes, ["$close"], start_time=dt, end_time=dt)
    price_map: Dict[str, float] = {}
    if df.empty:
        return price_map
    for (instrument, _date), row in df.iterrows():
        price = row["$close"]
        if pd.notna(price):
            price_map[str(instrument)] = float(price)
    return price_map


def infer_next_trade_date(trade_date: pd.Timestamp, calendar_dates: Iterable[pd.Timestamp]) -> pd.Timestamp:
    trade_date = pd.Timestamp(trade_date).normalize()
    normalized = sorted(pd.Timestamp(dt).normalize() for dt in calendar_dates)
    for dt in normalized:
        if dt > trade_date:
            return dt
    return (trade_date + pd.tseries.offsets.BDay(1)).normalize()


def build_nav_history_dataframe(report_df: pd.DataFrame, positions_dict: Dict[pd.Timestamp, object]) -> pd.DataFrame:
    report = report_df.reset_index().rename(columns={"index": "datetime"}).copy()
    report["datetime"] = pd.to_datetime(report["datetime"]).dt.normalize()
    rows = []
    for record in report.to_dict(orient="records"):
        dt = pd.Timestamp(record["datetime"]).normalize()
        position = positions_dict.get(dt)
        cash_delay = 0.0
        holdings = 0
        if position is not None:
            cash_delay = float(position.position.get("cash_delay", 0.0))
            holdings = len(position.get_stock_list())
        rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "account_value": float(record["account"]),
                "market_value": float(record["value"]),
                "cash": float(record["cash"]),
                "cash_delay": cash_delay,
                "holdings_count": holdings,
                "daily_return": float(record["return"]),
                "benchmark_return": float(record["bench"]),
                "cost_rate": float(record["cost"]),
                "excess_return_without_cost": float(record["return"] - record["bench"]),
                "excess_return_with_cost": float(record["return"] - record["bench"] - record["cost"]),
                "turnover": float(record["turnover"]),
                "total_turnover": float(record["total_turnover"]),
                "total_cost": float(record["total_cost"]),
            }
        )
    return pd.DataFrame(rows)


def build_fills_dataframe(indicator_obj, trade_date: pd.Timestamp) -> pd.DataFrame:
    trade_date = pd.Timestamp(trade_date).normalize()
    order_indicator = indicator_obj.order_indicator_his[trade_date]
    amount = _single_data_to_series(order_indicator.data["amount"])
    deal_amount = _single_data_to_series(order_indicator.data["deal_amount"])
    trade_price = _single_data_to_series(order_indicator.data["trade_price"])
    trade_cost = _single_data_to_series(order_indicator.data["trade_cost"])
    trade_dir = _single_data_to_series(order_indicator.data["trade_dir"])
    fill_rate = _single_data_to_series(order_indicator.data["ffr"])

    all_codes = sorted(set(amount.index) | set(deal_amount.index) | set(trade_dir.index))
    rows = []
    for code in all_codes:
        requested_qty = abs(float(amount.get(code, 0.0)))
        filled_qty = abs(float(deal_amount.get(code, 0.0)))
        direction = int(trade_dir.get(code, 1.0))
        fill_price = float(trade_price.get(code, 0.0))
        fill_cost = float(trade_cost.get(code, 0.0))
        rate = float(fill_rate.get(code, 0.0))
        if filled_qty <= 0:
            status = "unfilled"
        elif abs(rate - 1.0) <= 1e-8:
            status = "filled"
        else:
            status = "partial"
        rows.append(
            {
                "trade_date": trade_date.strftime("%Y-%m-%d"),
                "instrument": str(code),
                "side": "buy" if direction > 0 else "sell",
                "requested_qty": requested_qty,
                "filled_qty": filled_qty,
                "fill_rate": rate,
                "fill_price": fill_price if filled_qty > 0 else None,
                "filled_notional": filled_qty * fill_price if filled_qty > 0 else 0.0,
                "trade_cost": fill_cost,
                "status": status,
            }
        )
    return pd.DataFrame(rows).sort_values(["side", "instrument"]).reset_index(drop=True)


def build_empty_fills_dataframe(trade_date: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trade_date",
            "instrument",
            "side",
            "requested_qty",
            "filled_qty",
            "fill_rate",
            "fill_price",
            "filled_notional",
            "trade_cost",
            "status",
        ]
    )


class _CashOnlyPosition:
    def __init__(self, cash: float):
        self._cash = float(cash)
        self.position = {"cash": self._cash}
        self._pending_cash = []
        self._pending_stock = {}

    def get_stock_list(self):
        return []

    def get_stock_amount(self, _code):
        return 0.0

    def get_cash(self):
        return self._cash


def _locked_stock_map(position) -> Dict[str, float]:
    locked: Dict[str, float] = {}
    for code, entries in getattr(position, "_pending_stock", {}).items():
        locked[str(code)] = sum(float(amount) for _release, amount, _price in entries)
    return locked


def _compute_next_orders(
    *,
    pred_series: pd.Series,
    position,
    profile: PaperTradingProfile,
    signal_date: pd.Timestamp,
    next_trade_date: pd.Timestamp,
) -> pd.DataFrame:
    pred_series = pred_series.sort_values(ascending=False)
    current_codes = pd.Index(position.get_stock_list())
    last = pred_series.reindex(current_codes).sort_values(ascending=False).index
    candidate_count = max(profile.n_drop + profile.topk - len(last), 0)
    today = pred_series[~pred_series.index.isin(last)].sort_values(ascending=False).index[:candidate_count]
    comb = pred_series.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
    sell_set = set(comb[-profile.n_drop :]) if profile.n_drop > 0 else set()
    sell = [code for code in last if code in sell_set]
    buy = list(today[: len(sell) + profile.topk - len(last)])

    close_prices = fetch_close_prices(set(sell) | set(buy), signal_date)
    close_cost = float(profile.close_cost)
    locked_map = _locked_stock_map(position)
    qty_map = {str(code): float(position.get_stock_amount(code)) for code in position.get_stock_list()}
    rank_index = pred_series.index[: profile.topk]
    if profile.strategy == "bucket":
        rank_weights = default_bucket_weights(profile.topk)
    else:
        rank_weights = [1.0 / profile.topk] * profile.topk
    weight_map = {str(code): rank_weights[idx] for idx, code in enumerate(rank_index)}

    est_cash = float(position.get_cash())
    sell_rows = []
    for code in sell:
        code = str(code)
        current_qty = qty_map.get(code, 0.0)
        locked_qty = locked_map.get(code, 0.0)
        sellable_qty = max(current_qty - locked_qty, 0.0)
        blocked_by_tplus = current_qty > 0 and sellable_qty <= 0 and locked_qty > 0
        price_ref = close_prices.get(code)
        est_notional = sellable_qty * price_ref if price_ref is not None else None
        if est_notional is not None:
            est_cash += est_notional * (1.0 - close_cost)
        sell_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "intended_trade_date": next_trade_date.strftime("%Y-%m-%d"),
                "instrument": code,
                "side": "sell",
                "score": float(pred_series.get(code, float("nan"))),
                "rank": int(pred_series.index.get_loc(code)) + 1 if code in pred_series.index else None,
                "current_qty": current_qty,
                "sellable_qty": sellable_qty,
                "locked_qty": locked_qty,
                "target_weight": 0.0,
                "target_value_est": est_notional,
                "price_reference": price_ref,
                "quantity_basis": "exact_from_current_state",
                "order_qty_est": sellable_qty,
                "requires_open_reprice": False,
                "blocked_by_tplus": blocked_by_tplus,
                "price_model": "next_open_with_intraday_limit_check" if profile.limit_tplus else profile.effective_deal_price,
                "note": "dropout_sell_blocked_by_tplus" if blocked_by_tplus else "dropout_sell",
            }
        )

    buy_weights = [weight_map.get(str(code), 0.0) for code in buy]
    total_buy_weight = sum(buy_weights)
    buy_rows = []
    for code, bucket_weight in zip(buy, buy_weights):
        code = str(code)
        price_ref = close_prices.get(code)
        target_value = est_cash * profile.risk_degree * (bucket_weight / total_buy_weight) if total_buy_weight > 0 else 0.0
        order_qty_est = None
        if price_ref and price_ref > 0:
            order_qty_est = int(target_value // price_ref)
        buy_rows.append(
            {
                "signal_date": signal_date.strftime("%Y-%m-%d"),
                "intended_trade_date": next_trade_date.strftime("%Y-%m-%d"),
                "instrument": code,
                "side": "buy",
                "score": float(pred_series.get(code, float("nan"))),
                "rank": int(pred_series.index.get_loc(code)) + 1 if code in pred_series.index else None,
                "current_qty": qty_map.get(code, 0.0),
                "sellable_qty": max(qty_map.get(code, 0.0) - locked_map.get(code, 0.0), 0.0),
                "locked_qty": locked_map.get(code, 0.0),
                "target_weight": bucket_weight,
                "target_value_est": target_value,
                "price_reference": price_ref,
                "quantity_basis": "estimated_from_latest_close",
                "order_qty_est": order_qty_est,
                "requires_open_reprice": True,
                "blocked_by_tplus": False,
                "price_model": "next_open_with_intraday_limit_check" if profile.limit_tplus else profile.effective_deal_price,
                "note": "next_day_open_unknown_in_backtest_model",
            }
        )

    orders = pd.DataFrame(sell_rows + buy_rows)
    if orders.empty:
        return orders
    return orders.sort_values(["side", "rank", "instrument"], na_position="last").reset_index(drop=True)


@dataclass(frozen=True)
class ExtractedOutputs:
    trade_date: pd.Timestamp
    next_trade_date: pd.Timestamp
    nav_history: pd.DataFrame
    fills: pd.DataFrame
    orders_next_day: pd.DataFrame
    state_snapshot: object


def extract_outputs(
    *,
    profile: PaperTradingProfile,
    report_df: pd.DataFrame,
    positions_dict: Dict[pd.Timestamp, object],
    indicator_obj,
    pred_df: pd.DataFrame | pd.Series,
    calendar_dates: Iterable[pd.Timestamp],
    metadata: Dict[str, object],
) -> ExtractedOutputs:
    trade_date = sorted(positions_dict.keys())[-1].normalize()
    next_trade_date = infer_next_trade_date(trade_date, calendar_dates)
    position = positions_dict[trade_date]
    nav_history = build_nav_history_dataframe(report_df, positions_dict)
    fills = build_fills_dataframe(indicator_obj, trade_date)
    if isinstance(pred_df, pd.DataFrame):
        pred_series = pred_df.iloc[:, 0]
    else:
        pred_series = pred_df
    latest_pred_date = pred_series.index.get_level_values(0).max()
    latest_scores = pred_series.loc[pd.IndexSlice[latest_pred_date, :]]
    if isinstance(latest_scores, pd.Series) and isinstance(latest_scores.index, pd.MultiIndex):
        latest_scores.index = latest_scores.index.get_level_values(-1)
    latest_scores.index = latest_scores.index.map(str)
    orders_next_day = _compute_next_orders(
        pred_series=latest_scores,
        position=position,
        profile=profile,
        signal_date=pd.Timestamp(latest_pred_date).normalize(),
        next_trade_date=next_trade_date,
    )
    report_row = nav_history.iloc[-1]
    state_snapshot = snapshot_from_position(
        position=position,
        as_of_date=trade_date,
        next_trade_date=next_trade_date,
        account_value=float(report_row["account_value"]),
        market_value=float(report_row["market_value"]),
        metadata=metadata,
    )
    return ExtractedOutputs(
        trade_date=trade_date,
        next_trade_date=next_trade_date,
        nav_history=nav_history,
        fills=fills,
        orders_next_day=orders_next_day,
        state_snapshot=state_snapshot,
    )


def extract_preview_outputs(
    *,
    profile: PaperTradingProfile,
    pred_df: pd.DataFrame | pd.Series,
    as_of_date: pd.Timestamp,
    calendar_dates: Iterable[pd.Timestamp],
    metadata: Dict[str, object],
) -> ExtractedOutputs:
    trade_date = pd.Timestamp(as_of_date).normalize()
    next_trade_date = infer_next_trade_date(trade_date, calendar_dates)
    if isinstance(pred_df, pd.DataFrame):
        pred_series = pred_df.iloc[:, 0]
    else:
        pred_series = pred_df
    latest_pred_date = pred_series.index.get_level_values(0).max()
    latest_scores = pred_series.loc[pd.IndexSlice[latest_pred_date, :]]
    if isinstance(latest_scores, pd.Series) and isinstance(latest_scores.index, pd.MultiIndex):
        latest_scores.index = latest_scores.index.get_level_values(-1)
    latest_scores.index = latest_scores.index.map(str)

    cash_position = _CashOnlyPosition(profile.account)
    orders_next_day = _compute_next_orders(
        pred_series=latest_scores,
        position=cash_position,
        profile=profile,
        signal_date=pd.Timestamp(latest_pred_date).normalize(),
        next_trade_date=next_trade_date,
    )
    nav_history = pd.DataFrame(
        [
            {
                "date": trade_date.strftime("%Y-%m-%d"),
                "account_value": float(profile.account),
                "market_value": 0.0,
                "cash": float(profile.account),
                "cash_delay": 0.0,
                "holdings_count": 0,
                "daily_return": 0.0,
                "benchmark_return": 0.0,
                "cost_rate": 0.0,
                "excess_return_without_cost": 0.0,
                "excess_return_with_cost": 0.0,
                "turnover": 0.0,
                "total_turnover": 0.0,
                "total_cost": 0.0,
            }
        ]
    )
    fills = build_empty_fills_dataframe(trade_date)
    state_snapshot = snapshot_from_cash(
        as_of_date=trade_date,
        next_trade_date=next_trade_date,
        account_value=float(profile.account),
        metadata=metadata,
    )
    return ExtractedOutputs(
        trade_date=trade_date,
        next_trade_date=next_trade_date,
        nav_history=nav_history,
        fills=fills,
        orders_next_day=orders_next_day,
        state_snapshot=state_snapshot,
    )


def write_outputs(paths, outputs: ExtractedOutputs, metadata: Dict[str, object]) -> Dict[str, Path]:
    trade_day = outputs.trade_date.strftime("%Y-%m-%d")
    latest_dir = paths.latest_dir
    daily_dir = paths.daily_dir(trade_day)
    target_map = {
        "nav_history": "nav_history.csv",
        "orders_next_day": "orders_next_day.csv",
        "paper_state": "paper_state.json",
        "fills": f"fills_{trade_day}.csv",
        "metadata": "metadata.json",
    }
    output_paths = {name: daily_dir / filename for name, filename in target_map.items()}
    output_paths["latest_nav_history"] = latest_dir / target_map["nav_history"]
    output_paths["latest_orders_next_day"] = latest_dir / target_map["orders_next_day"]
    output_paths["latest_paper_state"] = latest_dir / target_map["paper_state"]
    output_paths["latest_fills"] = latest_dir / target_map["fills"]
    output_paths["latest_metadata"] = latest_dir / target_map["metadata"]

    outputs.nav_history.to_csv(output_paths["nav_history"], index=False)
    outputs.nav_history.to_csv(output_paths["latest_nav_history"], index=False)
    outputs.orders_next_day.to_csv(output_paths["orders_next_day"], index=False)
    outputs.orders_next_day.to_csv(output_paths["latest_orders_next_day"], index=False)
    outputs.fills.to_csv(output_paths["fills"], index=False)
    outputs.fills.to_csv(output_paths["latest_fills"], index=False)
    outputs.state_snapshot.write_json(output_paths["paper_state"])
    outputs.state_snapshot.write_json(output_paths["latest_paper_state"])
    metadata_payload = dict(metadata)
    metadata_payload["trade_date"] = outputs.trade_date.strftime("%Y-%m-%d")
    metadata_payload["next_trade_date"] = outputs.next_trade_date.strftime("%Y-%m-%d")
    (output_paths["metadata"]).write_text(pd.Series(metadata_payload).to_json(force_ascii=False, indent=2))
    (output_paths["latest_metadata"]).write_text(pd.Series(metadata_payload).to_json(force_ascii=False, indent=2))
    return output_paths
