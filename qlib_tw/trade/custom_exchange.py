#!/usr/bin/env python3
"""
Custom exchange implementation for Taiwan equities.

The research path uses a single `TWExchange` model:
- execute on the configured base deal price (`open`/`close`)
- apply Taiwan-specific fee logic, including board-lot vs odd-lot minimum fees
- apply T+N settlement timing on both cash and shares

The provider used by this repo already stores adjusted prices in the core OHLCV
fields, so exchange-side price adjustment is intentionally disabled.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.exchange import Exchange


class TWExchange(Exchange):
    """
    Taiwan exchange model with:
    - TW fee model (board lot / odd lot minimum fees)
    - adjusted-price provider passthrough
    - T+N settlement for shares and cash
    """

    def __init__(
        self,
        *args,
        settlement_lag: int = 2,
        odd_lot_min_cost: float = 1.0,
        board_lot_size: int = 1000,
        adjust_prices_for_backtest: bool = False,
        **kwargs,
    ):
        if adjust_prices_for_backtest:
            kwargs = dict(kwargs)
            self._ignored_adjust_prices_flag = True
        else:
            self._ignored_adjust_prices_flag = False

        super().__init__(
            *args,
            adjust_prices_for_backtest=False,
            price_basis="provider",
            factor_semantics="price_only",
            **kwargs,
        )
        self.settlement_lag = int(settlement_lag)
        self.odd_lot_min_cost = float(odd_lot_min_cost)
        self.board_lot_size = int(board_lot_size)
        if self._ignored_adjust_prices_flag:
            self.logger.info(
                "Provider core price fields are already adjusted; ignore adjust_prices_for_backtest."
            )

    def get_factor(
        self,
        stock_id: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ):
        if stock_id not in self.quote.get_all_stock():
            return None
        # In this repo, $factor comes from adj_close / close and is price-only.
        # Returning 1.0 keeps share rounding on the raw-share basis.
        return 1.0

    def _raw_share_count(self, deal_amount: float, factor: float | None) -> int:
        if factor is None or np.isnan(factor) or factor <= 0:
            return max(int(round(deal_amount)), 0)
        return max(int(round(deal_amount * factor)), 0)

    def _calc_tw_trade_cost(self, *, trade_val: float, deal_amount: float, cost_ratio: float, factor: float | None) -> float:
        if trade_val <= 1e-5 or deal_amount <= 1e-8:
            return 0.0
        raw_shares = self._raw_share_count(deal_amount, factor)
        if raw_shares <= 0:
            return max(trade_val * cost_ratio, self.odd_lot_min_cost)

        board_shares = 0
        odd_shares = raw_shares
        if self.board_lot_size > 0:
            board_shares = (raw_shares // self.board_lot_size) * self.board_lot_size
            odd_shares = raw_shares - board_shares

        total_cost = 0.0
        if board_shares > 0:
            board_trade_val = trade_val * (board_shares / raw_shares)
            total_cost += max(board_trade_val * cost_ratio, self.min_cost)
        if odd_shares > 0:
            odd_trade_val = trade_val * (odd_shares / raw_shares)
            total_cost += max(odd_trade_val * cost_ratio, self.odd_lot_min_cost)
        return total_cost

    def _get_buy_amount_by_cash_limit(self, trade_price: float, cash: float, cost_ratio: float, factor: float | None = None) -> float:
        if trade_price <= 0 or cash <= 0:
            return 0.0
        low = 0.0
        high = self.round_amount_by_trade_unit(cash / trade_price, factor)
        best = 0.0
        for _ in range(40):
            if high - low <= 1e-8:
                break
            mid = self.round_amount_by_trade_unit((low + high) / 2.0, factor)
            if mid <= best + 1e-8:
                break
            trade_val = mid * trade_price
            trade_cost = self._calc_tw_trade_cost(
                trade_val=trade_val,
                deal_amount=mid,
                cost_ratio=cost_ratio,
                factor=factor,
            )
            if trade_val + trade_cost <= cash + 1e-12:
                best = mid
                low = mid
            else:
                high = mid
        return best

    def _calc_trade_info_core(self, order, position, dealt_order_amount, trade_price):
        if trade_price is None or np.isnan(trade_price) or trade_price <= 0:
            order.deal_amount = 0.0
            return np.nan, 0.0, 0.0

        total_trade_val = float(self.get_volume(order.stock_id, order.start_time, order.end_time)) * trade_price
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        order.deal_amount = order.amount
        self._clip_amount_by_volume(order, dealt_order_amount)

        trade_val = order.deal_amount * trade_price
        if not total_trade_val or np.isnan(total_trade_val):
            adj_cost_ratio = self.impact_cost
        else:
            adj_cost_ratio = self.impact_cost * (trade_val / total_trade_val) ** 2

        if order.direction == Order.SELL:
            cost_ratio = self.close_cost + adj_cost_ratio
            if position is not None:
                current_amount = position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
                if not np.isclose(order.deal_amount, current_amount):
                    order.deal_amount = self.round_amount_by_trade_unit(min(current_amount, order.deal_amount), order.factor)
                sell_trade_cost = self._calc_tw_trade_cost(
                    trade_val=order.deal_amount * trade_price,
                    deal_amount=order.deal_amount,
                    cost_ratio=cost_ratio,
                    factor=order.factor,
                )
                if position.get_cash() + order.deal_amount * trade_price < sell_trade_cost:
                    order.deal_amount = 0
        elif order.direction == Order.BUY:
            cost_ratio = self.open_cost + adj_cost_ratio
            if position is not None:
                cash = position.get_cash()
                trade_val = order.deal_amount * trade_price
                buy_trade_cost = self._calc_tw_trade_cost(
                    trade_val=trade_val,
                    deal_amount=order.deal_amount,
                    cost_ratio=cost_ratio,
                    factor=order.factor,
                )
                if cash < buy_trade_cost:
                    order.deal_amount = 0
                elif cash < trade_val + buy_trade_cost:
                    max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, cost_ratio, order.factor)
                    order.deal_amount = self.round_amount_by_trade_unit(min(max_buy_amount, order.deal_amount), order.factor)
                else:
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
            else:
                order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
        else:
            raise NotImplementedError(f"order direction {order.direction} error")

        trade_val = order.deal_amount * trade_price
        trade_cost = self._calc_tw_trade_cost(
            trade_val=trade_val,
            deal_amount=order.deal_amount,
            cost_ratio=cost_ratio,
            factor=order.factor,
        )
        return float(trade_price), float(trade_val), float(trade_cost)

    @staticmethod
    def _release_pending(position, current_dt: pd.Timestamp):
        pend_cash = getattr(position, "_pending_cash", [])
        still_cash = []
        cash_delay_total = 0.0
        for rel_dt, amt in pend_cash:
            if rel_dt <= current_dt:
                position.position["cash"] += amt
            else:
                still_cash.append((rel_dt, amt))
                cash_delay_total += amt
        if cash_delay_total > 0:
            position.position["cash_delay"] = cash_delay_total
        elif "cash_delay" in position.position:
            del position.position["cash_delay"]
        position._pending_cash = still_cash

        pend_stock = getattr(position, "_pending_stock", {})
        to_delete = []
        for code, items in pend_stock.items():
            remaining = []
            for rel_dt, amt, price in items:
                if rel_dt > current_dt:
                    remaining.append((rel_dt, amt, price))
            if remaining:
                pend_stock[code] = remaining
            else:
                to_delete.append(code)
        for code in to_delete:
            del pend_stock[code]
        position._pending_stock = pend_stock

    def _settle_date(self, dt: pd.Timestamp) -> pd.Timestamp:
        return (dt + pd.tseries.offsets.BDay(self.settlement_lag)).normalize()

    def _available_amount(self, pos, code: str, current_dt: pd.Timestamp) -> float:
        total = pos.get_stock_amount(code)
        locked = 0.0
        for _, amt, _ in getattr(pos, "_pending_stock", {}).get(code, []):
            locked += amt
        return max(total - locked, 0.0)

    def deal_order(
        self,
        order: Order,
        trade_account=None,
        position=None,
        dealt_order_amount=defaultdict(float),
    ):
        if trade_account is not None and position is not None:
            raise ValueError("trade_account and position can only choose one")
        if trade_account is None and position is None:
            raise ValueError("position is required for TWExchange")

        pos = trade_account.current_position if trade_account else position

        current_dt = order.start_time.normalize()
        self._release_pending(pos, current_dt)

        if order.direction == Order.SELL:
            avail = self._available_amount(pos, order.stock_id, current_dt)
            if avail <= 0:
                order.deal_amount = 0.0
                return 0.0, 0.0, np.nan
            order.amount = min(order.amount, avail)

        trade_price = self.get_deal_price(
            order.stock_id,
            order.start_time,
            order.end_time,
            direction=order.direction,
        )
        trade_price, trade_val, trade_cost = self._calc_trade_info_core(
            order,
            pos,
            dealt_order_amount,
            trade_price,
        )
        if trade_val <= 1e-5:
            return float(trade_val), float(trade_cost), float(trade_price)

        pre_cash = pos.get_cash()
        pre_amt = pos.get_stock_amount(order.stock_id)

        if trade_account:
            trade_account.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)
        else:
            pos.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)

        settle_dt = self._settle_date(current_dt)

        if order.direction == Order.BUY:
            post_amt = pos.get_stock_amount(order.stock_id)
            delta_amt = max(post_amt - pre_amt, 0.0)
            if delta_amt > 0:
                pend_stock = getattr(pos, "_pending_stock", {})
                pend_stock.setdefault(order.stock_id, []).append((settle_dt, delta_amt, trade_price))
                pos._pending_stock = pend_stock
        elif order.direction == Order.SELL:
            post_cash = pos.get_cash()
            cash_in = post_cash - pre_cash
            if cash_in > 0:
                pos.position["cash"] -= cash_in
                pend_cash = getattr(pos, "_pending_cash", [])
                pend_cash.append((settle_dt, cash_in))
                pos._pending_cash = pend_cash
                pos.position["cash_delay"] = pos.position.get("cash_delay", 0.0) + cash_in
        else:
            raise NotImplementedError(f"direction {order.direction} is not supported")

        return float(trade_val), float(trade_cost), float(trade_price)
