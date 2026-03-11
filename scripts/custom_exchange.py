#!/usr/bin/env python3
"""
Custom exchange for a simple limit-order-style fill on Taiwan stocks.

This is a conservative day-level approximation:
- Uses base_price (default $open) and applies limit_slippage to form a limit price.
- Buy: fills only if day's high >= limit price; fill price = min(base_price, limit_price).
- Sell: fills only if day's low <= limit price; fill price = max(base_price, limit_price).
- No order book, no partial-day path, just a coarse filter to avoid assuming "always fills at close".

Note: This does not model partial fills or intraday queueing; it's meant to be stricter than the
default "always at close" assumption but still lightweight.
"""

from __future__ import annotations

from collections import defaultdict
import pandas as pd
import numpy as np

from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order, OrderDir


class TWLimitExchange(Exchange):
    """
    簡化的台股限價模擬：
    - 使用 base_price（預設 $open）乘上 limit_slippage 作為限價。
    - 買：若當日最高價 < 限價則不成交；成交價取 min(base_price, 限價)。
    - 賣：若當日最低價 > 限價則不成交；成交價取 max(base_price, 限價)。
    - 無訂單簿/部分成交/撮合，只是日頻保守近似。
    """

    def __init__(self, *args, limit_slippage: float = 0.01, **kwargs):
        # 確保高低價可用
        sub = kwargs.pop("subscribe_fields", [])
        for fld in ("$high", "$low", "$open"):
            if fld not in sub:
                sub.append(fld)
        super().__init__(*args, subscribe_fields=sub, **kwargs)
        self.limit_slippage = limit_slippage

    def _get_price_with_limit(
        self, stock_id: str, start_time: pd.Timestamp, end_time: pd.Timestamp, direction: OrderDir
    ) -> tuple[float | None, bool]:
        base_price = super().get_deal_price(stock_id, start_time, end_time, direction=direction)
        high = self.quote.get_data(stock_id, start_time, end_time, field="$high", method="ts_data_last")
        low = self.quote.get_data(stock_id, start_time, end_time, field="$low", method="ts_data_last")
        if base_price is None or np.isnan(base_price) or base_price <= 0:
            return None, False
        if direction == Order.BUY:
            if high is None or np.isnan(high) or high <= 0:
                return base_price, False
            limit_price = base_price * (1 + self.limit_slippage)
            if high < limit_price:
                return base_price, False
            return min(base_price, limit_price), True
        else:
            if low is None or np.isnan(low) or low <= 0:
                return base_price, False
            limit_price = base_price * (1 - self.limit_slippage)
            if low > limit_price:
                return base_price, False
            return max(base_price, limit_price), True

    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        trade_price, touched = self._get_price_with_limit(
            order.stock_id, order.start_time, order.end_time, direction=order.direction
        )
        if trade_price is None or not touched:
            order.deal_amount = 0.0
            return np.nan if trade_price is None else float(trade_price), 0.0, 0.0

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
                if position.get_cash() + order.deal_amount * trade_price < max(
                    order.deal_amount * trade_price * cost_ratio, self.min_cost
                ):
                    order.deal_amount = 0
        elif order.direction == Order.BUY:
            cost_ratio = self.open_cost + adj_cost_ratio
            if position is not None:
                cash = position.get_cash()
                trade_val = order.deal_amount * trade_price
                if cash < max(trade_val * cost_ratio, self.min_cost):
                    order.deal_amount = 0
                elif cash < trade_val + max(trade_val * cost_ratio, self.min_cost):
                    max_buy_amount = self._get_buy_amount_by_cash_limit(trade_price, cash, cost_ratio)
                    order.deal_amount = self.round_amount_by_trade_unit(min(max_buy_amount, order.deal_amount), order.factor)
                else:
                    order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
            else:
                order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
        else:
            raise NotImplementedError(f"order direction {order.direction} error")

        trade_val = order.deal_amount * trade_price
        trade_cost = max(trade_val * cost_ratio, self.min_cost)
        if trade_val <= 1e-5:
            trade_cost = 0.0
        return float(trade_price), float(trade_val), float(trade_cost)


class TPlusExchange(Exchange):
    """
    Simple T+N settlement wrapper:
    - 買進：現金立即扣除，股數 T+N 之後才入帳、可賣。
    - 賣出：股數立即扣除，現金 T+N 之後才入帳、可用。
    - 未結算現金會放在 cash_delay 供資產計算，但不影響可用現金。
    - 不覆蓋限價/漲跌停/交易量檢查，只改結算時點。
    """

    def __init__(self, *args, settlement_lag: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.settlement_lag = settlement_lag

    @staticmethod
    def _release_pending(position, current_dt: pd.Timestamp):
        """釋放到期的股數與現金回 Position。"""
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
        # 交易日頻率，簡化用工作日推遲 settlement_lag 天
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
        pos = trade_account.current_position if trade_account else position
        if pos is None:
            raise ValueError("position is required for TPlusExchange")

        current_dt = order.start_time.normalize()
        # 釋放到期款/股
        self._release_pending(pos, current_dt)

        # 檢查持股可用量（排除尚未結算）
        if order.direction == Order.SELL:
            avail = self._available_amount(pos, order.stock_id, current_dt)
            if avail <= 0:
                order.deal_amount = 0.0
                return 0.0, 0.0, np.nan
            order.amount = min(order.amount, avail)

        # 用原本機制算成交價/成本與可成交量
        trade_price, trade_val, trade_cost = self._calc_trade_info_by_order(order, pos, dealt_order_amount)
        if trade_val <= 1e-5:
            return float(trade_price), float(trade_val), float(trade_cost)

        settle_dt = self._settle_date(current_dt)

        if order.direction == Order.BUY:
            # 立即扣現金，股數掛 pending
            pos.position["cash"] -= trade_val + trade_cost
            pend_stock = getattr(pos, "_pending_stock", {})
            pend_stock.setdefault(order.stock_id, []).append((settle_dt, order.deal_amount, trade_price))
            pos._pending_stock = pend_stock
        elif order.direction == Order.SELL:
            # 扣股，現金掛 pending
            current_amt = pos.get_stock_amount(order.stock_id)
            new_amt = current_amt - order.deal_amount
            if new_amt <= 0:
                if order.stock_id in pos.position:
                    pos._del_stock(order.stock_id)
            else:
                pos.position[order.stock_id]["amount"] = new_amt
            pend_cash = getattr(pos, "_pending_cash", [])
            cash_in = trade_val - trade_cost
            pend_cash.append((settle_dt, cash_in))
            pos._pending_cash = pend_cash
            # 更新 cash_delay 方便計算總資產
            pos.position["cash_delay"] = pos.position.get("cash_delay", 0.0) + cash_in
        else:
            raise NotImplementedError(f"direction {order.direction} is not supported")

        return float(trade_val), float(trade_cost), float(trade_price)


class TPlusLimitExchange(TWLimitExchange):
    """
    簡化限價 + T+N 結算：
    - 限價邏輯沿用 TWLimitExchange（base_price * slippage，觸價才成交）
    - 買進：現金立即扣除，股數 T+N 之後才入帳、可賣
    - 賣出：股數立即扣除，現金 T+N 之後才入帳、可用
    """

    def __init__(self, *args, settlement_lag: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
        self.settlement_lag = settlement_lag

    def _available_amount(self, pos, code: str, current_dt: pd.Timestamp) -> float:
        total = pos.get_stock_amount(code)
        locked = 0.0
        for _, amt, _ in getattr(pos, "_pending_stock", {}).get(code, []):
            locked += amt
        return max(total - locked, 0.0)

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

    def deal_order(
        self,
        order: Order,
        trade_account=None,
        position=None,
        dealt_order_amount=defaultdict(float),
    ):
        if trade_account is None and position is None:
            raise ValueError("position is required for TPlusLimitExchange")
        pos = trade_account.current_position if trade_account else position

        current_dt = order.start_time.normalize()
        self._release_pending(pos, current_dt)

        # 僅已結算股數可賣
        if order.direction == Order.SELL:
            avail = self._available_amount(pos, order.stock_id, current_dt)
            if avail <= 0:
                order.deal_amount = 0.0
                return 0.0, 0.0, np.nan
            order.amount = min(order.amount, avail)

        # 先算成交條件
        trade_price, trade_val, trade_cost = TWLimitExchange._calc_trade_info_by_order(
            self, order, pos, dealt_order_amount
        )
        if trade_val <= 1e-5:
            return float(trade_price), float(trade_val), float(trade_cost)

        # 記錄更新前的現金/持股
        pre_cash = pos.get_cash()
        pre_amt = pos.get_stock_amount(order.stock_id)

        # 讓 trade_account 記錄 turnover/成本等
        if trade_account:
            trade_account.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)
        else:
            pos.update_order(order=order, trade_val=trade_val, cost=trade_cost, trade_price=trade_price)

        settle_dt = self._settle_date(current_dt)

        if order.direction == Order.BUY:
            # 將新買股標記為待結算，不影響 turnover 計算
            post_amt = pos.get_stock_amount(order.stock_id)
            delta_amt = max(post_amt - pre_amt, 0.0)
            if delta_amt > 0:
                pend_stock = getattr(pos, "_pending_stock", {})
                pend_stock.setdefault(order.stock_id, []).append((settle_dt, delta_amt, trade_price))
                pos._pending_stock = pend_stock
        elif order.direction == Order.SELL:
            # 現金掛待結算，立即可用現金扣回
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
