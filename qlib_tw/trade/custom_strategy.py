from __future__ import annotations

import copy
from typing import Iterable, List

import pandas as pd

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position


class BucketWeightTopkDropout(TopkDropoutStrategy):
    """
    TopkDropout variant with true target-weight rebalancing.

    This strategy keeps the same replacement logic as TopkDropout:
    - rank by score descending
    - buy the highest-ranked tradable outsiders
    - sell the lowest-ranked sellable incumbents only when they are displaced

    The difference is that bucket weights represent target portfolio weights
    for the final desired holdings, instead of only splitting newly available cash.
    """

    def __init__(self, *, bucket_weights: List[float] | None = None, **kwargs):
        kwargs.setdefault("method_buy", "top")
        kwargs.setdefault("method_sell", "bottom")
        kwargs.setdefault("only_tradable", True)
        if kwargs["method_buy"] != "top":
            raise ValueError("BucketWeightTopkDropout only supports method_buy='top'")
        if kwargs["method_sell"] != "bottom":
            raise ValueError("BucketWeightTopkDropout only supports method_sell='bottom'")
        super().__init__(**kwargs)
        self.bucket_weights = bucket_weights

    def _default_bucket_profile(self, rank_count: int) -> list[float]:
        if rank_count <= 0:
            return []
        if rank_count == 1:
            return [1.0]

        top_count = max(int(round(rank_count * 0.2)), 1)
        mid_count = max(int(round(rank_count * 0.4)), 1)
        if top_count + mid_count > rank_count:
            mid_count = max(rank_count - top_count, 0)
        low_count = max(rank_count - top_count - mid_count, 0)
        return [4.0] * top_count + [2.0] * mid_count + [1.0] * low_count

    def _get_target_weights(self, rank_count: int) -> list[float]:
        if rank_count <= 0:
            return []

        base_weights = list(self.bucket_weights) if self.bucket_weights else self._default_bucket_profile(self.topk)
        if len(base_weights) < rank_count:
            pad_weight = base_weights[-1] if base_weights else 1.0
            base_weights.extend([pad_weight] * (rank_count - len(base_weights)))
        weights = base_weights[:rank_count]
        total = sum(weights)
        if total <= 0:
            return [1.0 / rank_count] * rank_count
        return [w / total for w in weights]

    def _is_direction_tradable(
        self,
        stock_id: str,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
        direction: OrderDir,
    ) -> bool:
        return self.trade_exchange.is_stock_tradable(
            stock_id=stock_id,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=None if self.forbid_all_trade_at_limit else direction,
        )

    def _available_amount(self, position: Position, stock_id: str, current_dt: pd.Timestamp) -> float:
        available_fn = getattr(self.trade_exchange, "_available_amount", None)
        if callable(available_fn):
            return float(available_fn(position, stock_id, current_dt))
        return float(position.get_stock_amount(stock_id))

    @staticmethod
    def _rank_subset(pred_score: pd.Series, codes: Iterable[str]) -> pd.Index:
        code_index = pd.Index(list(codes))
        if code_index.empty:
            return code_index
        return pred_score.reindex(code_index).sort_values(ascending=False).index

    def _build_desired_holdings(
        self,
        pred_score: pd.Series,
        current_stock_list: list[str],
        sellable_ranked: pd.Index,
        buyable_outsiders_ranked: pd.Index,
    ) -> tuple[pd.Index, pd.Index]:
        frozen_holdings = pd.Index([code for code in current_stock_list if code not in sellable_ranked])
        open_slots = max(self.topk - len(current_stock_list), 0)
        outsider_pool = buyable_outsiders_ranked[: self.n_drop + open_slots]

        flexible_capacity = max(self.topk - len(frozen_holdings), 0)
        compare_pool = sellable_ranked.union(outsider_pool)
        desired_flexible = self._rank_subset(pred_score, compare_pool)[:flexible_capacity]
        sells = sellable_ranked[~sellable_ranked.isin(desired_flexible)]

        desired_names = pd.Index(list(frozen_holdings) + [code for code in desired_flexible if code not in frozen_holdings])
        desired_rank = self._rank_subset(pred_score, desired_names)
        return desired_rank, sells

    def _get_rebalance_price(
        self,
        stock_id: str,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> float | None:
        price = self.trade_exchange.get_deal_price(
            stock_id=stock_id,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=OrderDir.BUY,
        )
        if price is None or pd.isna(price) or price <= 0:
            return None
        return float(price)

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        current_temp: Position = copy.deepcopy(self.trade_position)
        current_dt = trade_start_time.normalize()
        release_pending = getattr(self.trade_exchange, "_release_pending", None)
        if callable(release_pending):
            release_pending(current_temp, current_dt)

        current_stock_list = current_temp.get_stock_list()
        ranked_all = pred_score.sort_values(ascending=False).index

        sellable_codes = []
        for code in current_stock_list:
            time_per_step = self.trade_calendar.get_freq()
            if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                continue
            if self.only_tradable and not self._is_direction_tradable(code, trade_start_time, trade_end_time, OrderDir.SELL):
                continue
            if self._available_amount(current_temp, code, current_dt) <= 0:
                continue
            sellable_codes.append(code)

        if self.only_tradable:
            buyable_outsiders = [
                code
                for code in ranked_all
                if code not in current_stock_list
                and self._is_direction_tradable(code, trade_start_time, trade_end_time, OrderDir.BUY)
            ]
        else:
            buyable_outsiders = [code for code in ranked_all if code not in current_stock_list]

        sellable_ranked = self._rank_subset(pred_score, sellable_codes)
        buyable_outsiders_ranked = pd.Index(buyable_outsiders)
        desired_rank, _ = self._build_desired_holdings(
            pred_score,
            current_stock_list,
            sellable_ranked,
            buyable_outsiders_ranked,
        )

        target_weights = self._get_target_weights(len(desired_rank))
        target_weight_map = {code: target_weights[idx] for idx, code in enumerate(desired_rank)}
        target_portfolio_value = float(current_temp.calculate_value()) * self.risk_degree

        sellable_set = set(sellable_codes)
        buyable_set = set(buyable_outsiders)
        sell_orders: list[Order] = []
        buy_orders: list[Order] = []

        sell_sequence = self._rank_subset(pred_score, current_stock_list)[::-1]
        for code in sell_sequence:
            current_amount = float(current_temp.get_stock_amount(code))
            if current_amount <= 0 or code not in sellable_set:
                continue
            if code not in target_weight_map:
                target_amount = 0.0
            else:
                mark_price = self._get_rebalance_price(code, trade_start_time, trade_end_time)
                if mark_price is None:
                    continue
                target_amount = (target_portfolio_value * target_weight_map[code]) / mark_price

            sell_amount = max(current_amount - target_amount, 0.0)
            if sell_amount <= 1e-8:
                continue

            sell_order = Order(
                stock_id=code,
                amount=sell_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.SELL,
            )
            if not self.trade_exchange.check_order(sell_order):
                continue
            trade_val, _, _ = self.trade_exchange.deal_order(sell_order, position=current_temp)
            if trade_val > 1e-8:
                sell_orders.append(sell_order)

        for code in desired_rank:
            current_amount = float(current_temp.get_stock_amount(code))
            if code not in current_stock_list and code not in buyable_set:
                continue
            if code in current_stock_list and self.only_tradable and not self._is_direction_tradable(
                code, trade_start_time, trade_end_time, OrderDir.BUY
            ):
                continue
            mark_price = self._get_rebalance_price(code, trade_start_time, trade_end_time)
            if mark_price is None:
                continue
            target_amount = (target_portfolio_value * target_weight_map[code]) / mark_price
            buy_amount = max(target_amount - current_amount, 0.0)
            if buy_amount <= 1e-8:
                continue

            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            if not self.trade_exchange.check_order(buy_order):
                continue
            trade_val, _, _ = self.trade_exchange.deal_order(buy_order, position=current_temp)
            if trade_val > 1e-8:
                buy_orders.append(buy_order)

        return TradeDecisionWO(sell_orders + buy_orders, self)
