from typing import List
import copy
import numpy as np
import pandas as pd

from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import OrderDir, TradeDecisionWO, Order
from qlib.backtest.position import Position


class BucketWeightTopkDropout(TopkDropoutStrategy):
    """
    TopkDropout variant: apply bucketed weights by rank instead of equal weighting.
    Default: top 10 at 4%%, rank 11-30 at 2%%, rank 31-50 at 1%% (sum 100%%).
    """

    def __init__(self, *, bucket_weights: List[float] | None = None, **kwargs):
        super().__init__(**kwargs)
        # bucket_weights length should equal topk; per-rank target weights should sum to 1
        self.bucket_weights = bucket_weights

    def _get_bucket_weights(self):
        if self.bucket_weights:
            return self.bucket_weights
        # default topk=50: 10*0.04 + 20*0.02 + 20*0.01 = 1.0
        weights = []
        weights += [0.04] * 10
        weights += [0.02] * 20
        weights += [0.01] * 20
        return weights

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        def get_first_n(li, n):
            return list(li)[:n]

        def get_last_n(li, n):
            return list(li)[-n:]

        def filter_stock(li):
            return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        current_dt = trade_start_time.normalize()
        release_pending = getattr(self.trade_exchange, "_release_pending", None)
        if callable(release_pending):
            release_pending(current_temp, current_dt)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()

        # Sort previous positions by score
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index
        # Today candidates (new picks by TopkDropout)
        today = get_first_n(
            pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
            self.n_drop + self.topk - len(last),
        )
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
        sell = last[last.isin(get_last_n(comb, self.n_drop))]
        # Sell first
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    self.trade_exchange.deal_order(sell_order, position=current_temp)

        # `deal_order` already updates the temporary position.
        # Re-read cash from the simulated position instead of manually adding sale proceeds.
        # This is critical for T+N settlement because pending cash must not become available
        # for immediate reuse on the same rebalance step.
        cash = current_temp.get_cash()
        held_after_sell = pd.Index(current_temp.get_stock_list())
        open_slots = max(self.topk - len(held_after_sell), 0)
        buy = get_first_n(
            pred_score[~pred_score.index.isin(held_after_sell)].sort_values(ascending=False).index,
            open_slots,
        )

        # Bucketed weights
        weights = self._get_bucket_weights()
        if len(weights) < self.topk:
            # If shorter than topk, pad with equal weights
            tail = self.topk - len(weights)
            weights += [1.0 / self.topk] * tail
        weights = weights[: self.topk]
        rank_index = pred_score.sort_values(ascending=False).index
        weight_map = {code: weights[i] for i, code in enumerate(rank_index[: self.topk])}

        # Allocate cash by relative weights only within buy list
        buy_weights = [weight_map.get(code, 0.0) for code in buy]
        total_w = sum(buy_weights)
        for code, w in zip(buy, buy_weights):
            if w <= 0 or total_w <= 0:
                continue
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            target_value = cash * self.risk_degree * (w / total_w)
            buy_amount = target_value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount <= 0:
                continue
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)
