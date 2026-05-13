from __future__ import annotations

import unittest

import pandas as pd

from qlib_tw.research.rolling import (
    RollingSplit,
    RollingWalkForwardConfig,
    _build_strategy_variants,
    _slice_report_metrics,
    _sort_strategy_search_rows,
    _strategy_search_config,
    generate_rolling_splits,
)


class RollingSplitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.calendar = list(pd.bdate_range("2016-01-01", "2020-12-31"))

    def test_quarter_alignment_and_embargo(self) -> None:
        plan = generate_rolling_splits(
            self.calendar,
            start_date="2016-01-01",
            end_date="2020-12-31",
            train_years=3,
            valid_quarters=2,
            trade_quarters=1,
            step_quarters=1,
            embargo_days=2,
        )
        first = plan.splits[0]
        self.assertEqual(first.train_start, pd.Timestamp("2016-01-01"))
        self.assertEqual(first.valid_start, pd.Timestamp("2019-01-01"))
        self.assertEqual(first.trade_start, pd.Timestamp("2019-07-01"))
        self.assertEqual(first.trade_end, pd.Timestamp("2019-09-30"))
        self.assertEqual(first.train_end, pd.Timestamp("2018-12-27"))
        self.assertEqual(first.valid_end, pd.Timestamp("2019-06-26"))

        index = {dt: idx for idx, dt in enumerate(self.calendar)}
        self.assertEqual(index[first.valid_start] - index[first.train_end] - 1, 2)
        self.assertEqual(index[first.trade_start] - index[first.valid_end] - 1, 2)

    def test_step_moves_one_quarter(self) -> None:
        plan = generate_rolling_splits(
            self.calendar,
            start_date="2016-01-01",
            end_date="2020-12-31",
        )
        first, second = plan.splits[:2]
        self.assertEqual(second.train_start, pd.Timestamp("2016-04-01"))
        self.assertEqual(second.valid_start, pd.Timestamp("2019-04-01"))
        self.assertEqual(second.trade_start, pd.Timestamp("2019-10-01"))
        self.assertLess(first.trade_end, second.trade_start)

    def test_trade_never_overlaps_train_or_valid(self) -> None:
        plan = generate_rolling_splits(
            self.calendar,
            start_date="2016-01-01",
            end_date="2020-12-31",
        )
        for split in plan.splits:
            self.assertLess(split.train_end, split.valid_start)
            self.assertLess(split.valid_end, split.trade_start)
            self.assertGreater(split.trade_start, split.valid_end)
            self.assertGreater(split.trade_start, split.train_end)

    def test_incomplete_requested_tail_is_skipped(self) -> None:
        plan = generate_rolling_splits(
            self.calendar,
            start_date="2016-01-01",
            end_date="2019-08-01",
        )
        self.assertEqual(plan.splits, [])
        self.assertEqual(len(plan.skipped_rounds), 1)
        self.assertIn("requested end_date", plan.skipped_rounds[0]["reason"])

    def test_incomplete_provider_calendar_is_skipped(self) -> None:
        short_calendar = list(pd.bdate_range("2016-01-01", "2019-08-30"))
        plan = generate_rolling_splits(
            short_calendar,
            start_date="2016-01-01",
            end_date="2019-12-31",
        )
        self.assertEqual(plan.splits, [])
        self.assertEqual(len(plan.skipped_rounds), 1)
        self.assertIn("provider calendar", plan.skipped_rounds[0]["reason"])

    def test_slice_report_metrics_includes_compounded_net_return(self) -> None:
        split = RollingSplit(
            round_id=1,
            train_start=pd.Timestamp("2016-01-01"),
            train_end=pd.Timestamp("2018-12-27"),
            valid_start=pd.Timestamp("2019-01-01"),
            valid_end=pd.Timestamp("2019-06-26"),
            trade_start=pd.Timestamp("2019-07-01"),
            trade_end=pd.Timestamp("2019-07-02"),
        )
        report = pd.DataFrame(
            {
                "return": [0.10, -0.05],
                "cost": [0.01, 0.02],
                "bench": [0.03, 0.02],
                "total_turnover": [100.0, 200.0],
            },
            index=pd.to_datetime(["2019-07-01", "2019-07-02"]),
        )
        metrics = _slice_report_metrics(report, split)
        expected_net = (1 + 0.10 - 0.01) * (1 - 0.05 - 0.02) - 1
        self.assertAlmostEqual(metrics["net_cumulative_return"], expected_net)

    def test_strategy_variants_skip_invalid_n_drop(self) -> None:
        config = RollingWalkForwardConfig(
            name="test",
            combo="alpha158_lgb",
            start_date="2016-01-01",
            end_date="2020-12-31",
            strategy_search={
                "topk_values": [1, 2],
                "n_drop_values": [1, 3],
                "risk_degree_values": [0.9],
            },
        )
        variants = _build_strategy_variants(config)
        self.assertEqual([(row["topk"], row["n_drop"]) for row in variants], [(1, 1), (2, 1)])

    def test_strategy_search_sort_uses_tie_break(self) -> None:
        config = RollingWalkForwardConfig(
            name="test",
            combo="alpha158_lgb",
            start_date="2016-01-01",
            end_date="2020-12-31",
            strategy_search={
                "ranking_metric": "score",
                "ranking_order": "desc",
                "tie_break": [
                    {"metric": "avg_turnover", "order": "asc"},
                    {"metric": "topk", "order": "desc"},
                ],
            },
        )
        rows = [
            {"score": 1.0, "avg_turnover": 0.5, "topk": 50},
            {"score": 1.0, "avg_turnover": 0.3, "topk": 20},
            {"score": 0.9, "avg_turnover": 0.1, "topk": 100},
        ]
        sorted_rows = _sort_strategy_search_rows(rows, _strategy_search_config(config))
        self.assertEqual(sorted_rows[0]["topk"], 20)


if __name__ == "__main__":
    unittest.main()
