# Paper Trading

This folder is an isolated forward paper-trading layer for Taiwan strategies.
It keeps outputs separate from the research backtest entrypoints.

## Design Goals

- Reuse the existing model, strategy, and T+2 exchange logic.
- Reuse the shared main provider at `Data/tw_data` so daily updates stay simple.
- Keep daily simulation outputs outside the current backtest folders.
- Avoid state drift by using full replay to the latest available trade date.
- Make risks explicit when the backtest model cannot be mapped 1:1 to live orders.

## Source Of Truth

Paper trading starts strictly from the configured `backtest_start`.

- Each daily run rebuilds the portfolio path from `backtest_start` to the latest available trade date.
- No position, cash, or pending settlement state is inherited from any earlier repo test segment.
- `paper_state.json` is a snapshot output, not the engine state that drives the next run.
- This avoids custom state-resume bugs, but it does mean historical data revisions can change past simulated decisions.

If `backtest_start` is set later than the latest available trade date, the runner switches to a fresh-start preview mode.

- `orders_next_day.csv` is generated from the latest signal using cash-only state.
- `paper_state.json` shows the pre-start cash snapshot with no inherited positions.
- Once market data for the start date exists, normal replay resumes from that start date.

## Output Contract

Daily files are written under:

- `outputs/paper_trading/<profile>/daily/<YYYY-MM-DD>/`
- `outputs/paper_trading/<profile>/latest/`

Files:

- `paper_state.json`
  - End-of-day account snapshot.
  - Includes `cash`, `cash_delay`, positions, `pending_cash`, and `pending_stock`.
- `fills_YYYY-MM-DD.csv`
  - Derived from Qlib `indicators_normal_1day_obj.pkl`.
  - Contains requested quantity, filled quantity, side, fill rate, fill price, and cost.
- `order_fill_comparison.csv`
  - Compares the prior signal day's planned order with the current trade day's actual requested and filled quantity.
  - Useful for checking how much buy sizing changed after the next-day `open` repricing.
  - Includes `comparison_status` such as `matched_and_filled`, `repriced_size_changed`, and `planned_but_not_requested`.
- `nav_history.csv`
  - Full daily account history up to the replay end date.
  - Includes account value, market value, cash, cash delay, turnover, cost, and excess return series.
- `orders_next_day.csv`
  - Intended next-day order list derived from the latest signal and current replay state.
  - Sell rows are exact from current settled holdings.
  - Buy rows are estimates based on the latest close because the current T+2 backtest model sizes buys with the next open, which is unknown before the next session.
  - `blocked_by_tplus=true` means the strategy wants to exit but the shares are still settlement-locked and cannot be sold yet.

## Shared Data Policy

- Paper trading now refreshes and reuses the shared provider at `Data/tw_data`.
- This keeps the daily workflow simple and aligned with the rest of the repo.
- Outputs remain isolated under `outputs/paper_trading/<profile>/`.
- A tiny calendar overlay may still be created under the paper output folder when the local provider ends on the latest available trade date.
  - This is only a runtime helper so Qlib can see the next session boundary.
  - It does not duplicate the full provider and does not modify `Data/tw_data`.

## Key Risks

- The current T+2 model is not a broker-native order model.
  - It now executes on the next-day `open` with T+2 settlement.
  - This is closer to the intended backtest assumption, but it is still not a broker-native pre-open order model.
- `orders_next_day.csv` buy quantities are estimates.
  - Exact next-day buy sizing requires the next open, which is unavailable at signal generation time.
- `next_trade_date` uses business-day fallback when future exchange sessions are not present in the local provider calendar.
  - Weekends are handled correctly.
  - Taiwan market holidays still need manual awareness unless a future-aware calendar source is added.
- The current Yahoo -> Qlib dump is full-overwrite.
  - Refreshing paper trading also refreshes the shared provider used by the repo.
  - This is simpler operationally, but it means future research runs will see the updated provider.
- Replay depends on stable historical data.
  - If the vendor revises older bars, previously simulated states can move.

## Profile Config

Use a separate JSON config, for example:

- [configs/trade/paper_trading.alpha158_lgb_run11_tplus.example.json](/home/nas2/Personal/Hank/qlib-public/configs/trade/paper_trading.alpha158_lgb_run11_tplus.example.json)

## Entry Point

Run the daily replay with:

```bash
./.venv/bin/python scripts/trade/paper_trade_daily.py \
  --config configs/trade/paper_trading.alpha158_lgb_run11_tplus.example.json \
  --target-date 2026-04-09
```

Refresh is on by default. Pass `--no-refresh-data` if you want to replay without updating `Data/tw_data` first.

For unattended runs, use the shell wrapper:

```bash
bash scripts/trade/run_paper_trade_daily.sh
```

It uses:

- `configs/trade/paper_trading.alpha158_lgb_run11_tplus.example.json` by default
- `.venv/bin/python` by default
- `outputs/paper_trading/_scheduler_logs/` for daily logs
- `flock` when available to prevent overlapping runs

You can override behavior with env vars:

- `PAPER_TRADE_CONFIG`
- `PAPER_TRADE_PYTHON`
- `PAPER_TRADE_TARGET_DATE`
- `PAPER_TRADE_REFRESH_DATA=0` to skip data refresh

A ready-to-paste cron sample is included at:

- [configs/trade/paper_trade_daily.crontab.example](/home/nas2/Personal/Hank/qlib-public/configs/trade/paper_trade_daily.crontab.example)

Recommended schedule:

- One run before the session starts to refresh the preview `orders_next_day.csv`.
- One run after the close to refresh `fills`, `paper_state`, `nav_history`, and `order_fill_comparison.csv`.
