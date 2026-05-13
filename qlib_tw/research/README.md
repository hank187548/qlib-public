# Research

This package holds reusable research and backtest logic inside `qlib_tw/`.

Use this area for:
- Qlib provider settings and universe definitions
- training and backtest builders
- report generation
- experiment search and promotion helpers

Module map:
- `settings.py` - shared provider, segments, costs, and combo definitions
- `builders.py` - task, strategy, executor, and exchange config builders
- `runner.py` - Qlib init, training, and backtest orchestration helpers
- `search.py` - model-search pipeline entry logic
- `backtest_search.py` - backtest-parameter search for a fixed trained model
- `rolling.py` - fixed-window rolling walk-forward split and backtest pipeline
- `reports.py` - report export helpers
- `publish.py` - best-run promotion/export helpers
- `paths.py` / `search_results.py` / `ic.py` - support utilities

CLI entrypoints live under `scripts/research/`.

Rolling walk-forward entrypoint:

```bash
python3 scripts/research/rolling_walk_forward.py \
  --config configs/research/rolling_walk_forward.example.json \
  --dry-run
```

Remove `--dry-run` to train one model per round, merge trade-quarter OOS
predictions, and run one continuous backtest. The splitter uses provider trading
dates, fixed 3Y/2Q/1Q windows, and a 2 trading-day embargo for T+1 to T+2 labels.

Set `strategy_mode` in the rolling config:
- `fixed` keeps one strategy parameter set for the whole rolling run.
- `validation_search` searches `topk`, `n_drop`, and `risk_degree` on each round's validation window, then uses the selected parameters for the next trade quarter in the same continuous OOS backtest.

Use `configs/research/rolling_walk_forward.validation_search.example.json` for a ready-to-run validation-search config.

Combo naming:

- `alpha158_*` / `alpha360_*` use the official Qlib handlers directly
- provider price semantics come from `Data/qlib_data/price_semantics.json`
- execution and paper-trading prices are still controlled by the exchange config, not by the handler
