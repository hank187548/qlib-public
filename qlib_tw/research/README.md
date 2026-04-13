# Research

This package holds reusable research and backtest logic inside `qlib_tw/`.

Use this area for:
- Qlib provider settings and universe definitions
- training and backtest builders
- report generation
- experiment search and promotion helpers
- adjusted-price research handlers that keep execution prices untouched

Module map:
- `settings.py` - shared provider, segments, costs, and combo definitions
- `adjusted_handlers.py` - forward-adjusted Alpha158/Alpha360 handlers for training and inference
- `builders.py` - task, strategy, executor, and exchange config builders
- `runner.py` - Qlib init, training, and backtest orchestration helpers
- `search.py` - search pipeline entry logic
- `reports.py` - report export helpers
- `publish.py` - best-run promotion/export helpers
- `paths.py` / `search_results.py` / `ic.py` - support utilities

CLI entrypoints live under `scripts/research/`.

Combo naming:

- `alpha158_*` / `alpha360_*` keep the original raw-price research handlers
- `alpha158_adj_*` / `alpha360_adj_*` use forward-adjusted prices for features and labels
- execution and paper-trading prices are still controlled by the exchange config, not by these handlers
