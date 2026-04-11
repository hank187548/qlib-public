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
- `search.py` - search pipeline entry logic
- `reports.py` - report export helpers
- `publish.py` - best-run promotion/export helpers
- `paths.py` / `search_results.py` / `ic.py` - support utilities

CLI entrypoints live under `scripts/research/`.
