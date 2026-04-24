# qlib_tw

All importable core logic lives here.

Package layout:
- `research/` - model training, backtests, reporting, and search
- `trade/` - paper trading, broker integration, and Taiwan exchange rules

If you want to change core behavior, start in this folder.
CLI entrypoints live separately under `scripts/research/` and `scripts/trade/`.
