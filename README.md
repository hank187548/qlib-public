# Qlib TW Workflow

This repository contains the Taiwan equity research and trading workflow built on Qlib.

The project is now organized into three layers:

1. `research`
   - data processing
   - model training
   - backtesting
   - model search
   - backtest search
2. `trade`
   - paper trading
   - replay
   - broker / order preparation
3. `outputs`
   - train, backtest, search, and paper-trading results

If you are working on research only, focus on:
- `qlib_tw/research/`
- `scripts/research/`
- `configs/research/`

## Main Pipeline

The current main pipeline is:

`Raw_data -> Process_data -> qlib_data -> Alpha158/Alpha360 -> DatasetH -> model -> strategy -> TWExchange -> reports`

Data semantics:
- `Data/Raw_data/`: raw downloaded data
- `Data/Process_data/`: intermediate processed data for debugging
- `Data/qlib_data/`: Qlib provider

The provider is an **adjusted-price provider**:
- `$open/$high/$low/$close/$vwap` are already adjusted
- `$factor` is treated as a price-only factor and is not used for share rounding

## Repository Layout

- `qlib_tw/research/`
  - reusable research and backtest logic
- `qlib_tw/trade/`
  - paper trading, replay, exchange, and broker logic
- `scripts/research/`
  - research CLI entrypoints
- `scripts/trade/`
  - trade CLI entrypoints
- `configs/research/`
  - `model_search` and `backtest_search` configs
- `configs/trade/`
  - paper-trading configs

## Canonical Combos

The research layer now keeps only these canonical combos:

- `alpha158_lgb`
- `alpha158_xgb`
- `alpha158_cat`
- `alpha360_lgb`
- `alpha360_xgb`
- `alpha360_cat`

If you want the most stable starting point, use:

- `alpha158_lgb`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pyqlib lightgbm xgboost catboost pandas numpy matplotlib plotly
```

## Research Workflow

### 0. Build Alpha158 cache

The `alpha158_*` combos read a precomputed Alpha158 cache from `Data/alpha_158_data`.
Rebuild it after updating `Data/qlib_data` or changing the Alpha158 date range.

```bash
.venv/bin/python scripts/research/build_alpha158_cache.py
```

The cache stores cleaned Alpha158 features and labels:

- `Data/alpha_158_data/alpha158_feature.parquet`
- `Data/alpha_158_data/alpha158_label.parquet`
- `Data/alpha_158_data/metadata.json`

### 1. Train a model

This trains a model only. It does not run a strategy backtest.

```bash
.venv/bin/python scripts/research/workflow_by_code_tw.py train --combo alpha158_lgb
```

Outputs are written to:

- `outputs/models/alpha158_lgb/`

Important files:
- `outputs/models/alpha158_lgb/train_metadata.json`
  - used by `backtest_search` and later by paper trading
- `outputs/models/alpha158_lgb/reports/summary.txt`
- `outputs/models/alpha158_lgb/reports/daily_ic.csv`
- `outputs/models/alpha158_lgb/figures/model_performance.html`

### 2. Run one backtest

If you want to test one specific strategy setup:

```bash
.venv/bin/python scripts/research/workflow_by_code_tw.py backtest \
  --combo alpha158_lgb \
  --strategy bucket \
  --topk 10 \
  --n-drop 1
```

Outputs are written to:

- `outputs/backtest/<run-name>/`

### 3. Run end-to-end

If you want one command that trains and backtests:

```bash
.venv/bin/python scripts/research/workflow_by_code_tw.py full \
  --combo alpha158_lgb \
  --strategy bucket \
  --topk 10 \
  --n-drop 1
```

## Search Workflow

Search is now split into two clearly separated stages.

### A. Model Search

Purpose:
- keep the combo fixed
- search model hyperparameters
- rank by `ic_mean / ic_ir`
- **do not run backtests**

Run:

```bash
.venv/bin/python scripts/research/model_search.py \
  --config configs/research/model_search.example.json
```

Outputs are written to:

- `outputs/model_search/<run-tag>/`

Important files:
- `model_search_results.csv`
- `top_model_candidates.csv`
- `best_model.json`

### B. Backtest Search

Purpose:
- fix one already trained model
- search strategy / backtest parameters
- rank by backtest performance

First train a model:

```bash
.venv/bin/python scripts/research/workflow_by_code_tw.py train --combo alpha158_lgb
```

Then run:

```bash
.venv/bin/python scripts/research/backtest_search.py \
  --config configs/research/backtest_search.example.json
```

The default config reads:

- `outputs/models/alpha158_lgb/train_metadata.json`

So if you want to use another model, change:

- `train_metadata`

inside:

- `configs/research/backtest_search.example.json`

Outputs are written to:

- `outputs/backtest_search/alpha158_lgb_backtest_search/`

Important files:
- `backtest_search_results.csv`
- `best_result.json`
- `strategy_variants.json`
- `resolved_config.json`

## Output Layout

The main output folders are:

- `outputs/models/`
  - train-only diagnostics
  - IC and model-performance outputs
  - `train_metadata.json`
- `outputs/backtest/`
  - one-off backtest outputs
- `outputs/model_search/`
  - model-search outputs
- `outputs/backtest_search/`
  - strategy / backtest-search outputs
- `outputs/paper_trading/`
  - paper-trading and replay outputs
- `outputs/best_run/`
  - tracked public snapshot stored in the repo

## Promote a Backtest Run

If you want to copy one local backtest result into the tracked public snapshot:

```bash
python3 scripts/research/promote_best_run.py --combo <your-backtest-run-name> --clean
```

This copies:
- `reports/`
- `figures/`

into:
- `outputs/best_run/`

## Paper Trading

This was not the main target of the current refactor, but the entrypoints are already organized.

Example config:

- `configs/trade/paper_trading.alpha158_lgb_tplus.example.json`

Run one paper-trading replay manually:

```bash
python3 scripts/trade/paper_trade_daily.py \
  --config configs/trade/paper_trading.alpha158_lgb_tplus.example.json \
  --target-date 2026-04-11
```

Scheduler wrapper:

```bash
bash scripts/trade/run_paper_trade_daily.sh
```

## Quick Guide

### I want to find a better model

Use:

- `scripts/research/model_search.py`

### I already chose a model and want to search strategy settings

Use:

- `scripts/research/backtest_search.py`

### I just want to train one model quickly

Use:

- `scripts/research/workflow_by_code_tw.py train`

### I just want to test one strategy setup quickly

Use:

- `scripts/research/workflow_by_code_tw.py backtest`

## What Is Still Left

The research mainline is basically complete.

The main areas that can still be cleaned up later are:
- paper trading
- replay
- broker integration
- additional trade-layer config and docs cleanup

If you are only doing research, you can treat this repository as:

**research is stable; the trade layer is the remaining cleanup area.**
