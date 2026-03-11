# Qlib TW Workflow (Public)

Public-safe Taiwan equity workflow based on Qlib:
- data pipeline
- model training
- backtesting and analysis
- signal-to-order scripts for broker execution (credentials from environment variables only)

## Best Run Snapshot

Best combo in this repo:
- combo: `alpha158_lgb_pro_fil_ndrop2_topk50`
- strategy cumulative return: `22.7785%`
- benchmark cumulative return: `16.8312%`
- annualized return with cost: `0.037858`
- information ratio with cost: `0.236445`
- max drawdown with cost: `-0.182317`

Source:
- `outputs/best_run/reports/summary.txt`

## Dashboard (Full Bundle)

This repo includes the full dashboard artifacts from the best run in:
- `outputs/best_run/figures/`

Main files:
- `analysis_dashboard.html`
- `model_performance.html`
- `score_ic.html`
- `risk_analysis.html`
- `report_graph.html`
- `equity_curve.png`
- `daily_ic.png`
- `turnover.png`
- `turnover_count.png`

Additional dashboard pages are also included:
- `model_performance_2.html` to `model_performance_6.html`
- `risk_analysis_2.html` to `risk_analysis_5.html`

Preview images:

![Equity Curve](outputs/best_run/figures/equity_curve.png)
![Daily IC](outputs/best_run/figures/daily_ic.png)
![Turnover](outputs/best_run/figures/turnover.png)

How to view interactive dashboard locally:
```bash
python3 -m http.server 8000
# then open:
# http://localhost:8000/outputs/best_run/figures/analysis_dashboard.html
```

## Repository Layout

- `scripts/Get_data_Tai.py`: market data preparation
- `scripts/train_tw.py`: model training entrypoint
- `scripts/backtest_tw.py`: backtest entrypoint
- `scripts/workflow_by_code_tw.py`: end-to-end workflow and report generation
- `scripts/custom_strategy.py`: bucket-weighted strategy
- `scripts/custom_exchange.py`: TW execution/settlement simulation
- `scripts/predict_and_prepare_orders.py`: generate top-k orders from trained model
- `scripts/place_orders_from_csv.py`: submit orders from CSV (default dry-run)
- `scripts/masterlink_trade.py`: manual single-order CLI
- `scripts/test_masterlink_sdk.py`: SDK login smoke test
- `configs/`: workflow config files
- `outputs/best_run/`: exported report and dashboard artifacts

## Order Execution Scripts

The order scripts are included, but no credentials are stored in this repo.

Required environment variables:
- `MASTERLINK_ID`
- `MASTERLINK_PASSWORD`
- `MASTERLINK_CERT`
- `MASTERLINK_CERT_PASSWORD`

Example dry-run:
```bash
python3 scripts/place_orders_from_csv.py outputs/live_orders/orders_alpha158_lgb_YYYY-MM-DD.csv
```

Example live submit:
```bash
python3 scripts/place_orders_from_csv.py outputs/live_orders/orders_alpha158_lgb_YYYY-MM-DD.csv --live
```

Generate order list from model:
```bash
python3 scripts/predict_and_prepare_orders.py --combo alpha158_lgb --topk 50 --strategy bucket
```

## Security Policy

- Never commit `.env`, `.env.*`, or certificate files.
- Never hardcode account/password/token in code.
- Keep broker certificate files under local `secrets/` only.
- Rotate credentials immediately if they were exposed in any previous repository.

## Notes

- This is a public-safe snapshot. Large training artifacts and private runtime data are intentionally excluded.
- Interactive HTML dashboards are stored in `outputs/best_run/figures/` for reproducibility.
