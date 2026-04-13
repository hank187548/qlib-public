# 2026-04-13 Adjusted HLC3 Study

This snapshot preserves the adjusted-price research results that were produced from the local `outputs/` directory and are intended to be reviewed on GitHub without re-running the full pipeline.

## Setup

- Data handler: `alpha158_adj_*`
- Price adjustment: forward-adjusted via `price * factor`
- VWAP proxy: `HLC3 = (high + low + close) / 3`
- Backtest execution: `open`
- Settlement: `T+2`
- Initial capital: `1,000,000`
- Costs:
  - buy fee `0.00092625`
  - sell fee + tax `0.00392625`
  - board-lot min fee `20`
  - odd-lot min fee `1`

## IC Summary

The best adjusted models by test IC in this batch were:

| combo | valid IC | test IC |
| --- | ---: | ---: |
| `alpha158_adj_cat` | `0.044849` | `0.014077` |
| `alpha158_adj_lgb_pro_fil` | `0.037509` | `0.017010` |
| `alpha158_adj_xgb` | `0.034850` | `0.017009` |

Source files:

- `ic/ic_summary.csv`
- `ic/ic_summary_wide.csv`

## Execution Search Summary

All three adjusted models screened here had acceptable IC but poor strategy performance under the current `open + T+2 + adjusted-price` backtest.

| combo | best variant | annualized excess return | information ratio | strategy cumulative return | benchmark cumulative return |
| --- | --- | ---: | ---: | ---: | ---: |
| `alpha158_adj_cat` | `bucket_topk20_ndrop1_risk0p98_day_open_tplus` | `-0.245583` | `-1.387151` | `0.354352` | `0.593156` |
| `alpha158_adj_xgb` | `equal_topk15_ndrop1_risk0p95_day_open_tplus` | `-0.234708` | `-1.222656` | `0.374652` | `0.593156` |
| `alpha158_adj_lgb_pro_fil` | `equal_topk20_ndrop1_risk0p95_day_open_tplus` | `-0.233793` | `-1.318427` | `0.366919` | `0.593156` |

Conclusion:

- The adjusted models in this batch produced usable ranking signal at the IC level.
- That signal did not translate into positive excess return after portfolio construction, costs, and `T+2` execution.
- This snapshot is worth keeping for comparison, but it is not a candidate to replace the current non-adjusted best run.

## Files

- `execution_grid/alpha158_adj_cat/scan_summary.json`
- `execution_grid/alpha158_adj_cat/grid_results_ranked.csv`
- `execution_grid/alpha158_adj_xgb/scan_summary.json`
- `execution_grid/alpha158_adj_xgb/grid_results_ranked.csv`
- `execution_grid/alpha158_adj_lgb_pro_fil/scan_summary.json`
- `execution_grid/alpha158_adj_lgb_pro_fil/grid_results_ranked.csv`

## Configs Used

The tracked configs used to produce these searches are:

- `configs/trade/execution_grid.alpha158_adj_cat_1m_tplus_adjpx.json`
- `configs/trade/execution_grid.alpha158_adj_xgb_1m_tplus_adjpx.json`
- `configs/trade/execution_grid.alpha158_adj_lgb_pro_fil_1m_tplus_adjpx.json`
