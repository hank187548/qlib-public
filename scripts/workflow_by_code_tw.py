#!/usr/bin/env python3
"""End-to-end Taiwan Qlib workflow without notebooks.

This script mirrors the previous Jupyter notebook:
1. 初始化台灣市場的 Qlib provider。
2. 訓練 LightGBM 模型。
3. 產生訊號並進行回測分析。
4. 輸出報表 CSV 與圖形到指定資料夾。
"""

import argparse
import logging
import os
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # 使用無頭模式繪圖
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
from plotly.subplots import make_subplots

import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.contrib.report import analysis_model, analysis_position

# ---------------------------------------------------------------------------
# 基本設定
# ---------------------------------------------------------------------------

WORK_DIR = Path(__file__).resolve().parent.parent
# 確保自訂模組 (scripts.custom_strategy) 可匯入
if str(WORK_DIR) not in sys.path:
    sys.path.append(str(WORK_DIR))
PROVIDER_URI = WORK_DIR.joinpath("Data", "tw_data")
REGION = "tw"
BENCHMARK = "^TWII"
DEFAULT_COMBO = "alpha158_lgb"
UNIVERSE: List[str] = []

BASE_DATA_HANDLER_CONFIG: Dict[str, object] = {
    "start_time": "2009-01-01",
    "end_time": "2025-11-26",
    "fit_start_time": "2009-01-01",
    "fit_end_time": "2025-11-26",
}

# 近期訓練窗口（可視最新資料定期滑動）
SEGMENTS: Dict[str, tuple[str, str]] = {
    "train": ("2020-01-01", "2024-06-30"),   # 約近 4.5 年
    "valid": ("2024-07-01", "2024-12-31"),   # 近半年做超參數/停利驗證
    # test 結束日設為倒數第二個交易日，避免 calendar 邊界溢位
    "test": ("2025-01-01", "2025-11-25"),
}

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "lgb": {
        "class": "LGBModel",
        "module_path": "qlib.contrib.model.gbdt",
        "kwargs": {
            "loss": "mse",
            "colsample_bytree": 0.75,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "lambda_l1": 50.0,
            "lambda_l2": 200.0,
            "max_depth": 6,
            "num_leaves": 96,
            "min_child_samples": 50,
            "num_threads": os.cpu_count() or 8,
        },
    },
    "xgb": {
        "class": "XGBModel",
        "module_path": "qlib.contrib.model.xgboost",
        "kwargs": {
            "eval_metric": "rmse",
            "colsample_bytree": 0.8879,
            "eta": 0.0421,
            "max_depth": 8,
            "n_estimators": 647,
            "subsample": 0.8789,
            "nthread": os.cpu_count() or 8,
        },
    },
    "cat": {
        "class": "CatBoostModel",
        "module_path": "qlib.contrib.model.catboost_model",
        "kwargs": {
            "loss_function": "RMSE",
            "iterations": 800,  # 降低迭代避免爆記憶體
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 3,
            "subsample": 0.9,
            "bootstrap_type": "Bernoulli",
            "random_strength": 0.8,
            "leaf_estimation_iterations": 5,
            # 強制走 CPU，限制執行緒，避免 GPU/記憶體 OOM
            "task_type": "CPU",
            "thread_count": min(4, os.cpu_count() or 4),
        },
    },
}

HANDLER_CONFIGS: Dict[str, Dict[str, str]] = {
    "alpha158": {"class": "Alpha158", "module_path": "qlib.contrib.data.handler"},
    "alpha360": {"class": "Alpha360", "module_path": "qlib.contrib.data.handler"},
    "alpha191": {"class": "Alpha191", "module_path": "qlib.contrib.data.handler"},
}

ALPHA158_INFER_PIPELINE = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]

COMBO_CONFIGS = {
    "alpha158_lgb": {"handler": "alpha158", "model": "lgb", "max_instruments": None, "infer_processors": []},
    "alpha158_lgb_pro_fil": {
        "handler": "alpha158",
        "model": "lgb",
        "max_instruments": None,
        "infer_processors": ALPHA158_INFER_PIPELINE,
    },
    "alpha360_lgb": {"handler": "alpha360", "model": "lgb", "max_instruments": None, "infer_processors": []},
    "alpha191_lgb": {"handler": "alpha191", "model": "lgb", "max_instruments": None, "infer_processors": []},
    "alpha158_xgb": {
        "handler": "alpha158",
        "model": "xgb",
        "max_instruments": None,
        "infer_processors": ALPHA158_INFER_PIPELINE,
    },
    "alpha360_xgb": {"handler": "alpha360", "model": "xgb", "max_instruments": None, "infer_processors": []},
    "alpha191_xgb": {"handler": "alpha191", "model": "xgb", "max_instruments": None, "infer_processors": []},
    "alpha158_cat": {"handler": "alpha158", "model": "cat", "max_instruments": None, "infer_processors": []},
    "alpha360_cat": {"handler": "alpha360", "model": "cat", "max_instruments": None, "infer_processors": []},
}


BASE_PORT_ANALYSIS_CONFIG: Dict[str, object] = {
    "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
            "time_per_step": "day",
            "generate_portfolio_metrics": True,
        },
    },
    "strategy": {
        "class": "BucketWeightTopkDropout",
        "module_path": "scripts.custom_strategy",
        "kwargs": {
            "model": None,  # 將於 runtime 指定
            "dataset": None,
            "topk": 50,
            "n_drop": 5,
            # 僅在對應方向檢查漲跌停：漲停只擋買、跌停只擋賣
            "forbid_all_trade_at_limit": False,
        },
    },
    "backtest": {
        "start_time": SEGMENTS["test"][0],
        "end_time": SEGMENTS["test"][1],
        "account": 100_000,
        "benchmark": BENCHMARK,
        "exchange_kwargs": {
            "freq": "day",
            # 漲跌幅依前一日收盤價的 ±9.5% 判斷：漲停不可買但可賣，跌停不可賣但可買
            "limit_threshold": (
                "$change >= 0.095 * Ref($close, 1)",   # 漲幅達 ~9.5% 視為漲停 → limit_buy
                "$change <= -0.095 * Ref($close, 1)",  # 跌幅達 ~9.5% 視為跌停 → limit_sell
            ),
            "deal_price": "close",
            # 元富電子交易 65 折：
            # - 手續費 0.1425% * 0.65 ≈ 0.00092625（買、賣皆同）
            # - 當沖/現股賣出證交稅 0.15% → 0.0015，併入 close_cost
            # - 零股最低手續費 1 元（統一使用 min_cost=1 近似）
            "open_cost": 0.00092625,   # 買進手續費（含折扣）
            "close_cost": 0.00242625,  # 賣出手續費 + 0.15% 證交稅
            "min_cost": 1,
            "trade_unit": 1,
        },
    },
}


def load_full_universe(provider_uri: Path) -> list[str]:
    """Load all available instruments (including benchmark)."""
    inst_path = provider_uri.joinpath('instruments', 'all.txt')
    symbols = set()
    if inst_path.exists():
        for line in inst_path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            code = parts[0].upper()
            symbols.add(code)
    features_dir = provider_uri.joinpath('features')
    if features_dir.exists():
        for entry in features_dir.iterdir():
            if entry.is_dir():
                symbols.add(entry.name.upper())
    benchmark_symbol = BENCHMARK.upper()
    if benchmark_symbol:
        symbols.add(benchmark_symbol)
    if not symbols:
        raise RuntimeError(f'No instruments found under {provider_uri}')
    return sorted(symbols)

ALL_CODES = load_full_universe(PROVIDER_URI)
UNIVERSE = [code for code in ALL_CODES if not code.startswith('^')]
if not UNIVERSE:
    raise RuntimeError(f'No equity instruments found under {PROVIDER_URI}')

def set_output_dirs(combo_name: str) -> None:
    global OUTPUT_ROOT, REPORT_DIR, FIG_DIR
    root = WORK_DIR.joinpath("outputs", "tw_workflow", combo_name)
    OUTPUT_ROOT = root
    REPORT_DIR = OUTPUT_ROOT.joinpath("reports")
    FIG_DIR = OUTPUT_ROOT.joinpath("figures")
    for directory in (OUTPUT_ROOT, REPORT_DIR, FIG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


set_output_dirs(DEFAULT_COMBO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Taiwan Qlib workflow")
    combo_choices = sorted(list(COMBO_CONFIGS.keys()) + ["all"])
    parser.add_argument(
        "--combo",
        choices=combo_choices,
        nargs="+",
        help=(
            "Specify one or more handler/model combos to run "
            "(default alpha158_lgb). Use 'all' to run every combo."
        ),
    )
    parser.add_argument(
        "--n-drop",
        type=int,
        default=None,
        help="Override TopkDropout n_drop (if provided, outputs will be under <combo>_ndrop<N>)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Override TopkDropout topk (if provided, outputs will be under <combo>_topk<K>)",
    )
    parser.add_argument(
        "--rebalance",
        choices=["day", "week"],
        default="day",
        help="Rebalance frequency for backtest/exchange (default day).",
    )
    parser.add_argument(
        "--strategy",
        choices=["bucket", "equal"],
        default="bucket",
        help="bucket: 分桶配重 (預設 4/2/1)，equal: 傳統 TopkDropout 等權。",
    )
    parser.add_argument(
        "--deal-price",
        choices=["close", "open"],
        default="close",
        help="成交價假設（回測用），預設 close，可改用 open 更接近隔日開盤成交假設。",
    )
    parser.add_argument(
        "--simulate-limit",
        action="store_true",
        help="使用簡化限價模擬（觸價才成交），搭配 --limit-slippage。",
    )
    parser.add_argument(
        "--limit-slippage",
        type=float,
        default=0.01,
        help="限價偏移（相對 base_price），預設 1%%。僅在 --simulate-limit 時生效。",
    )
    return parser.parse_args()


def build_task_config(
    handler_key: str,
    model_key: str,
    instruments: List[str],
    max_instruments: int | None = None,
    infer_processors: List[Dict[str, object]] | None = None,
) -> Dict[str, object]:
    handler_spec = HANDLER_CONFIGS[handler_key]
    model_spec = deepcopy(MODEL_CONFIGS[model_key])
    handler_kwargs = deepcopy(BASE_DATA_HANDLER_CONFIG)
    selected_instruments = instruments if max_instruments is None else instruments[:max_instruments]
    handler_kwargs["instruments"] = selected_instruments
    if infer_processors is not None:
        handler_kwargs["infer_processors"] = deepcopy(infer_processors)
    dataset_cfg = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": handler_spec["class"],
                "module_path": handler_spec["module_path"],
                "kwargs": handler_kwargs,
            },
            "segments": deepcopy(SEGMENTS),
        },
    }
    return {"model": model_spec, "dataset": dataset_cfg}


def build_port_analysis_config() -> Dict[str, object]:
    return deepcopy(BASE_PORT_ANALYSIS_CONFIG)


def resolve_combos(requested: List[str] | None) -> List[str]:
    if not requested:
        return [DEFAULT_COMBO]
    if "all" in requested:
        return list(COMBO_CONFIGS.keys())
    return requested

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
LOGGER = logging.getLogger("tw_workflow")



def _to_plotly_figures(obj) -> list:
    figures = []
    if obj is None:
        return figures
    if hasattr(obj, 'to_plotly_json'):
        try:
            figures.append(obj)
        except Exception as err:
            LOGGER.error('匯入 plotly figure 失敗: %s', err)
        return figures
    if isinstance(obj, dict):
        if 'application/vnd.plotly.v1+json' in obj:
            bundle = obj['application/vnd.plotly.v1+json']
            json_payload = bundle if isinstance(bundle, str) else json.dumps(bundle)
            try:
                figures.append(pio.from_json(json_payload))
            except Exception as err:
                LOGGER.error('解析 plotly bundle 失敗: %s', err)
            return figures
        if 'data' in obj and 'layout' in obj:
            try:
                json_payload = json.dumps({'data': obj['data'], 'layout': obj['layout']})
                figures.append(pio.from_json(json_payload))
            except Exception as err:
                LOGGER.error('解析 plotly dict 失敗: %s', err)
            return figures
    if isinstance(obj, (list, tuple)):
        for item in obj:
            figures.extend(_to_plotly_figures(item))
        return figures
    LOGGER.warning('無法解析 plotly 對象：%s', type(obj))
    return figures


def export_plotly_section(name: str, obj) -> list:
    figs = _to_plotly_figures(obj)
    saved = []
    if not figs:
        return saved
    for idx, figure in enumerate(figs, start=1):
        suffix = '' if idx == 1 else '_{}'.format(idx)
        file_path = FIG_DIR.joinpath('{}{}.html'.format(name, suffix))
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.write_html(str(file_path), include_plotlyjs='cdn')
            LOGGER.info('Plotly 圖表已儲存：%s', file_path)
            saved.append((file_path, figure))
        except Exception as err:
            LOGGER.error('儲存 plotly 圖表失敗 %s: %s', file_path, err)
    return saved

def export_plotly_dashboard(sections: list) -> None:
    if not sections:
        return
    dashboard_path = FIG_DIR.joinpath("analysis_dashboard.html")
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="zh-Hant">',
        "<head>",
        '<meta charset="utf-8" />',
        "<title>Analysis Dashboard</title>",
        "</head>",
        '<body style="max-width:1200px;margin:0 auto;font-family:sans-serif;">',
    ]
    include_js = "cdn"
    for name, entries in sections:
        for idx, (_, fig) in enumerate(entries, start=1):
            title = name if len(entries) == 1 else "{} ({})".format(name, idx)
            html_parts.append('<h2 style="margin-top:40px;">{}</h2>'.format(title))
            html_parts.append(
                pio.to_html(
                    fig,
                    include_plotlyjs=include_js,
                    full_html=False,
                    default_width="100%",
                    default_height="500px",
                )
            )
            include_js = False
    html_parts.append("</body></html>")
    dashboard_path.write_text("\n".join(html_parts), encoding="utf-8")
    LOGGER.info("Plotly 合併圖表已儲存：%s", dashboard_path)

def save_plotly(fig, path: Path) -> list:
    """Save Plotly figure to HTML file and return the parsed figures."""
    figures = _to_plotly_figures(fig)
    if not figures:
        return []
    primary = figures[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        primary.write_html(str(path), include_plotlyjs="cdn")
        LOGGER.info("Plotly 圖表已儲存：%s", path)
    except Exception as err:
        LOGGER.error("儲存 plotly 圖表失敗 %s: %s", path, err)
    return figures

def save_figure(fig, path: Path) -> None:
    """Save Matplotlib figure safely."""
    if fig is None:
        LOGGER.warning("收到空的 figure，跳過儲存：%s", path)
        return
    if isinstance(fig, tuple):
        # qlib 部分工具可能返回 (fig, axes) 組合
        fig = fig[0]
    if hasattr(fig, "figure") and fig.__class__.__name__ != "Figure":
        fig = fig.figure
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("圖表已儲存：%s", path)


def dump_report_frames(
    recorder,
    model_dataset,
    *,
    universe: List[str],
    data_handler_config: Dict[str, object],
    segments: Dict[str, tuple[str, str]],
    port_config: Dict[str, object],
) -> None:
    """Load recorder artifacts, export CSV/figures."""
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    positions_dict = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    pred_df = recorder.load_object("pred.pkl")
    try:
        indicator_summary_df = recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
    except (FileNotFoundError, KeyError):
        indicator_summary_df = None
    try:
        indicator_daily_df = recorder.load_object("portfolio_analysis/indicators_normal_1day.pkl")
    except (FileNotFoundError, KeyError):
        indicator_daily_df = None

    report_path = REPORT_DIR.joinpath("report_normal_1day.csv")
    analysis_path = REPORT_DIR.joinpath("port_analysis_1day.csv")
    report_df.to_csv(report_path)
    analysis_df.reset_index().to_csv(analysis_path, index=False)
    LOGGER.info("報表 CSV 已輸出至 %s 與 %s", report_path, analysis_path)

    # Export flat position weights
    records = []
    for dt, pos in positions_dict.items():
        weights = pos.get_stock_weight_dict()
        if not weights:
            continue
        for symbol, weight in weights.items():
            records.append((dt, symbol, weight))
    weights_path = REPORT_DIR.joinpath("positions_weight.csv")
    if records:
        weights_df = pd.DataFrame(records, columns=["datetime", "instrument", "weight"])
        weights_df.sort_values(["datetime", "instrument"]).to_csv(weights_path, index=False)
        LOGGER.info("權重明細已輸出至 %s", weights_path)
    else:
        LOGGER.warning("權重資料為空，未產生 positions_weight.csv")

    if indicator_summary_df is not None:
        indicator_summary_path = REPORT_DIR.joinpath("indicator_analysis_1day.csv")
        indicator_summary_df.to_csv(indicator_summary_path)
        LOGGER.info("指標摘要已輸出至 %s", indicator_summary_path)
    if indicator_daily_df is not None:
        indicator_daily_path = REPORT_DIR.joinpath("indicators_normal_1day.csv")
        indicator_daily_df.to_csv(indicator_daily_path)
        LOGGER.info("指標時間序列已輸出至 %s", indicator_daily_path)

    # Summary text
    cumulative_return = (1 + report_df["return"]).prod() - 1
    benchmark_return = (1 + report_df["bench"]).prod() - 1
    trade_days = int((report_df["total_turnover"] > 0).sum())
    summary_lines = [
        f"Universe 檔數: {len(universe)}",
        f"資料期間: {data_handler_config['start_time']} ~ {data_handler_config['end_time']}",
        f"訓練區間: {segments['train'][0]} ~ {segments['train'][1]}",
        f"驗證區間: {segments['valid'][0]} ~ {segments['valid'][1]}",
        f"測試區間: {segments['test'][0]} ~ {segments['test'][1]}",
        f"回測期間: {port_config['backtest']['start_time']} ~ {port_config['backtest']['end_time']}",
        f"策略累積報酬: {cumulative_return:.4%}",
        f"基準累積報酬: {benchmark_return:.4%}",
        f"有交易日數: {trade_days}",
    ]
    if isinstance(analysis_df, pd.DataFrame) and 'risk' in analysis_df.columns:
        risk_series = analysis_df['risk']
        if isinstance(risk_series, pd.Series):
            for (category, metric), value in risk_series.items():
                if pd.notna(value):
                    summary_lines.append(f"risk.{category}.{metric}: {value:.6f}")
    if indicator_summary_df is not None and 'value' in indicator_summary_df.columns:
        for idx, value in indicator_summary_df['value'].items():
            if pd.notna(value):
                summary_lines.append(f"indicator_summary.{idx}: {value:.6f}")
    if indicator_daily_df is not None:
        numeric_cols = indicator_daily_df.select_dtypes(include='number')
        if not numeric_cols.empty:
            stats = numeric_cols.mean(numeric_only=True)
            for col, val in stats.items():
                if pd.notna(val):
                    summary_lines.append(f"indicator_daily_mean.{col}: {val:.6f}")
    summary_path = REPORT_DIR.joinpath("summary.txt")
    summary_path.write_text("\n".join(summary_lines))
    LOGGER.info("摘要統計已輸出至 %s", summary_path)

    # Turnover count by symbol (how many instruments changed weight each day)
    turnover_records = []
    prev_weights = None
    for dt in sorted(positions_dict.keys()):
        curr_weights = positions_dict[dt].get_stock_weight_dict()
        if prev_weights is None:
            changed = len(curr_weights)
        else:
            symbols = set(prev_weights) | set(curr_weights)
            changed = sum(
                1
                for sym in symbols
                if abs(curr_weights.get(sym, 0.0) - prev_weights.get(sym, 0.0)) > 1e-6
            )
        turnover_records.append((pd.to_datetime(dt), changed))
        prev_weights = curr_weights
    if turnover_records:
        turnover_df = pd.DataFrame(turnover_records, columns=["datetime", "changed_instruments"])
        turnover_df.sort_values("datetime", inplace=True)
        turnover_df.to_csv(REPORT_DIR.joinpath("turnover_count.csv"), index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(turnover_df["datetime"], turnover_df["changed_instruments"], width=1.0, align="center")
        ax.set_title("Daily Turnover (Count of Instruments Changed)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        fig.autofmt_xdate()
        save_figure(fig, FIG_DIR.joinpath("turnover_count.png"))

    # Custom figures
    strategy_curve = (1 + report_df["return"]).cumprod()
    benchmark_curve = (1 + report_df["bench"]).cumprod()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(strategy_curve.index, strategy_curve.values, label="Strategy")
    ax.plot(benchmark_curve.index, benchmark_curve.values, label="Benchmark")
    ax.set_title("Strategy vs Benchmark Cumulative Return")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save_figure(fig, FIG_DIR.joinpath("equity_curve.png"))

    if "total_turnover" in report_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(report_df.index, report_df["total_turnover"], width=1.0, align="center")
        ax.set_title("Daily Turnover")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover")
        fig.autofmt_xdate()
        save_figure(fig, FIG_DIR.joinpath("turnover.png"))

    # Information Coefficient (IC) analysis
    label_df = model_dataset.prepare("test", col_set="label")
    label_df.columns = ["label"]
    combined = pred_df.join(label_df, how="left").dropna()
    pred_label_path = REPORT_DIR.joinpath("pred_label.csv")
    combined.reset_index().to_csv(pred_label_path, index=False)
    LOGGER.info("預測與標籤資料已輸出至 %s", pred_label_path)

    def _calc_ic(group: pd.DataFrame) -> float:
        if group["score"].nunique() <= 1 or group["label"].nunique() <= 1:
            return float("nan")
        return group["score"].corr(group["label"], method="spearman")

    ic_series = combined.groupby(level=0).apply(_calc_ic)
    ic_series.to_csv(REPORT_DIR.joinpath("daily_ic.csv"), header=["ic"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ic_series.index, ic_series.values, marker="o", linestyle="-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Information Coefficient")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman IC")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save_figure(fig, FIG_DIR.joinpath("daily_ic.png"))

    # Official qlib analysis figures (HTML)
    plotly_sections = []
    try:
        plotly_sections.append(("report_graph", analysis_position.report_graph(report_df, show_notebook=False)))
    except Exception as err:
        LOGGER.error("分析圖 report_graph 生成失敗: %s", err)
    try:
        plotly_sections.append(("risk_analysis", analysis_position.risk_analysis_graph(analysis_df, report_df, show_notebook=False)))
    except Exception as err:
        LOGGER.error("分析圖 risk_analysis 生成失敗: %s", err)
    try:
        plotly_sections.append(("score_ic", analysis_position.score_ic_graph(combined, show_notebook=False)))
    except Exception as err:
        LOGGER.error("分析圖 score_ic 生成失敗: %s", err)
    try:
        plotly_sections.append(("model_performance", analysis_model.model_performance_graph(combined, show_notebook=False)))
    except Exception as err:
        LOGGER.error("分析圖 model_performance 生成失敗: %s", err)

    plotly_saved = []
    for name, section in plotly_sections:
        saved_entries = export_plotly_section(name, section)
        if saved_entries:
            plotly_saved.append((name, saved_entries))
    export_plotly_dashboard(plotly_saved)

    LOGGER.info("報表與圖表生成完成")


def run_combo(
    combo_name: str,
    handler_key: str,
    model_key: str,
    max_instruments: int | None,
    infer_processors: List[Dict[str, object]] | None = None,
    n_drop_override: int | None = None,
    topk_override: int | None = None,
    rebalance: str = "day",
    strategy_choice: str = "bucket",
    deal_price: str = "close",
    simulate_limit: bool = False,
    limit_slippage: float = 0.01,
) -> None:
    available = len(UNIVERSE)
    if max_instruments is None:
        LOGGER.info(
            "=== 執行組合：%s (handler=%s, model=%s, instruments=%d) ===",
            combo_name,
            handler_key,
            model_key,
            available,
        )
    else:
        LOGGER.info(
            "=== 執行組合：%s (handler=%s, model=%s, instruments=%d/%d) ===",
            combo_name,
            handler_key,
            model_key,
            min(available, max_instruments),
            available,
        )
    effective_name = combo_name
    if n_drop_override is not None:
        effective_name = f"{combo_name}_ndrop{n_drop_override}"
    if topk_override is not None:
        effective_name = f"{effective_name}_topk{topk_override}"
    if rebalance != "day":
        effective_name = f"{effective_name}_{rebalance}"
    if strategy_choice != "bucket":
        effective_name = f"{effective_name}_{strategy_choice}"
    set_output_dirs(effective_name)

    task_config = build_task_config(handler_key, model_key, UNIVERSE, max_instruments, infer_processors)
    port_config = build_port_analysis_config()
    if strategy_choice == "equal":
        port_config["strategy"]["class"] = "TopkDropoutStrategy"
        port_config["strategy"]["module_path"] = "qlib.contrib.strategy.signal_strategy"
    if n_drop_override is not None:
        port_config["strategy"]["kwargs"]["n_drop"] = n_drop_override
    if topk_override is not None:
        port_config["strategy"]["kwargs"]["topk"] = topk_override
    if rebalance != "day":
        port_config["executor"]["kwargs"]["time_per_step"] = rebalance
        port_config["backtest"]["exchange_kwargs"]["freq"] = rebalance

    # 成交價與限價模擬
    port_config["backtest"]["exchange_kwargs"]["deal_price"] = deal_price
    if simulate_limit:
        base_ex_kwargs = deepcopy(port_config["backtest"]["exchange_kwargs"])
        exchange_cfg = {
            "class": "TWLimitExchange",
            "module_path": "scripts.custom_exchange",
            "kwargs": {
                **base_ex_kwargs,
                "start_time": port_config["backtest"]["start_time"],
                "end_time": port_config["backtest"]["end_time"],
                "codes": task_config["dataset"]["kwargs"]["handler"]["kwargs"]["instruments"],
                "limit_slippage": limit_slippage,
            },
        }
        port_config["backtest"]["exchange_kwargs"] = {"exchange": exchange_cfg}

    LOGGER.info("建立模型與資料集配置")
    model = init_instance_by_config(task_config["model"])
    dataset = init_instance_by_config(task_config["dataset"])

    train_exp = f"tw_train_model_{combo_name}"
    LOGGER.info("開始訓練模型 - 實驗 %s", train_exp)
    with R.start(experiment_name=train_exp):
        R.log_params(**flatten_dict(task_config))
        model.fit(dataset)
        R.save_objects(trained_model=model)
        model_rid = R.get_recorder().id
    LOGGER.info("模型訓練完成，recorder id = %s", model_rid)

    backtest_exp = f"tw_backtest_{combo_name}"
    LOGGER.info("開始產生訊號與回測 - 實驗 %s", backtest_exp)
    with R.start(experiment_name=backtest_exp):
        train_recorder = R.get_recorder(recorder_id=model_rid, experiment_name=train_exp)
        trained_model = train_recorder.load_object("trained_model")

        current_recorder = R.get_recorder()
        signal_rec = SignalRecord(trained_model, dataset, current_recorder)
        signal_rec.generate()

        strategy_kwargs = port_config["strategy"]["kwargs"]
        strategy_kwargs["model"] = trained_model
        strategy_kwargs["dataset"] = dataset

        port_rec = PortAnaRecord(current_recorder, port_config, "day")
        port_rec.generate()
        backtest_rid = current_recorder.id

    LOGGER.info("回測完成，recorder id = %s", backtest_rid)

    LOGGER.info("載入結果並輸出報表/圖表")
    recorder = R.get_recorder(recorder_id=backtest_rid, experiment_name=backtest_exp)
    handler_kwargs = task_config["dataset"]["kwargs"]["handler"]["kwargs"]
    segments = task_config["dataset"]["kwargs"]["segments"]
    dump_report_frames(
        recorder,
        dataset,
        universe=handler_kwargs["instruments"],
        data_handler_config=handler_kwargs,
        segments=segments,
        port_config=port_config,
    )
    LOGGER.info("組合 %s 完成。輸出根目錄：%s", effective_name, OUTPUT_ROOT)



def main() -> None:
    args = parse_args()
    combos = resolve_combos(args.combo)

    LOGGER.info("初始化 Qlib，資料來源：%s", PROVIDER_URI)
    qlib.init(provider_uri=str(PROVIDER_URI), region=REGION)
    LOGGER.info("Universe 規模：%d 檔", len(UNIVERSE))

    for combo_name in combos:
        spec = COMBO_CONFIGS[combo_name]
        run_combo(
            combo_name,
            spec["handler"],
            spec["model"],
            spec.get("max_instruments"),
            spec.get("infer_processors"),
            n_drop_override=args.n_drop,
            topk_override=args.topk,
            rebalance=args.rebalance,
            strategy_choice=args.strategy,
            deal_price=args.deal_price,
            simulate_limit=args.simulate_limit,
            limit_slippage=args.limit_slippage,
        )


if __name__ == "__main__":
    main()
