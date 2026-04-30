from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
import pandas as pd
import plotly.io as pio
from qlib.contrib.report import analysis_model, analysis_position
from qlib.workflow.record_temp import SignalRecord

from qlib_tw.research.paths import WorkflowPaths


matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOGGER = logging.getLogger("qlib_tw.research.reports")

PLOTLY_DASHBOARD_SECTION_ORDER = {
    "report_graph": 0,
    "risk_analysis": 1,
    "score_ic": 2,
    "model_performance": 3,
}


def _to_plotly_figures(obj) -> list:
    figures = []
    if obj is None:
        return figures
    if hasattr(obj, "to_plotly_json"):
        try:
            figures.append(obj)
        except Exception as err:
            LOGGER.error("Failed to import plotly figure: %s", err)
        return figures
    if isinstance(obj, dict):
        if "application/vnd.plotly.v1+json" in obj:
            bundle = obj["application/vnd.plotly.v1+json"]
            json_payload = bundle if isinstance(bundle, str) else json.dumps(bundle)
            try:
                figures.append(pio.from_json(json_payload))
            except Exception as err:
                LOGGER.error("Failed to parse plotly bundle: %s", err)
            return figures
        if "data" in obj and "layout" in obj:
            try:
                json_payload = json.dumps({"data": obj["data"], "layout": obj["layout"]})
                figures.append(pio.from_json(json_payload))
            except Exception as err:
                LOGGER.error("Failed to parse plotly dict: %s", err)
            return figures
    if isinstance(obj, (list, tuple)):
        for item in obj:
            figures.extend(_to_plotly_figures(item))
        return figures
    LOGGER.warning("Unable to parse plotly object: %s", type(obj))
    return figures


def export_plotly_section(paths: WorkflowPaths, name: str, obj) -> list:
    figs = _to_plotly_figures(obj)
    saved = []
    if not figs:
        return saved
    for idx, figure in enumerate(figs, start=1):
        suffix = "" if idx == 1 else f"_{idx}"
        file_path = paths.fig_dir / f"{name}{suffix}.html"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            figure.write_html(str(file_path), include_plotlyjs="cdn")
            LOGGER.info("Saved Plotly chart: %s", file_path)
            saved.append((file_path, figure))
        except Exception as err:
            LOGGER.error("Failed to save Plotly chart %s: %s", file_path, err)
    return saved


def export_plotly_dashboard(paths: WorkflowPaths, sections: list) -> None:
    if not sections:
        return
    dashboard_path = paths.fig_dir / "analysis_dashboard.html"
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
            title = name if len(entries) == 1 else f"{name} ({idx})"
            html_parts.append(f'<h2 style="margin-top:40px;">{title}</h2>')
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
    LOGGER.info("Saved combined Plotly dashboard: %s", dashboard_path)


def _ordered_plotly_sections(sections: list) -> list:
    fallback_order = len(PLOTLY_DASHBOARD_SECTION_ORDER)
    return sorted(
        sections,
        key=lambda item: PLOTLY_DASHBOARD_SECTION_ORDER.get(item[0], fallback_order),
    )


def save_figure(fig, path: Path) -> None:
    if fig is None:
        LOGGER.warning("Received empty figure, skip saving: %s", path)
        return
    if isinstance(fig, tuple):
        fig = fig[0]
    if hasattr(fig, "figure") and fig.__class__.__name__ != "Figure":
        fig = fig.figure
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved chart: %s", path)


def _load_raw_label(recorder, model_dataset) -> pd.DataFrame | None:
    try:
        raw_label = recorder.load_object("label.pkl")
    except (FileNotFoundError, KeyError):
        raw_label = SignalRecord.generate_label(model_dataset)
    if raw_label is None:
        LOGGER.warning("Raw label is unavailable; skip pred/label analysis outputs")
        return None
    return raw_label


def _calc_ic(group: pd.DataFrame) -> float:
    if group["score"].nunique() <= 1 or group["label"].nunique() <= 1:
        return float("nan")
    return group["score"].corr(group["label"], method="spearman")


def _export_prediction_outputs(recorder, model_dataset, paths: WorkflowPaths) -> tuple[pd.DataFrame | None, pd.Series | None, list]:
    pred_df = recorder.load_object("pred.pkl")
    label_df = _load_raw_label(recorder, model_dataset)
    if label_df is None:
        LOGGER.warning("Prediction/label report generation skipped because raw label is missing")
        return None, None, []
    label_df.columns = ["label"]
    combined = pred_df.join(label_df, how="left").dropna()
    pred_label_path = paths.report_dir / "pred_label.csv"
    combined.reset_index().to_csv(pred_label_path, index=False)
    LOGGER.info("Prediction/raw-label data exported to %s", pred_label_path)

    ic_series = combined.groupby(level=0).apply(_calc_ic)
    ic_series.to_csv(paths.report_dir / "daily_ic.csv", header=["ic"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ic_series.index, ic_series.values, marker="o", linestyle="-")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Daily Information Coefficient")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spearman IC")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    save_figure(fig, paths.fig_dir / "daily_ic.png")

    plotly_sections = []
    try:
        plotly_sections.append(("score_ic", analysis_position.score_ic_graph(combined, show_notebook=False)))
    except Exception as err:
        LOGGER.error("Failed to generate score_ic: %s", err)
    try:
        plotly_sections.append(("model_performance", analysis_model.model_performance_graph(combined, show_notebook=False)))
    except Exception as err:
        LOGGER.error("Failed to generate model_performance: %s", err)
    return combined, ic_series, plotly_sections


def _prediction_summary_lines(combined: pd.DataFrame | None, ic_series: pd.Series | None) -> list[str]:
    if combined is None or ic_series is None:
        return []

    mean_ic = float(ic_series.mean()) if not ic_series.empty else float("nan")
    std_ic = float(ic_series.std()) if not ic_series.empty else float("nan")
    icir = mean_ic / std_ic if pd.notna(mean_ic) and pd.notna(std_ic) and std_ic != 0 else float("nan")
    positive_days = int((ic_series > 0).sum()) if not ic_series.empty else 0
    total_days = int(ic_series.notna().sum()) if not ic_series.empty else 0

    return [
        f"Prediction rows: {len(combined)}",
        f"Prediction days: {combined.index.get_level_values(0).nunique()}",
        f"Mean daily IC: {mean_ic:.6f}" if pd.notna(mean_ic) else "Mean daily IC: nan",
        f"IC std: {std_ic:.6f}" if pd.notna(std_ic) else "IC std: nan",
        f"ICIR: {icir:.6f}" if pd.notna(icir) else "ICIR: nan",
        f"Positive IC days: {positive_days}/{total_days}",
    ]


def _save_summary(paths: WorkflowPaths, lines: list[str]) -> None:
    summary_path = paths.report_dir / "summary.txt"
    summary_path.write_text("\n".join(lines))
    LOGGER.info("Summary stats exported to %s", summary_path)


def _save_plotly_sections(paths: WorkflowPaths, plotly_sections: list) -> None:
    plotly_saved = []
    for name, section in plotly_sections:
        saved_entries = export_plotly_section(paths, name, section)
        if saved_entries:
            plotly_saved.append((name, saved_entries))
    export_plotly_dashboard(paths, _ordered_plotly_sections(plotly_saved))


def dump_model_frames(
    recorder,
    model_dataset,
    *,
    universe: List[str],
    data_handler_config: Dict[str, object],
    segments: Dict[str, tuple[str, str]],
    paths: WorkflowPaths,
) -> None:
    combined, ic_series, plotly_sections = _export_prediction_outputs(recorder, model_dataset, paths)
    summary_lines = [
        f"Universe size: {len(universe)}",
        f"Data period: {data_handler_config['start_time']} ~ {data_handler_config['end_time']}",
        f"Train range: {segments['train'][0]} ~ {segments['train'][1]}",
        f"Validation range: {segments['valid'][0]} ~ {segments['valid'][1]}",
        f"Test range: {segments['test'][0]} ~ {segments['test'][1]}",
    ]
    summary_lines.extend(_prediction_summary_lines(combined, ic_series))
    _save_summary(paths, summary_lines)
    _save_plotly_sections(paths, plotly_sections)
    LOGGER.info("Model analysis export complete")


def dump_report_frames(
    recorder,
    model_dataset,
    *,
    universe: List[str],
    data_handler_config: Dict[str, object],
    segments: Dict[str, tuple[str, str]],
    port_config: Dict[str, object],
    paths: WorkflowPaths,
) -> None:
    report_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    positions_dict = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    try:
        indicator_summary_df = recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
    except (FileNotFoundError, KeyError):
        indicator_summary_df = None
    try:
        indicator_daily_df = recorder.load_object("portfolio_analysis/indicators_normal_1day.pkl")
    except (FileNotFoundError, KeyError):
        indicator_daily_df = None

    report_path = paths.report_dir / "report_normal_1day.csv"
    analysis_path = paths.report_dir / "port_analysis_1day.csv"
    report_df.to_csv(report_path)
    analysis_df.reset_index().to_csv(analysis_path, index=False)
    LOGGER.info("Report CSV exported to %s and %s", report_path, analysis_path)

    records = []
    for dt, pos in positions_dict.items():
        weights = pos.get_stock_weight_dict()
        if not weights:
            continue
        for symbol, weight in weights.items():
            records.append((dt, symbol, weight))
    weights_path = paths.report_dir / "positions_weight.csv"
    if records:
        weights_df = pd.DataFrame(records, columns=["datetime", "instrument", "weight"])
        weights_df.sort_values(["datetime", "instrument"]).to_csv(weights_path, index=False)
        LOGGER.info("Position weights exported to %s", weights_path)
    else:
        LOGGER.warning("Position weight data is empty; positions_weight.csv not generated")

    if indicator_summary_df is not None:
        indicator_summary_path = paths.report_dir / "indicator_analysis_1day.csv"
        indicator_summary_df.to_csv(indicator_summary_path)
        LOGGER.info("Indicator summary exported to %s", indicator_summary_path)
    if indicator_daily_df is not None:
        indicator_daily_path = paths.report_dir / "indicators_normal_1day.csv"
        indicator_daily_df.to_csv(indicator_daily_path)
        LOGGER.info("Indicator time series exported to %s", indicator_daily_path)

    cumulative_return = (1 + report_df["return"]).prod() - 1
    benchmark_return = (1 + report_df["bench"]).prod() - 1
    trade_days = int((report_df["total_turnover"] > 0).sum())
    summary_lines = [
        f"Universe size: {len(universe)}",
        f"Data period: {data_handler_config['start_time']} ~ {data_handler_config['end_time']}",
        f"Train range: {segments['train'][0]} ~ {segments['train'][1]}",
        f"Validation range: {segments['valid'][0]} ~ {segments['valid'][1]}",
        f"Test range: {segments['test'][0]} ~ {segments['test'][1]}",
        f"Backtest period: {port_config['backtest']['start_time']} ~ {port_config['backtest']['end_time']}",
        f"Strategy cumulative return: {cumulative_return:.4%}",
        f"Benchmark cumulative return: {benchmark_return:.4%}",
        f"Trading days: {trade_days}",
    ]
    if isinstance(analysis_df, pd.DataFrame) and "risk" in analysis_df.columns:
        risk_series = analysis_df["risk"]
        if isinstance(risk_series, pd.Series):
            for (category, metric), value in risk_series.items():
                if pd.notna(value):
                    summary_lines.append(f"risk.{category}.{metric}: {value:.6f}")
    if indicator_summary_df is not None and "value" in indicator_summary_df.columns:
        for idx, value in indicator_summary_df["value"].items():
            if pd.notna(value):
                summary_lines.append(f"indicator_summary.{idx}: {value:.6f}")
    if indicator_daily_df is not None:
        numeric_cols = indicator_daily_df.select_dtypes(include="number")
        if not numeric_cols.empty:
            stats = numeric_cols.mean(numeric_only=True)
            for col, val in stats.items():
                if pd.notna(val):
                    summary_lines.append(f"indicator_daily_mean.{col}: {val:.6f}")
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
        turnover_df.to_csv(paths.report_dir / "turnover_count.csv", index=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(turnover_df["datetime"], turnover_df["changed_instruments"], width=1.0, align="center")
        ax.set_title("Daily Turnover (Count of Instruments Changed)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Count")
        fig.autofmt_xdate()
        save_figure(fig, paths.fig_dir / "turnover_count.png")

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
    save_figure(fig, paths.fig_dir / "equity_curve.png")

    if "total_turnover" in report_df.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(report_df.index, report_df["total_turnover"], width=1.0, align="center")
        ax.set_title("Daily Turnover")
        ax.set_xlabel("Date")
        ax.set_ylabel("Turnover")
        fig.autofmt_xdate()
        save_figure(fig, paths.fig_dir / "turnover.png")
    combined, ic_series, plotly_sections = _export_prediction_outputs(recorder, model_dataset, paths)
    summary_lines.extend(_prediction_summary_lines(combined, ic_series))
    _save_summary(paths, summary_lines)

    try:
        plotly_sections.append(("report_graph", analysis_position.report_graph(report_df, show_notebook=False)))
    except Exception as err:
        LOGGER.error("Failed to generate report_graph: %s", err)
    try:
        plotly_sections.append(("risk_analysis", analysis_position.risk_analysis_graph(analysis_df, report_df, show_notebook=False)))
    except Exception as err:
        LOGGER.error("Failed to generate risk_analysis: %s", err)
    _save_plotly_sections(paths, plotly_sections)
    LOGGER.info("Report and chart generation complete")
