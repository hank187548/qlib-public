from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(value: pd.Series | pd.DataFrame) -> pd.Series | None:
    if isinstance(value, pd.DataFrame):
        value = value.iloc[:, 0]
    if not isinstance(value, pd.Series) or value.index.nlevels != 2:
        return None
    return value


def _daily_corr_series(pred: pd.Series, label: pd.Series, *, method: str) -> pd.Series:
    pred_series = _as_series(pred)
    label_series = _as_series(label)
    if pred_series is None or label_series is None:
        return pd.Series(dtype="float64")

    pred_df = pd.DataFrame({"score": pred_series})
    label_df = pd.DataFrame({"label": label_series})
    combined = pred_df.join(label_df, how="left").dropna()
    if combined.empty:
        return pd.Series(dtype="float64")

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 2 or group["score"].nunique() <= 1 or group["label"].nunique() <= 1:
            return np.nan
        return group["score"].corr(group["label"], method=method)

    return combined.groupby(level=0).apply(_ic).dropna()


def _summary(series: pd.Series, prefix: str) -> dict[str, float | int]:
    mean = float(series.mean()) if not series.empty else float("nan")
    std = float(series.std()) if not series.empty else float("nan")
    icir = mean / std if pd.notna(mean) and pd.notna(std) and std != 0 else float("nan")
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_icir": float(icir),
        f"{prefix}_days": int(series.count()),
    }


def calc_daily_ic(pred: pd.Series, label: pd.Series) -> tuple[float, float, int]:
    ic_series = _daily_corr_series(pred, label, method="spearman")
    if len(ic_series) == 0:
        return float("nan"), float("nan"), 0
    return float(ic_series.mean()), float(ic_series.std()), int(ic_series.count())


def calc_ic_metrics(pred: pd.Series | pd.DataFrame, label: pd.Series | pd.DataFrame) -> dict[str, float | int]:
    pearson_series = _daily_corr_series(pred, label, method="pearson")
    rank_series = _daily_corr_series(pred, label, method="spearman")
    metrics: dict[str, float | int] = {}
    metrics.update(_summary(pearson_series, "ic"))
    metrics.update(_summary(rank_series, "rank_ic"))
    return metrics


def calc_daily_rank_ic(pred: pd.Series, label: pd.Series) -> tuple[float, float, int]:
    ic_series = _daily_corr_series(pred, label, method="spearman")
    if len(ic_series) == 0:
        return float("nan"), float("nan"), 0
    return float(ic_series.mean()), float(ic_series.std()), int(ic_series.count())


def calc_daily_pearson_ic(pred: pd.Series, label: pd.Series) -> tuple[float, float, int]:
    ic_series = _daily_corr_series(pred, label, method="pearson")
    if len(ic_series) == 0:
        return float("nan"), float("nan"), 0
    return float(ic_series.mean()), float(ic_series.std()), int(ic_series.count())
