from __future__ import annotations

import numpy as np
import pandas as pd


def calc_daily_ic(pred: pd.Series, label: pd.Series) -> tuple[float, float, int]:
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    if not isinstance(pred, pd.Series) or pred.index.nlevels != 2:
        return float("nan"), float("nan"), 0

    if isinstance(label, pd.DataFrame):
        label = label.iloc[:, 0]
    if not isinstance(label, pd.Series) or label.index.nlevels != 2:
        return float("nan"), float("nan"), 0

    pred_df = pd.DataFrame({"score": pred})
    label_df = pd.DataFrame({"label": label})
    combined = pred_df.join(label_df, how="left").dropna()
    if combined.empty:
        return float("nan"), float("nan"), 0

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 2 or group["score"].nunique() <= 1 or group["label"].nunique() <= 1:
            return np.nan
        return group["score"].corr(group["label"], method="spearman")

    ic_series = combined.groupby(level=0).apply(_ic).dropna()
    if len(ic_series) == 0:
        return float("nan"), float("nan"), 0
    return float(ic_series.mean()), float(ic_series.std()), int(ic_series.count())
