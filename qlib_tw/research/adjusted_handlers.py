from __future__ import annotations

from qlib.contrib.data.handler import (
    Alpha158,
    _DEFAULT_INFER_PROCESSORS,
    _DEFAULT_LEARN_PROCESSORS,
    check_transform_proc,
)
from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from qlib.data.dataset.handler import DataHandlerLP


_PRICE_TOKENS = ("open", "high", "low", "close", "vwap")


def _to_adjusted_expr(expr: str) -> str:
    adjusted = expr
    for token in _PRICE_TOKENS:
        adjusted = adjusted.replace(f"${token}", f"({'$' + token}*$factor)")
    return adjusted


def _to_adjusted_exprs(fields: list[str]) -> list[str]:
    return [_to_adjusted_expr(field) for field in fields]


class AdjustedAlpha158(Alpha158):
    """
    Alpha158 features on forward-adjusted prices.

    Prices are transformed to `price * factor` in the research handler so training and
    inference share the same adjusted scale. Execution logic remains unchanged and still
    uses the raw provider prices.
    """

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        fields, names = Alpha158DL.get_feature_config(conf)
        return _to_adjusted_exprs(fields), names

    def get_label_config(self):
        return _to_adjusted_exprs(["Ref($close, -2)/Ref($close, -1) - 1"]), ["LABEL0"]


class AdjustedAlpha158vwap(AdjustedAlpha158):
    def get_label_config(self):
        return _to_adjusted_exprs(["Ref($vwap, -2)/Ref($vwap, -1) - 1"]), ["LABEL0"]


class AdjustedAlpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        feature_fields, feature_names = Alpha360DL.get_feature_config()

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": (_to_adjusted_exprs(feature_fields), feature_names),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return _to_adjusted_exprs(["Ref($close, -2)/Ref($close, -1) - 1"]), ["LABEL0"]


class AdjustedAlpha360vwap(AdjustedAlpha360):
    def get_label_config(self):
        return _to_adjusted_exprs(["Ref($vwap, -2)/Ref($vwap, -1) - 1"]), ["LABEL0"]
