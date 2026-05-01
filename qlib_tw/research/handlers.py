from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.data.dataset.handler import DataHandlerLP

from qlib_tw.data_layout import ALPHA158_CACHE_DIR, resolve_workspace_path


class CachedAlpha158(DataHandlerLP):
    """Load precomputed Alpha158 feature/label cache for normal research runs."""

    def __init__(
        self,
        instruments="all",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        cache_dir: str | Path | None = None,
        **kwargs,
    ):
        del freq
        kwargs.pop("price_basis", None)
        kwargs.pop("filter_pipe", None)
        kwargs.pop("inst_processors", None)

        resolved_cache_dir = resolve_workspace_path(cache_dir) if cache_dir is not None else ALPHA158_CACHE_DIR.resolve()
        feature_path = resolved_cache_dir / "alpha158_feature.parquet"
        label_path = resolved_cache_dir / "alpha158_label.parquet"
        missing = [str(path) for path in (feature_path, label_path) if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Alpha158 cache is missing. Build it first with "
                "`.venv/bin/python scripts/research/build_alpha158_cache.py`. "
                f"Missing: {', '.join(missing)}"
            )

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)
        feature_df = pd.read_parquet(feature_path, engine="pyarrow")
        label_df = pd.read_parquet(label_path, engine="pyarrow")
        data_loader = {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": feature_df,
                    "label": label_df,
                }
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )
