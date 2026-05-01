#!/usr/bin/env python3
"""Build a reusable Alpha158 parquet cache from the Qlib OHLCV provider."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import qlib
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset.handler import DataHandlerLP


ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.data_layout import ALPHA158_CACHE_DIR, build_exp_manager_config, resolve_provider_uri, resolve_workspace_path
from qlib_tw.research.settings import BASE_DATA_HANDLER_CONFIG, REGION, UNIVERSE


LOGGER = logging.getLogger("qlib_tw.research.build_alpha158_cache")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build precomputed Alpha158 feature/label cache")
    parser.add_argument("--provider-uri", type=Path, default=None, help="Qlib provider uri; defaults to Data/qlib_data")
    parser.add_argument("--output-dir", type=Path, default=None, help="Cache output dir; defaults to Data/alpha_158_data")
    parser.add_argument("--start-time", default=None, help="Feature start date")
    parser.add_argument("--end-time", default=None, help="Feature end date")
    parser.add_argument("--fit-start-time", default=None, help="Processor fit start date metadata")
    parser.add_argument("--fit-end-time", default=None, help="Processor fit end date metadata")
    parser.add_argument("--max-instruments", type=int, default=None, help="Optional cap for smoke-testing cache builds")
    parser.add_argument(
        "--max-abs-value",
        type=float,
        default=None,
        help="Optionally replace feature values with abs(value) above this threshold by NaN",
    )
    return parser


def _resolved_output_dir(path: Path | None) -> Path:
    if path is None:
        return ALPHA158_CACHE_DIR.resolve()
    resolved = resolve_workspace_path(path)
    if resolved is None:
        raise ValueError(f"Unable to resolve output dir: {path}")
    return resolved


def _clean_frame(df: pd.DataFrame, *, max_abs_value: float | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    stats = {
        "nan_count": int(cleaned.isna().sum().sum()),
    }
    if max_abs_value is not None:
        numeric = cleaned.select_dtypes(include="number")
        too_large = numeric.abs() > max_abs_value
        stats["too_large_count"] = int(too_large.sum().sum())
        if stats["too_large_count"]:
            cleaned.loc[:, numeric.columns] = numeric.mask(too_large)
            stats["nan_count"] = int(cleaned.isna().sum().sum())
    return cleaned, stats


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    provider_uri = resolve_provider_uri(args.provider_uri)
    output_dir = _resolved_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    handler_config = dict(BASE_DATA_HANDLER_CONFIG)
    start_time = args.start_time or str(handler_config["start_time"])
    end_time = args.end_time or str(handler_config["end_time"])
    fit_start_time = args.fit_start_time or str(handler_config["fit_start_time"])
    fit_end_time = args.fit_end_time or str(handler_config["fit_end_time"])
    price_basis = str(handler_config.get("price_basis", "provider"))
    instruments = UNIVERSE if args.max_instruments is None else UNIVERSE[: args.max_instruments]

    LOGGER.info("Initialize Qlib, provider uri: %s", provider_uri)
    qlib.init(provider_uri=str(provider_uri), region=REGION, exp_manager=build_exp_manager_config())
    LOGGER.info("Build Alpha158 cache for %d instruments, %s ~ %s", len(instruments), start_time, end_time)

    handler = Alpha158(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        infer_processors=[],
        learn_processors=[],
        price_basis=price_basis,
    )
    feature = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
    feature, feature_stats = _clean_frame(feature, max_abs_value=args.max_abs_value)
    label, label_stats = _clean_frame(label)

    feature_path = output_dir / "alpha158_feature.parquet"
    label_path = output_dir / "alpha158_label.parquet"
    metadata_path = output_dir / "metadata.json"
    feature.to_parquet(feature_path, engine="pyarrow")
    label.to_parquet(label_path, engine="pyarrow")

    metadata = {
        "provider_uri": str(provider_uri),
        "output_dir": str(output_dir),
        "start_time": start_time,
        "end_time": end_time,
        "fit_start_time": fit_start_time,
        "fit_end_time": fit_end_time,
        "price_basis": price_basis,
        "instrument_count": len(instruments),
        "feature_shape": list(feature.shape),
        "label_shape": list(label.shape),
        "feature_stats": feature_stats,
        "label_stats": label_stats,
        "max_abs_value": args.max_abs_value,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved Alpha158 feature cache: %s", feature_path)
    LOGGER.info("Saved Alpha158 label cache: %s", label_path)
    LOGGER.info("Saved Alpha158 cache metadata: %s", metadata_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    raise SystemExit(main())
