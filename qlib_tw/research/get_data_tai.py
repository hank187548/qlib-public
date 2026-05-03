#!/usr/bin/env python3
"""
Download Taiwan stock data from Yahoo Finance and TWSE open APIs, then
transform them into a Qlib provider dataset.

Two workflows are supported:
- `collect`: Use Yahoo Finance to fetch historical daily bars for Taiwan
  tickers, store the raw download snapshot under `Raw_data/`, build
  processed Qlib-ready CSV snapshots under `Process_data/`, then convert
  them into Qlib binary storage under `qlib_data/`.
- `process`: Rebuild `Process_data/` directly from an existing `Raw_data/`
  snapshot without re-downloading Yahoo data.
- `dump`: Rebuild `qlib_data/` directly from an existing `Process_data/`
  snapshot.
- `openapi`: Fetch the latest TWSE open-data datasets (e.g., BWIBBU_ALL) and
  store the JSON/CSV snapshots locally.

The script avoids touching the original Qlib package so you can extend it
freely.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qlib_tw.data_layout import PROCESS_DATA_DIR, QLIB_DATA_DIR, RAW_DATA_DIR

LOG = logging.getLogger("GetDataTai")
SYMBOL_CSV_DTYPES = {"symbol": "string", "source_symbol": "string"}

DEFAULT_COLLECT_CONFIG = {
    # Update these two dates to control the Yahoo download window.
    "start": "2009-01-01",
    "end": pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
    # Update these paths if you want to store the dataset elsewhere.
    "target_dir": QLIB_DATA_DIR,
    "raw_dir": RAW_DATA_DIR,
    "process_dir": PROCESS_DATA_DIR,
    # Leave symbols=None to fetch the entire TWSE list automatically.
    "symbols": None,
    "suffix": ".TW",
    "pause": 0.5,
}

# Always make sure benchmark data is available even if the TWSE API skips it.
DEFAULT_BENCHMARKS = ["^TWII"]

TWSE_OPENAPI_ENDPOINTS = {
    "bwibbu_all": "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL",
    "stock_day_avg_all": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_AVG_ALL",
    "stock_day_all": "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL",
    "fmsrfk_all": "https://openapi.twse.com.tw/v1/exchangeReport/FMSRFK_ALL",
    "fmnptk_all": "https://openapi.twse.com.tw/v1/exchangeReport/FMNPTK_ALL",
    "mi_index": "https://openapi.twse.com.tw/v1/exchangeReport/MI_INDEX",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def code_to_fname(code: str) -> str:
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *{f"COM{i}" for i in range(10)},
        *{f"LPT{i}" for i in range(10)},
    }
    prefix = "_qlib_"
    if str(code).upper() in reserved:
        code = prefix + str(code)
    return code


def fname_to_code(fname: str) -> str:
    prefix = "_qlib_"
    return fname[len(prefix) :] if fname.startswith(prefix) else fname


def ensure_taiwan_yahoo_symbol(symbol: str, default_suffix: str = ".TW") -> str:
    symbol = symbol.strip()
    if symbol.endswith(".TW") or symbol.endswith(".TWO"):
        return symbol
    if symbol.endswith(".T") or symbol.endswith(".TW".lower()):
        return symbol.upper()
    if symbol.isdigit():
        return f"{symbol}{default_suffix}"
    return symbol.upper()


# ---------------------------------------------------------------------------
# Yahoo Finance workflow
# ---------------------------------------------------------------------------


class YahooTaiwanCollector:
    def __init__(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        raw_dir: Path,
        pause: float = 0.5,
        default_suffix: str = ".TW",
    ):
        self.symbols = list(symbols)
        self.start = start.floor("D")
        self.end = end.floor("D")
        self.raw_dir = Path(raw_dir).expanduser().resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.default_suffix = default_suffix
        self._yf = None  # Lazy import holder
        self.failed_symbols: Dict[str, Tuple[str, Optional[str]]] = {}

    def _ensure_yfinance(self):
        if self._yf is None:
            try:
                import yfinance as yf  # type: ignore
            except ImportError as exc:  # pragma: no cover - runtime check
                raise SystemExit(
                    "yfinance is required. Please run `pip install yfinance` before using the collect command"
                ) from exc
            self._yf = yf
        return self._yf

    @staticmethod
    def fetch_default_symbols() -> List[str]:
        try:
            resp = requests.get(TWSE_OPENAPI_ENDPOINTS["stock_day_all"], timeout=30)
            resp.raise_for_status()
            data = resp.json()
            raw_codes = {item.get("Code", "").strip() for item in data if item.get("Code")}
            filtered = sorted(code for code in raw_codes if code and code.isdigit() and len(code) == 4)
            return filtered
        except requests.RequestException as err:
            LOG.error("Failed to fetch symbol universe from TWSE open API: %s", err)
            return []

    def _download_history(self, base_symbol: str, yahoo_symbol: str) -> pd.DataFrame:
        period_end = (self.end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        yf_module = self._ensure_yfinance()
        self.failed_symbols.pop(base_symbol, None)
        try:
            df = yf_module.download(
                yahoo_symbol,
                start=self.start.strftime("%Y-%m-%d"),
                end=period_end,
                interval="1d",
                auto_adjust=False,
                progress=False,
            )
        except Exception as err:  # pragma: no cover - network/runtime guard
            LOG.warning("Skipping %s due to Yahoo download error: %s", base_symbol, err)
            self.failed_symbols[base_symbol] = ("download_error", str(err))
            return pd.DataFrame()
        if df.empty:
            LOG.warning("Skipping %s because Yahoo returned an empty dataset", base_symbol)
            self.failed_symbols[base_symbol] = ("empty_dataframe", None)
            return df
        df.reset_index(inplace=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [str(col[0]).strip().lower() for col in df.columns]
        else:
            df.columns = [str(col).strip().lower() for col in df.columns]
        df.columns = [col.replace(" ", "_") for col in df.columns]
        name_map = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "adj_close": "adj_close",
            "adjclose": "adj_close",
            "volume": "volume",
        }
        df.rename(columns={col: name_map[col] for col in list(df.columns) if col in name_map}, inplace=True)
        required = {"open", "high", "low", "close"}
        missing = {col for col in required if col not in df.columns}
        if missing:
            LOG.warning("Skipping %s due to missing columns: %s", base_symbol, sorted(missing))
            self.failed_symbols[base_symbol] = ("missing_columns", ",".join(sorted(missing)))
            return pd.DataFrame()
        return df

    def save_dataframe(self, base_symbol: str, yahoo_symbol: str, df: pd.DataFrame) -> Optional[Path]:
        if df.empty:
            return None
        raw_df = df.copy()
        raw_df.insert(0, "symbol", base_symbol.upper())
        raw_df.insert(1, "source_symbol", yahoo_symbol.upper())
        csv_path = self.raw_dir / f"{code_to_fname(base_symbol).lower()}.csv"
        raw_df.to_csv(csv_path, index=False)
        return csv_path

    def run(self) -> List[Path]:
        saved: List[Path] = []
        for code in self.symbols:
            base_symbol = code.upper()
            yahoo_symbol = ensure_taiwan_yahoo_symbol(code, self.default_suffix)
            LOG.info("Downloading Yahoo data for %s (%s)", base_symbol, yahoo_symbol)
            df = self._download_history(base_symbol, yahoo_symbol)
            if df.empty:
                LOG.warning("No data returned for %s", base_symbol)
                continue
            path = self.save_dataframe(base_symbol, yahoo_symbol, df)
            if path:
                saved.append(path)
            time.sleep(self.pause)
        LOG.info("Finished Yahoo download for %d symbols", len(saved))
        return saved


def _normalize_factor(series: pd.Series) -> pd.Series:
    factor = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    factor = factor.where(factor > 0)
    return factor.fillna(1.0).astype(np.float32)


def build_processed_dataframe(base_symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    processed = df.copy()
    processed["symbol"] = base_symbol.upper()
    if "source_symbol" not in processed.columns:
        processed["source_symbol"] = base_symbol.upper()
    processed["date"] = pd.to_datetime(processed["date"])
    processed.sort_values("date", inplace=True)

    def _series_for(primary: str, fallback: str | None = None, *, default=np.nan) -> pd.Series:
        if primary in processed.columns:
            return pd.to_numeric(processed[primary], errors="coerce")
        if fallback is not None and fallback in processed.columns:
            return pd.to_numeric(processed[fallback], errors="coerce")
        return pd.Series(default, index=processed.index, dtype="float64")

    raw_open = _series_for("raw_open", "open")
    raw_high = _series_for("raw_high", "high")
    raw_low = _series_for("raw_low", "low")
    raw_close = _series_for("raw_close", "close")
    raw_volume = _series_for("raw_volume", "volume", default=0.0).fillna(0.0)

    raw_adj_close = _series_for("raw_adj_close", "adj_close")
    if not raw_adj_close.isna().all():
        factor = _normalize_factor(raw_adj_close / raw_close.replace(0, np.nan))
    elif "factor" in processed.columns:
        factor = _normalize_factor(processed["factor"])
        raw_adj_close = raw_close * factor
    else:
        raise ValueError(
            f"{base_symbol}: unable to build Process_data without adj_close or factor in the raw input"
        )

    raw_vwap = _series_for("raw_vwap", "vwap")
    if raw_vwap.isna().all():
        raw_vwap = (raw_high + raw_low + raw_close) / 3.0

    processed["raw_open"] = raw_open
    processed["raw_high"] = raw_high
    processed["raw_low"] = raw_low
    processed["raw_close"] = raw_close
    processed["raw_adj_close"] = raw_adj_close
    processed["raw_volume"] = raw_volume
    processed["raw_value"] = raw_close * raw_volume
    processed["raw_change"] = raw_close.pct_change(fill_method=None).fillna(0.0)
    processed["raw_vwap"] = raw_vwap

    processed["open"] = raw_open * factor
    processed["high"] = raw_high * factor
    processed["low"] = raw_low * factor
    processed["close"] = raw_adj_close.where(raw_adj_close.notna(), raw_close * factor)
    processed["volume"] = raw_volume
    processed["value"] = processed["close"] * processed["volume"]
    processed["change"] = processed["close"].pct_change(fill_method=None).fillna(0.0)
    processed["transactions"] = np.nan
    processed["vwap"] = raw_vwap * factor
    processed["factor"] = factor

    processed.dropna(subset=["close"], inplace=True)

    columns = [
        "symbol",
        "source_symbol",
        "date",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_adj_close",
        "raw_volume",
        "raw_value",
        "raw_change",
        "raw_vwap",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "value",
        "change",
        "transactions",
        "vwap",
        "factor",
    ]
    return processed[columns]


def build_process_data(
    *,
    raw_dir: Path,
    process_dir: Path,
    source_files: Optional[Sequence[Path]] = None,
) -> List[Path]:
    raw_path = Path(raw_dir).expanduser().resolve()
    process_path = Path(process_dir).expanduser().resolve()
    raw_path.mkdir(parents=True, exist_ok=True)
    process_path.mkdir(parents=True, exist_ok=True)

    csv_files = [Path(path).expanduser().resolve() for path in source_files] if source_files else sorted(raw_path.glob("*.csv"))
    if not csv_files:
        LOG.error("No raw CSV files found under %s", raw_path)
        return []

    saved: List[Path] = []
    for csv_path in csv_files:
        df = pd.read_csv(csv_path, parse_dates=["date"], dtype=SYMBOL_CSV_DTYPES)
        if df.empty:
            LOG.warning("Skip empty raw snapshot: %s", csv_path)
            continue
        symbol = str(df["symbol"].iloc[0]).upper() if "symbol" in df.columns and not df["symbol"].empty else fname_to_code(csv_path.stem).upper()
        try:
            processed_df = build_processed_dataframe(symbol, df)
        except ValueError as err:
            LOG.error("Failed to normalize raw snapshot %s: %s", csv_path, err)
            continue
        if processed_df.empty:
            LOG.warning("Processed snapshot is empty after normalization: %s", csv_path)
            continue
        out_path = process_path / f"{code_to_fname(symbol).lower()}.csv"
        processed_df.to_csv(out_path, index=False)
        saved.append(out_path)
    LOG.info("Built %d processed CSV snapshots under %s", len(saved), process_path)
    return saved


# ---------------------------------------------------------------------------
# TWSE Open API workflow
# ---------------------------------------------------------------------------


def roc_to_datetime(date_str: str) -> datetime:
    date_str = str(date_str).strip()
    if not date_str:
        raise ValueError("empty date string")
    year = int(date_str[:3]) + 1911
    month = int(date_str[3:5])
    day = int(date_str[5:])
    return datetime(year, month, day)


class TWSEOpenAPICollector:
    def __init__(
        self,
        output_dir: Path,
        endpoints: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.endpoints = endpoints if endpoints is not None else TWSE_OPENAPI_ENDPOINTS
        self.headers = headers or {"accept": "application/json"}

    def _fetch(self, name: str, url: str) -> Optional[Tuple[List[Dict[str, str]], str]]:
        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as err:
            LOG.error("OpenAPI request failed (%s): %s", name, err)
            return None
        try:
            data = resp.json()
        except ValueError as err:
            LOG.error("OpenAPI json decode failed (%s): %s | snippet=%r", name, err, resp.text[:120])
            return None
        return data, resp.text

    def _normalize(self, data: List[Dict[str, str]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        if "Date" in df.columns:
            def _to_iso(value: str) -> str:
                value = str(value).strip()
                if not value:
                    return ""
                try:
                    if len(value) == 7 and value.isdigit():
                        return roc_to_datetime(value).strftime("%Y-%m-%d")
                    return value
                except Exception:
                    return ""
            df["date_iso"] = df["Date"].apply(_to_iso)
        return df

    def run(self) -> None:
        json_dir = self.output_dir / "json"
        csv_dir = self.output_dir / "csv"
        json_dir.mkdir(parents=True, exist_ok=True)
        csv_dir.mkdir(parents=True, exist_ok=True)
        for name, url in self.endpoints.items():
            LOG.info("Fetching OpenAPI dataset %s", name)
            result = self._fetch(name, url)
            if result is None:
                continue
            data, raw_text = result
            df = self._normalize(data)
            (json_dir / f"{name}.json").write_text(raw_text, encoding="utf-8")
            df.to_csv(csv_dir / f"{name}.csv", index=False, encoding="utf-8")
            LOG.info("Saved %s rows for %s", len(df), name)


# ---------------------------------------------------------------------------
# Qlib dumper
# ---------------------------------------------------------------------------


class QlibDumper:
    CAL_DIR = "calendars"
    FEA_DIR = "features"
    INST_DIR = "instruments"

    def __init__(
        self,
        source_files: Sequence[Path],
        output_dir: Path,
        date_field: str = "date",
        symbol_field: str = "symbol",
        numeric_fields: Optional[Sequence[str]] = None,
    ):
        self.source_files = [Path(p) for p in source_files]
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.date_field = date_field
        self.symbol_field = symbol_field
        self.numeric_fields = list(numeric_fields) if numeric_fields else [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value",
            "change",
            "transactions",
            "vwap",
            "factor",
        ]
        self.calendar: List[pd.Timestamp] = []
        self.instrument_rows: List[str] = []
        self.required_process_fields = {
            self.symbol_field,
            self.date_field,
            "source_symbol",
            "raw_open",
            "raw_high",
            "raw_low",
            "raw_close",
            "raw_adj_close",
            "raw_volume",
            "raw_value",
            "raw_change",
            "raw_vwap",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "value",
            "change",
            "transactions",
            "vwap",
            "factor",
        }

    def run(self) -> int:
        if not self.source_files:
            LOG.error("No source CSV files produced; aborting dump.")
            return 0
        all_data = self._load_all_data()
        if not all_data:
            LOG.error("No parsed data rows; aborting dump.")
            return 0
        self.calendar = sorted({row[self.date_field] for rows in all_data.values() for row in rows})
        if not self.calendar:
            LOG.error("Calendar is empty; aborting dump.")
            return 0
        self._dump_calendar()
        dumped_symbols = 0
        for symbol, rows in all_data.items():
            dumped_symbols += int(self._dump_symbol(symbol, rows))
        self._dump_instruments()
        self._dump_metadata()
        LOG.info("Dumped data into %s", self.output_dir)
        return dumped_symbols

    def _load_all_data(self) -> Dict[str, List[Dict[str, float]]]:
        data: Dict[str, List[Dict[str, float]]] = {}
        for csv_path in self.source_files:
            df = pd.read_csv(csv_path, parse_dates=[self.date_field], dtype=SYMBOL_CSV_DTYPES)
            if df.empty:
                continue
            symbol = fname_to_code(csv_path.stem).upper()
            data[symbol] = self._to_qlib_frame(df).to_dict(orient="records")
        return data

    def _to_qlib_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        qlib_df = df.copy()
        qlib_df[self.date_field] = pd.to_datetime(qlib_df[self.date_field])
        missing_fields = sorted(self.required_process_fields - set(qlib_df.columns))
        if missing_fields:
            raise ValueError(
                "Process_data schema mismatch; dump only accepts standard Process_data CSVs. "
                f"Missing fields: {missing_fields}"
            )

        for field in self.required_process_fields - {self.symbol_field, self.date_field, "source_symbol"}:
            qlib_df[field] = pd.to_numeric(qlib_df[field], errors="coerce")
        keep_cols = [self.symbol_field, self.date_field, *[field for field in self.numeric_fields if field in qlib_df.columns]]
        return qlib_df[keep_cols]

    def _dump_calendar(self) -> None:
        cal_dir = self.output_dir / self.CAL_DIR
        cal_dir.mkdir(parents=True, exist_ok=True)
        day_file = cal_dir / "day.txt"
        with day_file.open("w", encoding="utf-8") as fp:
            for dt in self.calendar:
                fp.write(pd.Timestamp(dt).strftime("%Y-%m-%d") + "\n")

    def _dump_symbol(self, symbol: str, rows: List[Dict[str, float]]) -> bool:
        df = pd.DataFrame(rows)
        df[self.date_field] = pd.to_datetime(df[self.date_field])
        df.set_index(self.date_field, inplace=True)
        aligned, sub_calendar = self._align_to_calendar(df)
        if not sub_calendar or aligned.empty:
            LOG.warning("Symbol %s has no calendar overlap; skipped.", symbol)
            return False
        valid_dates = aligned.dropna(subset=["close"]).index
        if valid_dates.empty:
            LOG.warning("Symbol %s has no price records; skipped.", symbol)
            return False
        start = valid_dates.min().strftime("%Y-%m-%d")
        end = valid_dates.max().strftime("%Y-%m-%d")
        self.instrument_rows.append(f"{symbol}\t{start}\t{end}")

        feature_dir = self.output_dir / self.FEA_DIR / code_to_fname(symbol).lower()
        feature_dir.mkdir(parents=True, exist_ok=True)
        start_index = self.calendar.index(sub_calendar[0])
        for field in self.numeric_fields:
            if field not in aligned.columns:
                continue
            values = aligned[field].to_numpy(dtype=np.float32, copy=True)
            if np.isnan(values).all():
                continue
            payload = np.concatenate([np.array([start_index], dtype=np.float32), values])
            bin_path = feature_dir / f"{field.lower()}.day.bin"
            payload.astype(np.float32).tofile(bin_path)
        return True

    def _align_to_calendar(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
        start = df.index.min()
        end = df.index.max()
        sub_calendar = [dt for dt in self.calendar if start <= dt <= end]
        if not sub_calendar:
            return pd.DataFrame(), []
        aligned = df.reindex(pd.Index(sub_calendar, name=self.date_field))
        return aligned, sub_calendar

    def _dump_instruments(self) -> None:
        inst_dir = self.output_dir / self.INST_DIR
        inst_dir.mkdir(parents=True, exist_ok=True)
        inst_path = inst_dir / "all.txt"
        with inst_path.open("w", encoding="utf-8") as fp:
            for row in sorted(self.instrument_rows):
                fp.write(row + "\n")

    def _dump_metadata(self) -> None:
        meta_path = self.output_dir / "price_semantics.json"
        payload = {
            "schema": "qlib_tw_price_semantics/v1",
            "price_basis": "adjusted",
            "factor_semantics": "price_only",
            "fields": {
                "open": "adjusted",
                "high": "adjusted",
                "low": "adjusted",
                "close": "adjusted",
                "vwap": "adjusted",
                "volume": "raw_shares",
                "value": "adjusted_price_times_raw_volume",
                "change": "adjusted_close_pct_change",
                "factor": "adjusted_over_raw",
            },
        }
        meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Download TWSE/Yahoo data and dump to Qlib format")
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="Fetch daily bars from Yahoo Finance")
    collect_parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    collect_parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    collect_parser.add_argument("--target-dir", type=Path, default=None, help="Qlib provider output directory (default: Data/qlib_data)")
    collect_parser.add_argument("--raw-dir", type=Path, default=None, help="Raw download snapshot directory (default: Data/Raw_data)")
    collect_parser.add_argument("--process-dir", type=Path, default=None, help="Processed Qlib-ready CSV directory (default: Data/Process_data)")
    collect_parser.add_argument(
        "--symbols",
        type=str,
        nargs="*",
        help="Optional list of codes (e.g. 2330 0050 or 2330,2317). Defaults to entire TWSE if omitted.",
    )
    collect_parser.add_argument(
        "--suffix",
        type=str,
        default=".TW",
        help="Default Yahoo suffix for numeric codes (default: .TW)",
    )
    collect_parser.add_argument("--pause", type=float, default=0.5, help="Sleep seconds between Yahoo requests")

    process_parser = subparsers.add_parser("process", help="Build Process_data from an existing raw snapshot")
    process_parser.add_argument("--raw-dir", type=Path, default=None, help="Raw download snapshot directory (default: Data/Raw_data)")
    process_parser.add_argument("--process-dir", type=Path, default=None, help="Processed Qlib-ready CSV directory (default: Data/Process_data)")

    dump_parser = subparsers.add_parser("dump", help="Build a Qlib provider from an existing processed snapshot")
    dump_parser.add_argument("--target-dir", type=Path, default=None, help="Qlib provider output directory (default: Data/qlib_data)")
    dump_parser.add_argument("--process-dir", type=Path, default=None, help="Processed Qlib-ready CSV directory (default: Data/Process_data)")

    openapi_parser = subparsers.add_parser("openapi", help="Fetch TWSE OpenAPI datasets")
    openapi_parser.add_argument("--target-dir", type=Path, required=True, help="Output directory for raw JSON/CSV")
    openapi_parser.add_argument(
        "--endpoints",
        type=str,
        nargs="*",
        choices=sorted(TWSE_OPENAPI_ENDPOINTS.keys()),
        help="Subset of datasets to download (default: all)",
    )

    return parser, parser.parse_args(argv)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s"))
    LOG.setLevel(logging.INFO)
    LOG.addHandler(handler)


def _normalize_symbols(symbols: Optional[Sequence[str]]) -> List[str]:
    if not symbols:
        return []
    raw_symbols: List[str] = []
    for token in symbols:
        if token is None:
            continue
        raw_symbols.extend(part.strip() for part in str(token).split(",") if part.strip())
    seen: set[str] = set()
    unique: List[str] = []
    for token in raw_symbols:
        key = token.upper()
        if key in seen:
            continue
        seen.add(key)
        unique.append(token)
    return unique


def run_collect(
    *,
    start: str,
    end: str,
    target_dir: Path,
    raw_dir: Optional[Path] = None,
    process_dir: Optional[Path] = None,
    symbols: Optional[Sequence[str]] = None,
    suffix: str = ".TW",
    pause: float = 0.5,
) -> int:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        LOG.error("Start date is after end date (%s > %s)", start_ts.date(), end_ts.date())
        return 1

    target_path = Path(target_dir).expanduser().resolve()
    tmp_path = Path(raw_dir).expanduser().resolve() if raw_dir else RAW_DATA_DIR.resolve()
    process_path = Path(process_dir).expanduser().resolve() if process_dir else PROCESS_DATA_DIR.resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    process_path.mkdir(parents=True, exist_ok=True)

    requested_symbols = _normalize_symbols(symbols)
    if requested_symbols:
        LOG.info("Using %d user-specified symbols", len(requested_symbols))
    else:
        requested_symbols = YahooTaiwanCollector.fetch_default_symbols()
        LOG.info("Fetched %d symbols from TWSE OpenAPI", len(requested_symbols))

    if not requested_symbols:
        LOG.error("No symbols specified or fetched; aborting")
        return 1

    seen = {code.upper() for code in requested_symbols}
    for bench in DEFAULT_BENCHMARKS:
        if bench.upper() not in seen:
            requested_symbols.append(bench)
            seen.add(bench.upper())
            LOG.info("Appended benchmark symbol %s", bench)

    collector = YahooTaiwanCollector(
        symbols=requested_symbols,
        start=start_ts,
        end=end_ts,
        raw_dir=tmp_path,
        pause=pause,
        default_suffix=suffix,
    )
    raw_files = collector.run()

    skipped_log = target_path.joinpath("yahoo_skipped.log")
    if collector.failed_symbols:
        skipped_log.parent.mkdir(parents=True, exist_ok=True)
        with skipped_log.open("w", encoding="utf-8") as fp:
            for symbol, (reason, details) in sorted(collector.failed_symbols.items()):
                fp.write(f"{symbol}\t{reason}\t{(details or '')}\n")
        LOG.warning(
            "Logged %s skipped symbols to %s",
            len(collector.failed_symbols),
            skipped_log,
        )
    elif skipped_log.exists():
        skipped_log.unlink()

    if not raw_files:
        LOG.error("No raw data generated because no valid Yahoo CSV files were available.")
        return 1

    processed_files = build_process_data(raw_dir=tmp_path, process_dir=process_path, source_files=raw_files)
    if not processed_files:
        LOG.error("No processed data generated from the downloaded raw snapshots.")
        return 1

    return run_dump(target_dir=target_path, process_dir=process_path)


def run_process(
    *,
    raw_dir: Optional[Path] = None,
    process_dir: Optional[Path] = None,
    source_files: Optional[Sequence[Path]] = None,
) -> int:
    raw_path = Path(raw_dir).expanduser().resolve() if raw_dir else RAW_DATA_DIR.resolve()
    process_path = Path(process_dir).expanduser().resolve() if process_dir else PROCESS_DATA_DIR.resolve()
    processed_files = build_process_data(raw_dir=raw_path, process_dir=process_path, source_files=source_files)
    if not processed_files:
        return 1
    return 0


def run_dump(
    *,
    target_dir: Path,
    process_dir: Optional[Path] = None,
    source_files: Optional[Sequence[Path]] = None,
) -> int:
    target_path = Path(target_dir).expanduser().resolve()
    process_path = Path(process_dir).expanduser().resolve() if process_dir else PROCESS_DATA_DIR.resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    process_path.mkdir(parents=True, exist_ok=True)

    csv_files = [Path(path).expanduser().resolve() for path in source_files] if source_files else sorted(process_path.glob("*.csv"))
    if not csv_files:
        LOG.error("No processed CSV files found under %s", process_path)
        return 1

    dumper = QlibDumper(source_files=csv_files, output_dir=target_path)
    try:
        dumped_symbols = dumper.run()
    except ValueError as err:
        LOG.error("Failed to dump qlib_data: %s", err)
        return 1
    if dumped_symbols <= 0:
        LOG.error("Qlib dump produced no valid symbols")
        return 1
    LOG.info("Done. Data available at %s", target_path)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    configure_logging()

    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        LOG.info("No CLI arguments detected. Running default collect configuration.")
        return run_collect(**DEFAULT_COLLECT_CONFIG)

    parser, args = parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 2

    if args.command == "collect":
        config = DEFAULT_COLLECT_CONFIG.copy()
        if args.start:
            config["start"] = args.start
        if args.end:
            config["end"] = args.end
        if args.target_dir:
            config["target_dir"] = args.target_dir
        if args.raw_dir:
            config["raw_dir"] = args.raw_dir
        if args.process_dir:
            config["process_dir"] = args.process_dir
        return run_collect(
            start=config["start"],
            end=config["end"],
            target_dir=config["target_dir"],
            raw_dir=config.get("raw_dir"),
            process_dir=config.get("process_dir"),
            symbols=args.symbols or config["symbols"],
            suffix=args.suffix,
            pause=args.pause,
        )

    if args.command == "process":
        return run_process(
            raw_dir=args.raw_dir,
            process_dir=args.process_dir,
        )

    if args.command == "dump":
        return run_dump(
            target_dir=args.target_dir or DEFAULT_COLLECT_CONFIG["target_dir"],
            process_dir=args.process_dir,
        )

    if args.command == "openapi":
        target_dir = args.target_dir or DEFAULT_COLLECT_CONFIG["target_dir"]
        target_dir = Path(target_dir).expanduser().resolve()
        endpoints = (
            {name: TWSE_OPENAPI_ENDPOINTS[name] for name in args.endpoints}
            if args.endpoints
            else TWSE_OPENAPI_ENDPOINTS
        )
        collector = TWSEOpenAPICollector(output_dir=target_dir, endpoints=endpoints)
        collector.run()
        LOG.info("OpenAPI datasets stored at %s", target_dir)
        return 0

    LOG.error("Unsupported command: %s", args.command)
    return 2


if __name__ == "__main__":
    sys.exit(main())
