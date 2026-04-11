#!/usr/bin/env python3
"""
Download Taiwan stock data from Yahoo Finance and TWSE open APIs, then
transform them into Qlib-compatible format.

Two workflows are supported:
- `collect`: Use Yahoo Finance to fetch historical daily bars for Taiwan
  tickers, convert them into Qlib binary storage (`calendars/`, `features/`,
  `instruments/`).
- `openapi`: Fetch the latest TWSE open-data datasets (e.g., BWIBBU_ALL) and
  store the JSON/CSV snapshots locally.

The script avoids touching the original Qlib package so you can extend it
freely.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

LOG = logging.getLogger("GetDataTai")

DEFAULT_COLLECT_CONFIG = {
    # Update these two dates to control the Yahoo download window.
    "start": "2009-01-01",
    "end": "2025-11-03",
    # Update these paths if you want to store the dataset elsewhere.
    "target_dir": Path("Data/tw_data"),
    "tmp_dir": Path("Data/_raw_yahoo"),
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

    def _transform(self, base_symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df["symbol"] = base_symbol.upper()
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.dropna(subset=["close"], inplace=True)
        df["value"] = df["close"] * df["volume"].fillna(0)
        df["change"] = df["close"].diff().fillna(0)
        df["transactions"] = np.nan
        df["vwap"] = df["value"] / df["volume"].where(df["volume"] != 0, np.nan)
        if "adj_close" in df.columns:
            adj_factor = df["adj_close"] / df["close"].replace(0, np.nan)
            df["factor"] = adj_factor.fillna(1.0)
        else:
            df["factor"] = 1.0
        columns = [
            "symbol",
            "date",
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
        return df[columns]

    def save_dataframe(self, base_symbol: str, df: pd.DataFrame) -> Optional[Path]:
        if df.empty:
            return None
        csv_path = self.raw_dir / f"{code_to_fname(base_symbol).lower()}.csv"
        df.to_csv(csv_path, index=False)
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
            tf = self._transform(base_symbol, df)
            path = self.save_dataframe(base_symbol, tf)
            if path:
                saved.append(path)
            time.sleep(self.pause)
        LOG.info("Finished Yahoo download for %d symbols", len(saved))
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
# Qlib dumper (unchanged)
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

    def run(self) -> None:
        if not self.source_files:
            LOG.error("No source CSV files produced; aborting dump.")
            return
        all_data = self._load_all_data()
        if not all_data:
            LOG.error("No parsed data rows; aborting dump.")
            return
        self.calendar = sorted({row[self.date_field] for rows in all_data.values() for row in rows})
        if not self.calendar:
            LOG.error("Calendar is empty; aborting dump.")
            return
        self._dump_calendar()
        for symbol, rows in all_data.items():
            self._dump_symbol(symbol, rows)
        self._dump_instruments()
        LOG.info("Dumped data into %s", self.output_dir)

    def _load_all_data(self) -> Dict[str, List[Dict[str, float]]]:
        data: Dict[str, List[Dict[str, float]]] = {}
        for csv_path in self.source_files:
            df = pd.read_csv(csv_path, parse_dates=[self.date_field])
            if df.empty:
                continue
            symbol = fname_to_code(csv_path.stem).upper()
            data[symbol] = df.to_dict(orient="records")
        return data

    def _dump_calendar(self) -> None:
        cal_dir = self.output_dir / self.CAL_DIR
        cal_dir.mkdir(parents=True, exist_ok=True)
        day_file = cal_dir / "day.txt"
        with day_file.open("w", encoding="utf-8") as fp:
            for dt in self.calendar:
                fp.write(pd.Timestamp(dt).strftime("%Y-%m-%d") + "\n")

    def _dump_symbol(self, symbol: str, rows: List[Dict[str, float]]) -> None:
        df = pd.DataFrame(rows)
        df[self.date_field] = pd.to_datetime(df[self.date_field])
        df.set_index(self.date_field, inplace=True)
        aligned, sub_calendar = self._align_to_calendar(df)
        if not sub_calendar or aligned.empty:
            LOG.warning("Symbol %s has no calendar overlap; skipped.", symbol)
            return
        valid_dates = aligned.dropna(subset=["close"]).index
        if valid_dates.empty:
            LOG.warning("Symbol %s has no price records; skipped.", symbol)
            return
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Download TWSE/Yahoo data and dump to Qlib format")
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser("collect", help="Fetch daily bars from Yahoo Finance")
    collect_parser.add_argument("--start", type=str, default=None, help="Start date (YYYY-MM-DD)")
    collect_parser.add_argument("--end", type=str, default=None, help="End date (YYYY-MM-DD)")
    collect_parser.add_argument("--target-dir", type=Path, default=None, help="Destination directory")
    collect_parser.add_argument(
        "--tmp-dir",
        type=Path,
        default=None,
        help="Intermediate CSV directory (default: <target_parent>/_raw_yahoo)",
    )
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
    tmp_dir: Optional[Path] = None,
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
    tmp_path = Path(tmp_dir).expanduser().resolve() if tmp_dir else target_path.parent.joinpath("_raw_yahoo")
    target_path.mkdir(parents=True, exist_ok=True)
    tmp_path.mkdir(parents=True, exist_ok=True)

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
    csv_files = collector.run()
    saved_set = {p.resolve() for p in csv_files}
    for csv_path in tmp_path.glob("*.csv"):
        if csv_path.resolve() not in saved_set and csv_path.is_file():
            try:
                csv_path.unlink()
            except FileNotFoundError:
                pass

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

    if csv_files:
        dumper = QlibDumper(source_files=csv_files, output_dir=target_path)
        dumper.run()
        LOG.info("Done. Data available at %s", target_path)
    else:
        LOG.warning("No Qlib data generated because no valid CSV files were available.")
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
        if args.tmp_dir:
            config["tmp_dir"] = args.tmp_dir
        return run_collect(
            start=config["start"],
            end=config["end"],
            target_dir=config["target_dir"],
            tmp_dir=config["tmp_dir"],
            symbols=args.symbols or config["symbols"],
            suffix=args.suffix,
            pause=args.pause,
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
