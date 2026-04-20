from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


WORK_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = WORK_DIR / "Data"

RAW_DATA_DIR = DATA_DIR / "Raw_data"
PROCESS_DATA_DIR = DATA_DIR / "Process_data"
QLIB_DATA_DIR = DATA_DIR / "qlib_data"

PRICE_SEMANTICS_FILE = "price_semantics.json"
PRICE_BASIS_ADJUSTED = "adjusted"
FACTOR_SEMANTICS_PRICE_ONLY = "price_only"


def _resolve_local_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = WORK_DIR / resolved
    return resolved.expanduser().resolve()


def resolve_provider_uri(path: str | Path | None = None) -> Path:
    resolved = _resolve_local_path(path)
    if resolved is not None:
        return resolved
    return QLIB_DATA_DIR.resolve()


def resolve_raw_data_dir(path: str | Path | None = None) -> Path:
    resolved = _resolve_local_path(path)
    if resolved is not None:
        return resolved
    return RAW_DATA_DIR.resolve()


def resolve_process_data_dir(path: str | Path | None = None) -> Path:
    resolved = _resolve_local_path(path)
    if resolved is not None:
        return resolved
    return PROCESS_DATA_DIR.resolve()


def active_provider_uri_from_qlib() -> Path | None:
    try:
        from qlib.config import C
    except ImportError:
        return None
    provider_uri = getattr(C, "provider_uri", None) or C.get("provider_uri", {})
    if not provider_uri:
        return None
    if isinstance(provider_uri, dict):
        if "day" in provider_uri:
            return Path(provider_uri["day"]).expanduser().resolve()
        if "__DEFAULT_FREQ" in provider_uri:
            return Path(provider_uri["__DEFAULT_FREQ"]).expanduser().resolve()
        first = next(iter(provider_uri.values()), None)
        if first:
            return Path(first).expanduser().resolve()
        return None
    return Path(provider_uri).expanduser().resolve()


def load_price_semantics(provider_uri: str | Path | None = None) -> Dict[str, Any]:
    resolved = resolve_provider_uri(provider_uri)
    meta_path = resolved / PRICE_SEMANTICS_FILE
    if meta_path.exists():
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass
    return {
        "price_basis": PRICE_BASIS_ADJUSTED,
        "factor_semantics": FACTOR_SEMANTICS_PRICE_ONLY,
    }


def provider_prices_are_adjusted(provider_uri: str | Path | None = None) -> bool:
    semantics = load_price_semantics(provider_uri)
    return str(semantics.get("price_basis", "")).strip().lower() == PRICE_BASIS_ADJUSTED


def provider_factor_is_price_only(provider_uri: str | Path | None = None) -> bool:
    semantics = load_price_semantics(provider_uri)
    return str(semantics.get("factor_semantics", "")).strip().lower() == FACTOR_SEMANTICS_PRICE_ONLY
