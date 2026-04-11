from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PendingCashEntry:
    release_date: str
    amount: float


@dataclass(frozen=True)
class PendingStockEntry:
    instrument: str
    release_date: str
    amount: float
    price: float


@dataclass(frozen=True)
class PositionEntry:
    instrument: str
    quantity: float
    sellable_quantity: float
    locked_quantity: float
    price: float
    weight: float
    count_day: int


@dataclass(frozen=True)
class PaperStateSnapshot:
    as_of_date: str
    next_trade_date: str
    account_value: float
    market_value: float
    cash: float
    cash_delay: float
    positions: list[PositionEntry]
    pending_cash: list[PendingCashEntry]
    pending_stock: list[PendingStockEntry]
    metadata: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))


def _locked_stock_map(position) -> Dict[str, float]:
    locked: Dict[str, float] = {}
    for code, entries in getattr(position, "_pending_stock", {}).items():
        locked[str(code)] = sum(_as_float(amount) for _release, amount, _price in entries)
    return locked


def _iter_pending_cash(position) -> Iterable[PendingCashEntry]:
    for release_date, amount in getattr(position, "_pending_cash", []):
        yield PendingCashEntry(
            release_date=pd.Timestamp(release_date).strftime("%Y-%m-%d"),
            amount=_as_float(amount),
        )


def _iter_pending_stock(position) -> Iterable[PendingStockEntry]:
    for code, entries in getattr(position, "_pending_stock", {}).items():
        for release_date, amount, price in entries:
            yield PendingStockEntry(
                instrument=str(code),
                release_date=pd.Timestamp(release_date).strftime("%Y-%m-%d"),
                amount=_as_float(amount),
                price=_as_float(price),
            )


def snapshot_from_position(
    *,
    position,
    as_of_date: pd.Timestamp,
    next_trade_date: pd.Timestamp,
    account_value: float,
    market_value: float,
    metadata: Dict[str, object],
) -> PaperStateSnapshot:
    locked_map = _locked_stock_map(position)
    position_rows: list[PositionEntry] = []
    for code in sorted(position.get_stock_list()):
        stock_info = position.position.get(code, {})
        quantity = _as_float(stock_info.get("amount", position.get_stock_amount(code)))
        locked_quantity = locked_map.get(str(code), 0.0)
        position_rows.append(
            PositionEntry(
                instrument=str(code),
                quantity=quantity,
                sellable_quantity=max(quantity - locked_quantity, 0.0),
                locked_quantity=locked_quantity,
                price=_as_float(stock_info.get("price")),
                weight=_as_float(stock_info.get("weight")),
                count_day=int(stock_info.get("count_day", 0)),
            )
        )
    return PaperStateSnapshot(
        as_of_date=as_of_date.strftime("%Y-%m-%d"),
        next_trade_date=next_trade_date.strftime("%Y-%m-%d"),
        account_value=_as_float(account_value),
        market_value=_as_float(market_value),
        cash=_as_float(position.get_cash()),
        cash_delay=_as_float(position.position.get("cash_delay")),
        positions=position_rows,
        pending_cash=list(_iter_pending_cash(position)),
        pending_stock=list(_iter_pending_stock(position)),
        metadata=metadata,
    )

