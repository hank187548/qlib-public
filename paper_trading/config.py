from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

from tw_workflow.search_results import extract_model_kwargs, load_search_result_row
from tw_workflow.settings import COMBO_CONFIGS, MODEL_CONFIGS, PROVIDER_URI, REGION, WORK_DIR


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = WORK_DIR / path
    return path.resolve()


@dataclass(frozen=True)
class PaperTradingProfile:
    name: str
    combo: str
    train_experiment: str
    train_recorder_id: str
    strategy: str = "bucket"
    topk: int = 10
    n_drop: int = 1
    rebalance: str = "day"
    deal_price: str = "close"
    limit_tplus: bool = True
    limit_slippage: float = 0.01
    settlement_lag: int = 2
    risk_degree: float = 0.95
    account: float = 1_000_000.0
    open_cost: float = 0.00092625
    close_cost: float = 0.00392625
    min_cost: float = 20.0
    odd_lot_min_cost: float = 1.0
    board_lot_size: int = 1000
    trade_unit: int = 1
    backtest_start: str = "2025-07-01"
    data_start: str = "2018-01-01"
    data_refresh_start: str = "2018-01-01"
    provider_uri: Path = field(default_factory=lambda: PROVIDER_URI.resolve())
    output_root: Path = field(default_factory=lambda: WORK_DIR / "outputs" / "paper_trading")
    search_results_csv: Path | None = None
    search_run_index: int | None = None
    model_kwargs: Dict[str, object] = field(default_factory=dict)
    reference_backtest_experiment: str | None = None
    reference_backtest_recorder_id: str | None = None
    region: str = REGION

    @classmethod
    def from_json(cls, path: str | Path) -> "PaperTradingProfile":
        config_path = _resolve_path(path)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(f"Paper trading config not found: {path}")
        payload = json.loads(config_path.read_text())
        provider_uri = _resolve_path(payload.get("provider_uri")) or PROVIDER_URI.resolve()
        output_root = _resolve_path(payload.get("output_root")) or (WORK_DIR / "outputs" / "paper_trading")
        search_results_csv = _resolve_path(payload.get("search_results_csv"))
        profile = cls(
            name=str(payload["name"]),
            combo=str(payload["combo"]),
            train_experiment=str(payload["train_experiment"]),
            train_recorder_id=str(payload["train_recorder_id"]),
            strategy=str(payload.get("strategy", "bucket")),
            topk=int(payload.get("topk", 10)),
            n_drop=int(payload.get("n_drop", 1)),
            rebalance=str(payload.get("rebalance", "day")),
            deal_price=str(payload.get("deal_price", "close")),
            limit_tplus=bool(payload.get("limit_tplus", True)),
            limit_slippage=float(payload.get("limit_slippage", 0.01)),
            settlement_lag=int(payload.get("settlement_lag", 2)),
            risk_degree=float(payload.get("risk_degree", 0.95)),
            account=float(payload.get("account", 1_000_000.0)),
            open_cost=float(payload.get("open_cost", 0.00092625)),
            close_cost=float(payload.get("close_cost", 0.00392625)),
            min_cost=float(payload.get("min_cost", 20.0)),
            odd_lot_min_cost=float(payload.get("odd_lot_min_cost", 1.0)),
            board_lot_size=int(payload.get("board_lot_size", 1000)),
            trade_unit=int(payload.get("trade_unit", 1)),
            backtest_start=str(payload.get("backtest_start", "2025-07-01")),
            data_start=str(payload.get("data_start", "2018-01-01")),
            data_refresh_start=str(payload.get("data_refresh_start", payload.get("data_start", "2018-01-01"))),
            provider_uri=provider_uri.resolve(),
            output_root=output_root,
            search_results_csv=search_results_csv,
            search_run_index=int(payload["search_run_index"]) if payload.get("search_run_index") is not None else None,
            model_kwargs=dict(payload.get("model_kwargs", {})),
            reference_backtest_experiment=payload.get("reference_backtest_experiment"),
            reference_backtest_recorder_id=payload.get("reference_backtest_recorder_id"),
            region=str(payload.get("region", REGION)),
        )
        if profile.combo not in COMBO_CONFIGS:
            raise ValueError(f"Unsupported combo in paper trading config: {profile.combo}")
        return profile

    @property
    def combo_spec(self) -> Dict[str, object]:
        return COMBO_CONFIGS[self.combo]

    @property
    def resolved_provider_uri(self) -> Path:
        return self.provider_uri.resolve()

    @property
    def resolved_output_root(self) -> Path:
        return (self.output_root / self.name).resolve()

    @property
    def resolved_raw_dir(self) -> Path:
        return (self.resolved_provider_uri.parent / "_raw_yahoo").resolve()

    @property
    def effective_deal_price(self) -> str:
        if self.limit_tplus:
            return "open"
        return self.deal_price

    def resolved_model_kwargs(self) -> Dict[str, object]:
        if self.model_kwargs:
            return dict(self.model_kwargs)
        if self.search_results_csv is None or self.search_run_index is None:
            return {}
        allowed_keys = MODEL_CONFIGS[self.combo_spec["model"]]["kwargs"].keys()
        row = load_search_result_row(self.search_results_csv, self.search_run_index)
        return extract_model_kwargs(row, allowed_keys)

    def to_metadata(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["provider_uri"] = str(self.resolved_provider_uri)
        payload["output_root"] = str(self.resolved_output_root)
        payload["raw_dir"] = str(self.resolved_raw_dir)
        payload["effective_deal_price"] = self.effective_deal_price
        if self.search_results_csv is not None:
            payload["search_results_csv"] = str(self.search_results_csv)
        return payload
