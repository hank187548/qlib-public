#!/usr/bin/env python3
"""
Generate latest TopK signals from a trained Qlib model and prepare an order list (optional live submission).

By default use alpha360_lgb with bucket weights, and load the latest training experiment
`tw_train_model_<combo>` to fetch the latest trained_model. If recorder-id is not provided,
the newest recorder in that experiment is used automatically.

By default only prints/exports CSV without sending orders. Use --place-orders for live submission and confirm
Masterlink credentials/certificate are provided via environment variables or CLI args.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import sys

import qlib
from qlib.data import D
from qlib.utils import init_instance_by_config
from qlib.workflow import R

# Ensure local scripts package is importable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from scripts.workflow_by_code_tw import (  # noqa: E402
    COMBO_CONFIGS,
    UNIVERSE,
    WORK_DIR,
    PROVIDER_URI,
    build_task_config,
)
from masterlink_sdk import (
    MasterlinkSDK,
    Order,
    PriceType,
    BSAction,
    MarketType,
    TimeInForce,
    OrderType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    def _env_default(name: str) -> str | None:
        value = os.environ.get(name)
        return value.strip() if value else None

    parser = argparse.ArgumentParser(description="Run inference and prepare/submit orders for TW market")
    parser.add_argument(
        "--combo",
        default="alpha360_lgb",
        choices=sorted(COMBO_CONFIGS.keys()),
        help="Handler/model combo, default alpha360_lgb",
    )
    parser.add_argument("--recorder-id", default=None, help="Specify recorder id from training experiment; use latest when omitted")
    parser.add_argument("--topk", type=int, default=50, help="Number of selected symbols")
    parser.add_argument(
        "--strategy",
        choices=["bucket", "equal"],
        default="bucket",
        help="bucket=4/2/1 bucketed weights, equal=equal weights",
    )
    parser.add_argument("--budget", type=float, default=1_000_000, help="Total order budget (used for share quantity estimation)")
    parser.add_argument(
        "--price-type",
        choices=["limit", "market", "reference"],
        default="limit",
        help="Order price type; limit uses close price as limit reference",
    )
    parser.add_argument(
        "--limit-slippage",
        type=float,
        default=0.0,
        help="Limit price offset relative to close price (e.g. 0.01 means +1%%)",
    )
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=None,
        help="Override price CSV (must include instrument and price); used to recompute limit/quantity after open",
    )
    parser.add_argument(
        "--price-col",
        type=str,
        default="price",
        help="Price column name in price-csv (default price)",
    )
    parser.add_argument(
        "--instrument-col",
        type=str,
        default="instrument",
        help="Instrument column name in price-csv (default instrument)",
    )
    parser.add_argument(
        "--place-orders",
        action="store_true",
        help="Submit orders via Masterlink SDK; otherwise only export order list",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORK_DIR / "outputs" / "live_orders",
        help="Output directory for order list",
    )
    parser.add_argument("--account-index", type=int, default=0, help="Account index when multiple accounts are returned")
    parser.add_argument("--market-type", default="common", choices=["common", "fixing", "odd", "intraday-odd", "emg"])
    parser.add_argument("--time-in-force", default="rod", choices=["rod", "ioc", "fok"])
    parser.add_argument(
        "--user-id",
        default=_env_default("MASTERLINK_ID"),
        help="Login ID (env MASTERLINK_ID)",
    )
    parser.add_argument(
        "--password",
        default=_env_default("MASTERLINK_PASSWORD"),
        help="Trading password (env MASTERLINK_PASSWORD)",
    )
    parser.add_argument(
        "--cert",
        default=_env_default("MASTERLINK_CERT"),
        help="Path to .pfx certificate (env MASTERLINK_CERT)",
    )
    parser.add_argument(
        "--cert-password",
        default=_env_default("MASTERLINK_CERT_PASSWORD"),
        help="Certificate password (env MASTERLINK_CERT_PASSWORD)",
    )
    return parser.parse_args()


def choose_latest_recorder(experiment: str) -> str:
    recs = R.list_recorders(experiment_name=experiment)
    if not recs:
        raise RuntimeError(f"No recorder found for experiment {experiment}; train model first")
    # Sort by start_time and pick latest
    def _ts(rid: str) -> float:
        info = recs[rid].info
        ts_str = info.get("start_time")
        try:
            return datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            return 0.0

    latest = sorted(recs.keys(), key=_ts, reverse=True)[0]
    return latest


def default_bucket_weights(topk: int) -> List[float]:
    # 4/2/1 weighting without strategy instantiation to avoid signal requirement
    weights = [0.04] * 10 + [0.02] * 20 + [0.01] * 20
    if len(weights) < topk:
        tail = topk - len(weights)
        weights += [1.0 / topk] * tail
    return weights[:topk]


def fetch_latest_predictions(model, dataset, topk: int) -> Tuple[pd.Timestamp, pd.Series]:
    pred = model.predict(dataset, segment="test")
    if not isinstance(pred, pd.Series):
        raise RuntimeError("Prediction result is not a pandas Series")
    dates = pred.index.get_level_values(0)
    last_dt = dates.max()
    today_scores = pred.loc[pd.IndexSlice[last_dt, :]].sort_values(ascending=False)
    return last_dt, today_scores.head(topk)


def fetch_close_prices(codes: List[str], dt: pd.Timestamp) -> Dict[str, float]:
    df = D.features(codes, ["$close"], start_time=dt, end_time=dt)
    price_map: Dict[str, float] = {}
    if df.empty:
        return price_map
    for (inst, _dt), row in df.iterrows():
        price = row["$close"]
        if pd.notna(price):
            price_map[str(inst)] = float(price)
    return price_map


def build_orders(
    scores: pd.Series,
    weights: List[float],
    prices: Dict[str, float],
    budget: float,
    price_type: str,
    slippage: float,
) -> List[Dict[str, object]]:
    orders = []
    total_w = sum(weights[: len(scores)])
    for (code, score), w in zip(scores.items(), weights):
        if total_w <= 0:
            break
        close_price = prices.get(code)
        if close_price is None or close_price <= 0:
            continue
        target_value = budget * (w / total_w)
        limit_price = close_price * (1 + slippage) if price_type == "limit" else None
        shares = np.floor(target_value / (limit_price or close_price))
        if shares <= 0:
            continue
        orders.append(
            {
                "instrument": code,
                "score": float(score),
                "weight": w,
                "close": close_price,
                "price": round(limit_price, 3) if limit_price else None,
                "quantity": int(shares),
                "price_type": price_type,
            }
        )
    return orders


def place_orders(orders: List[Dict[str, object]], args: argparse.Namespace) -> None:
    user_id = args.user_id if hasattr(args, "user_id") else None
    password = args.password if hasattr(args, "password") else None
    cert = args.cert if hasattr(args, "cert") else None
    cert_password = args.cert_password if hasattr(args, "cert_password") else None
    if not (user_id and password and cert and cert_password):
        raise RuntimeError(
            "Missing Masterlink login fields. Set MASTERLINK_ID / MASTERLINK_PASSWORD / "
            "MASTERLINK_CERT / MASTERLINK_CERT_PASSWORD, or provide via CLI args."
        )
    cert_path = Path(cert).expanduser().resolve()
    if not cert_path.exists():
        raise RuntimeError(f"Certificate file not found:{cert_path}")

    sdk = MasterlinkSDK()
    accounts = sdk.login(user_id, password, str(cert_path), cert_password)
    if not accounts:
        raise RuntimeError("Login returned no account")
    if args.account_index < 0 or args.account_index >= len(accounts):
        raise RuntimeError(f"account_index {args.account_index} out of range (total {len(accounts)} accounts)")
    account = accounts[args.account_index]

    price_type_map = {
        "limit": PriceType.Limit,
        "market": PriceType.Market,
        "reference": PriceType.Reference,
    }
    mt_map = {
        "common": MarketType.Common,
        "fixing": MarketType.Fixing,
        "odd": MarketType.Odd,
        "intraday-odd": MarketType.IntradayOdd,
        "emg": MarketType.Emg,
    }
    tif_map = {
        "rod": TimeInForce.ROD,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }

    for od in orders:
        order = Order(
            buy_sell=BSAction.Buy,
            symbol=od["instrument"],
            quantity=od["quantity"],
            price=None if od["price"] is None else f"{od['price']}",
            market_type=mt_map[args.market_type],
            price_type=price_type_map[od["price_type"]],
            time_in_force=tif_map[args.time_in_force],
            order_type=OrderType.Stock,
        )
        resp = sdk.stock.place_order(account, order)
        print(f"[sent] {od['instrument']} qty={od['quantity']} price={od['price']} -> {resp}")


def main() -> None:
    args = parse_args()

    qlib.init(provider_uri=str(PROVIDER_URI), region="tw")

    combo = COMBO_CONFIGS[args.combo]
    experiment = f"tw_train_model_{args.combo}"
    recorder_id = args.recorder_id or choose_latest_recorder(experiment)
    print(f"Using training experiment {experiment}, recorder={recorder_id}")

    # Build dataset and load model
    task_cfg = build_task_config(
        handler_key=combo["handler"],
        model_key=combo["model"],
        instruments=UNIVERSE,
        max_instruments=combo.get("max_instruments"),
        infer_processors=combo.get("infer_processors"),
    )
    dataset = init_instance_by_config(task_cfg["dataset"])
    train_rec = R.get_recorder(experiment_name=experiment, recorder_id=recorder_id)
    model = train_rec.load_object("trained_model")

    last_dt, today_scores = fetch_latest_predictions(model, dataset, topk=args.topk)
    print(f"Latest trading date: {last_dt.date()}, top {len(today_scores)} symbols")

    if args.strategy == "bucket":
        weights = default_bucket_weights(args.topk)
    else:
        weights = [1.0 / args.topk] * args.topk

    price_codes = [str(idx[-1]) if isinstance(idx, tuple) else str(idx) for idx in today_scores.index]
    prices = fetch_close_prices(price_codes, last_dt)
    if args.price_csv:
        df_price = pd.read_csv(args.price_csv)
        if args.instrument_col not in df_price.columns or args.price_col not in df_price.columns:
            raise SystemExit(f"price-csv missing required columns {args.instrument_col}/{args.price_col}")
        override = {
            str(row[args.instrument_col]): float(row[args.price_col])
            for _, row in df_price.iterrows()
            if pd.notna(row[args.price_col])
        }
        prices.update(override)

    orders = build_orders(
        scores=today_scores if today_scores.index.nlevels == 1 else today_scores.droplevel(0),
        weights=weights,
        prices=prices,
        budget=args.budget,
        price_type=args.price_type,
        slippage=args.limit_slippage,
    )

    if not orders:
        print("No tradable symbol (missing close price or budget too small)")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"orders_{args.combo}_{last_dt.date()}.csv"
    pd.DataFrame(orders).to_csv(out_path, index=False)
    print(f"Order list exported:{out_path}")
    print(json.dumps(orders, ensure_ascii=False, indent=2))

    if args.place_orders:
        place_orders(orders, args)
    else:
        print("No --place-orders flag set; only exported order list. Re-run with --place-orders to submit.")


if __name__ == "__main__":
    main()
