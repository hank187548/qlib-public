#!/usr/bin/env python3
"""Command-line helper for Masterlink SDK trading.

This script logs into the Masterlink API, optionally performs the one-time
`register_api_auth`, and submits a stock order according to CLI arguments.

Example usage::

    python scripts/trade/masterlink_trade.py \
        --user-id YOUR_LOGIN_ID \
        --password YOUR_PASSWORD \
        --cert secrets/broker_cert.pfx \
        --cert-password YOUR_CERT_PASSWORD \
        --symbol 2888 \
        --side buy \
        --quantity 1000 \
        --price 15.2 \
        --price-type limit

All sensitive values can also be provided via environment variables (see the
argument help text).  The script performs minimal validation; please double
check order parameters before running it in production.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

from masterlink_sdk import (
    MasterlinkSDK,
    Order,
    TimeInForce,
    OrderType,
    PriceType,
    MarketType,
    BSAction,
)

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
DEFAULT_CERT = ROOT_DIR / "secrets" / "broker_cert.pfx"


def _env_default(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() if value else None


SIDE_MAP: Dict[str, BSAction] = {"buy": BSAction.Buy, "sell": BSAction.Sell}

PRICE_TYPE_MAP: Dict[str, PriceType] = {
    "limit": PriceType.Limit,
    "market": PriceType.Market,
    "limitup": PriceType.LimitUp,
    "limitdown": PriceType.LimitDown,
    "reference": PriceType.Reference,
}


def _json_dump(obj: Any) -> str:
    def _to_serializable(value: Any):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple, set)):
            return [_to_serializable(item) for item in value]
        if isinstance(value, dict):
            return {key: _to_serializable(val) for key, val in value.items()}
        if hasattr(value, "to_dict"):
            return _to_serializable(value.to_dict())
        if hasattr(value, "__dict__"):
            return _to_serializable({k: v for k, v in vars(value).items() if not k.startswith("_")})
        return str(value)

    return json.dumps(_to_serializable(obj), indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Place a stock order via Masterlink SDK")
    parser.add_argument("--user-id", default=_env_default("MASTERLINK_ID"), help="Login ID (env MASTERLINK_ID)")
    parser.add_argument(
        "--password",
        default=_env_default("MASTERLINK_PASSWORD"),
        help="Trading password (env MASTERLINK_PASSWORD)",
    )
    parser.add_argument(
        "--cert",
        default=str(_env_default("MASTERLINK_CERT") or DEFAULT_CERT),
        help=f"Path to .pfx certificate (env MASTERLINK_CERT, default {DEFAULT_CERT})",
    )
    parser.add_argument(
        "--cert-password",
        default=_env_default("MASTERLINK_CERT_PASSWORD"),
        help="Certificate password (env MASTERLINK_CERT_PASSWORD)",
    )
    parser.add_argument("--register", action="store_true", help="Call register_api_auth before trading")
    parser.add_argument("--account-index", type=int, default=0, help="Account index when multiple accounts returned")
    parser.add_argument("--symbol", required=True, help="Stock symbol (e.g. 2888)")
    parser.add_argument(
        "--side",
        choices=sorted(SIDE_MAP.keys()),
        required=True,
        help="buy or sell",
    )
    parser.add_argument("--quantity", type=int, required=True, help="Order quantity")
    parser.add_argument(
        "--price",
        type=float,
        default=None,
        help="Limit price (omit for market/limit-up/down as needed)",
    )
    parser.add_argument(
        "--price-type",
        choices=sorted(PRICE_TYPE_MAP.keys()),
        default="limit",
        help="Price type (default limit)",
    )
    parser.add_argument(
        "--time-in-force",
        choices=["rod", "ioc", "fok"],
        default="rod",
        help="Day order policy",
    )
    parser.add_argument(
        "--market-type",
        choices=["common", "fixing", "odd", "intraday-odd", "emg"],
        default="common",
        help="Market session (e.g., common=regular, fixing=after-hours)",
    )
    parser.add_argument(
        "--order-type",
        choices=["stock"],
        default="stock",
        help="Currently only stock supported",
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip place_order, only print payload")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    missing = [name for name in ("user_id", "password", "cert", "cert_password") if getattr(args, name) is None]
    if missing:
        raise SystemExit(f"Missing required credentials: {', '.join(missing)}")

    cert_path = Path(args.cert).expanduser().resolve()
    if not cert_path.exists():
        raise SystemExit(f"Certificate not found: {cert_path}")

    sdk = MasterlinkSDK()
    print("Logging in...")
    accounts = sdk.login(args.user_id, args.password, str(cert_path), args.cert_password)
    if not accounts:
        raise SystemExit("No accounts returned from login")

    if args.account_index < 0 or args.account_index >= len(accounts):
        raise SystemExit(f"account-index {args.account_index} out of range (got {len(accounts)} accounts)")
    account = accounts[args.account_index]
    print("Selected account:")
    print(_json_dump(account))

    if args.register:
        print("Calling register_api_auth...")
        sdk.register_api_auth(account)
        print("register_api_auth completed")

    price_type = PRICE_TYPE_MAP[args.price_type]
    if price_type == PriceType.Limit and args.price is None:
        raise SystemExit("Limit orders require --price")
    price_value = None if args.price is None else f"{args.price}"
    if price_type != PriceType.Limit:
        # Ensure accidental price is cleared for non-limit types unless explicitly needed.
        price_value = price_value if args.price is not None else None

    order = Order(
        buy_sell=SIDE_MAP[args.side],
        symbol=args.symbol,
        quantity=args.quantity,
        price=price_value,
        market_type={
            "common": MarketType.Common,
            "fixing": MarketType.Fixing,
            "odd": MarketType.Odd,
            "intraday-odd": MarketType.IntradayOdd,
            "emg": MarketType.Emg,
        }[args.market_type],
        price_type=price_type,
        time_in_force={
            "rod": TimeInForce.ROD,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }[args.time_in_force],
        order_type=OrderType.Stock,
    )
    print("Prepared order payload:")
    print(_json_dump(order))

    if args.dry_run:
        print("Dry-run mode: skipping place_order")
        return 0

    print("Submitting order...")
    response = sdk.stock.place_order(account, order)
    print("Order response:")
    print(_json_dump(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
