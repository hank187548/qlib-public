#!/usr/bin/env python3
"""Simple smoke test for the Masterlink SDK login flow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from masterlink_sdk import MasterlinkSDK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_serializable(obj):
    """Convert SDK account objects into JSON-friendly structures."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [_to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    if hasattr(obj, "to_dict"):
        return _to_serializable(obj.to_dict())
    if hasattr(obj, "__dict__"):
        public_attrs = {k: v for k, v in vars(obj).items() if not k.startswith("_")}
        return {key: _to_serializable(value) for key, value in public_attrs.items()}
    return str(obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Masterlink SDK login smoke test")
    parser.add_argument("--register", action="store_true", help="Call register_api_auth on the first account")
    return parser.parse_args()


def _load_credentials() -> tuple[str, str, Path, str]:
    user_id = os.environ.get("MASTERLINK_ID")
    password = os.environ.get("MASTERLINK_PASSWORD")
    cert = os.environ.get("MASTERLINK_CERT")
    cert_password = os.environ.get("MASTERLINK_CERT_PASSWORD")

    missing = [
        name
        for name, value in (
            ("MASTERLINK_ID", user_id),
            ("MASTERLINK_PASSWORD", password),
            ("MASTERLINK_CERT", cert),
            ("MASTERLINK_CERT_PASSWORD", cert_password),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    cert_path = Path(cert).expanduser().resolve()
    return user_id, password, cert_path, cert_password


def main() -> int:
    args = parse_args()
    user_id, password, cert_path, cert_password = _load_credentials()

    if not cert_path.exists():
        raise SystemExit(f"Certificate file not found: {cert_path}")

    sdk = MasterlinkSDK()
    print("Logging in with Masterlink SDK...")
    accounts = sdk.login(user_id, password, str(cert_path), cert_password)
    if not accounts:
        raise SystemExit("Login returned no accounts")

    print("Login successful. Available accounts JSON:")
    print(json.dumps(_to_serializable(accounts), indent=2, ensure_ascii=False))

    if args.register:
        print("Calling register_api_auth on the first account...")
        sdk.register_api_auth(accounts[0])
        print("register_api_auth completed.")

    print("Masterlink SDK test completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
