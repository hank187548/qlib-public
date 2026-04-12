#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PAPER_TRADE_PYTHON:-$ROOT_DIR/.venv/bin/python}"
CONFIG_PATH="${PAPER_TRADE_CONFIG:-$ROOT_DIR/configs/trade/paper_trading.alpha158_lgb_run11_tplus.example.json}"
LOG_DIR="${PAPER_TRADE_LOG_DIR:-$ROOT_DIR/outputs/paper_trading/_scheduler_logs}"
RUN_LABEL="${PAPER_TRADE_RUN_LABEL:-paper_trade_daily}"
LOCK_FILE="${PAPER_TRADE_LOCK_FILE:-$LOG_DIR/${RUN_LABEL}.lock}"
REFRESH_DATA="${PAPER_TRADE_REFRESH_DATA:-1}"
TARGET_DATE="${PAPER_TRADE_TARGET_DATE:-}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Paper trading config not found: $CONFIG_PATH" >&2
  exit 1
fi

LOG_DATE="${TARGET_DATE:-$(date +%F)}"
LOG_FILE="$LOG_DIR/${RUN_LABEL}_${LOG_DATE}.log"

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    printf '[%s] Skip: another paper-trading run is already active\n' "$(date '+%F %T')" >>"$LOG_FILE"
    exit 0
  fi
fi

cmd=("$PYTHON_BIN" "scripts/trade/paper_trade_daily.py" "--config" "$CONFIG_PATH")
if [[ -n "$TARGET_DATE" ]]; then
  cmd+=("--target-date" "$TARGET_DATE")
fi
case "${REFRESH_DATA,,}" in
  0|false|no)
    cmd+=("--no-refresh-data")
    ;;
esac

{
  printf '[%s] Start paper-trading run\n' "$(date '+%F %T')"
  printf '[%s] Workdir: %s\n' "$(date '+%F %T')" "$ROOT_DIR"
  printf '[%s] Command:' "$(date '+%F %T')"
  printf ' %q' "${cmd[@]}"
  printf '\n'
  "${cmd[@]}"
  printf '[%s] Finished successfully\n' "$(date '+%F %T')"
} >>"$LOG_FILE" 2>&1
