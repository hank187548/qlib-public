#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SCHEDULER_TZ="${PAPER_TRADE_SCHEDULER_TZ:-Asia/Taipei}"
SCHEDULE_SPEC="${PAPER_TRADE_SCHEDULE:-08:30 20:00}"
POLL_SECONDS="${PAPER_TRADE_POLL_SECONDS:-30}"
EXIT_AFTER_RUN="${PAPER_TRADE_SCHEDULER_EXIT_AFTER_RUN:-0}"

LOG_DIR="${PAPER_TRADE_LOG_DIR:-$ROOT_DIR/outputs/paper_trading/_scheduler_logs}"
SCHEDULER_LABEL="${PAPER_TRADE_SCHEDULER_LABEL:-paper_trade_scheduler}"
SCHEDULER_LOG_FILE="${PAPER_TRADE_SCHEDULER_LOG_FILE:-$LOG_DIR/${SCHEDULER_LABEL}.log}"
STATE_FILE="${PAPER_TRADE_SCHEDULER_STATE_FILE:-$LOG_DIR/${SCHEDULER_LABEL}.state}"
LOCK_FILE="${PAPER_TRADE_SCHEDULER_LOCK_FILE:-$LOG_DIR/${SCHEDULER_LABEL}.lock}"
DAILY_WRAPPER="${PAPER_TRADE_DAILY_WRAPPER:-$ROOT_DIR/scripts/trade/run_paper_trade_daily.sh}"

mkdir -p "$LOG_DIR"

if [[ ! -x "$DAILY_WRAPPER" ]]; then
  echo "Paper trading wrapper not found or not executable: $DAILY_WRAPPER" >&2
  exit 1
fi

if ! [[ "$POLL_SECONDS" =~ ^[0-9]+$ ]] || [[ "$POLL_SECONDS" -le 0 ]]; then
  echo "PAPER_TRADE_POLL_SECONDS must be a positive integer, got: $POLL_SECONDS" >&2
  exit 1
fi

if command -v flock >/dev/null 2>&1; then
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    printf '[%s] Skip: another scheduler instance is already active\n' "$(TZ="$SCHEDULER_TZ" date '+%F %T')" >>"$SCHEDULER_LOG_FILE"
    exit 0
  fi
fi

declare -a SCHEDULE_TIMES=()
for slot in $SCHEDULE_SPEC; do
  if ! [[ "$slot" =~ ^[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Invalid schedule slot: $slot" >&2
    exit 1
  fi
  SCHEDULE_TIMES+=("$slot")
done

if [[ "${#SCHEDULE_TIMES[@]}" -eq 0 ]]; then
  echo "No schedule times configured" >&2
  exit 1
fi

state_value() {
  local slot="$1"
  local key="${slot/:/}"
  if [[ -f "$STATE_FILE" ]]; then
    awk -F'|' -v key="$key" '$1 == key {print $2}' "$STATE_FILE" | tail -n 1
  fi
}

write_state() {
  local slot="$1"
  local run_date="$2"
  local key="${slot/:/}"
  local tmp_file
  tmp_file="$(mktemp "$LOG_DIR/${SCHEDULER_LABEL}.state.XXXX")"
  if [[ -f "$STATE_FILE" ]]; then
    awk -F'|' -v key="$key" '$1 != key {print $0}' "$STATE_FILE" >"$tmp_file"
  fi
  printf '%s|%s\n' "$key" "$run_date" >>"$tmp_file"
  mv "$tmp_file" "$STATE_FILE"
}

log_line() {
  printf '[%s] %s\n' "$(TZ="$SCHEDULER_TZ" date '+%F %T')" "$*" >>"$SCHEDULER_LOG_FILE"
}

run_daily_wrapper() {
  local slot="$1"
  log_line "Trigger slot $slot"
  if "$DAILY_WRAPPER"; then
    log_line "Slot $slot finished successfully"
    return 0
  fi
  log_line "Slot $slot failed"
  return 1
}

log_line "Scheduler started"
log_line "Timezone: $SCHEDULER_TZ"
log_line "Schedule: ${SCHEDULE_TIMES[*]}"
log_line "Poll seconds: $POLL_SECONDS"

while true; do
  current_date="$(TZ="$SCHEDULER_TZ" date +%F)"
  current_time="$(TZ="$SCHEDULER_TZ" date +%H:%M)"

  for slot in "${SCHEDULE_TIMES[@]}"; do
    last_run_date="$(state_value "$slot")"
    if [[ "$current_time" < "$slot" ]]; then
      continue
    fi
    if [[ "$last_run_date" == "$current_date" ]]; then
      continue
    fi
    if run_daily_wrapper "$slot"; then
      write_state "$slot" "$current_date"
      if [[ "$EXIT_AFTER_RUN" == "1" ]]; then
        log_line "Exit after run requested; scheduler stopping"
        exit 0
      fi
    fi
  done

  sleep "$POLL_SECONDS"
done
