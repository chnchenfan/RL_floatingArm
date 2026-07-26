#!/usr/bin/env bash
# Lightweight out-of-process monitor for the formal Phase-2 systemd service.

set -u

unit="${1:-phase2-residual-formal.service}"
meta_file="${2:-/home/windylab/code/isaac_arm_rl/logs/phase2_residual/formal_4096.meta}"
status_file="${3:-/home/windylab/code/isaac_arm_rl/logs/phase2_residual/formal_4096.status}"
interval_seconds="${PHASE2_MONITOR_INTERVAL_SECONDS:-25}"

log_file="$(sed -n 's/^log=//p' "$meta_file" 2>/dev/null | head -n 1)"
if test -z "$log_file"; then
  echo "[monitor] missing log path in $meta_file" >&2
  exit 2
fi

write_status() {
  local active_state sub_state result main_pid iteration throughput
  local reward position_error orientation_error orientation_limit
  local saturation total_timesteps gpu checkpoint
  local tmp_file

  active_state="$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)"
  sub_state="$(systemctl --user show "$unit" -p SubState --value 2>/dev/null)"
  result="$(systemctl --user show "$unit" -p Result --value 2>/dev/null)"
  main_pid="$(systemctl --user show "$unit" -p MainPID --value 2>/dev/null)"

  iteration="$(rg 'Learning iteration [0-9]+/[0-9]+' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*Learning iteration ([0-9]+\/[0-9]+).*/\1/')"
  throughput="$(rg 'Computation: [0-9]+ steps/s' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*Computation: ([0-9]+ steps\/s).*/\1/')"
  reward="$(rg 'reward/mean:' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*reward\/mean:[[:space:]]*//')"
  position_error="$(rg 'tracking/position_error_mm:' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*tracking\/position_error_mm:[[:space:]]*//')"
  orientation_error="$(rg 'tracking/orientation_error_deg:' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*tracking\/orientation_error_deg:[[:space:]]*//')"
  orientation_limit="$(rg 'tracking/orientation_limit_fraction:' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*tracking\/orientation_limit_fraction:[[:space:]]*//')"
  saturation="$(rg 'residual/saturation_fraction:' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*residual\/saturation_fraction:[[:space:]]*//')"
  total_timesteps="$(rg 'Total timesteps:' "$log_file" 2>/dev/null |
    tail -n 1 | sed -E 's/.*Total timesteps:[[:space:]]*//')"
  checkpoint="$(find /home/windylab/code/isaac_arm_rl/logs/phase2_residual \
    -maxdepth 2 -type f -name 'model_*.pt' -newer "$meta_file" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -n 1 | cut -d' ' -f2-)"
  gpu="$(nvidia-smi \
    --query-gpu=memory.used,memory.free,utilization.gpu,power.draw,temperature.gpu \
    --format=csv,noheader,nounits 2>/dev/null | head -n 1)"

  tmp_file="${status_file}.tmp"
  {
    printf 'checked=%s\n' "$(date --iso-8601=seconds)"
    printf 'unit=%s\n' "$unit"
    printf 'active_state=%s\n' "${active_state:-unknown}"
    printf 'sub_state=%s\n' "${sub_state:-unknown}"
    printf 'result=%s\n' "${result:-unknown}"
    printf 'pid=%s\n' "${main_pid:-0}"
    printf 'iteration=%s\n' "${iteration:-not_started}"
    printf 'throughput=%s\n' "${throughput:-not_available}"
    printf 'reward_mean=%s\n' "${reward:-not_available}"
    printf 'position_error_mm=%s\n' "${position_error:-not_available}"
    printf 'orientation_error_deg=%s\n' "${orientation_error:-not_available}"
    printf 'orientation_limit_fraction=%s\n' "${orientation_limit:-not_available}"
    printf 'residual_saturation=%s\n' "${saturation:-not_available}"
    printf 'total_timesteps=%s\n' "${total_timesteps:-0}"
    printf 'gpu_used_free_util_power_temp=%s\n' "${gpu:-not_available}"
    printf 'latest_checkpoint=%s\n' "${checkpoint:-not_available}"
    printf 'log=%s\n' "$log_file"
  } > "$tmp_file"
  mv "$tmp_file" "$status_file"

  printf '[monitor] %s state=%s/%s iteration=%s reward=%s error_mm=%s ori_deg=%s gpu="%s"\n' \
    "$(date --iso-8601=seconds)" "${active_state:-unknown}" \
    "${sub_state:-unknown}" "${iteration:-not_started}" \
    "${reward:-n/a}" "${position_error:-n/a}" \
    "${orientation_error:-n/a}" "${gpu:-n/a}"
}

while true; do
  write_status
  active_state="$(systemctl --user show "$unit" -p ActiveState --value 2>/dev/null)"
  if test "$active_state" != "active" && test "$active_state" != "activating"; then
    if rg -q '\[phase2\] DONE' "$log_file" 2>/dev/null; then
      message="Phase 2 training completed successfully."
      urgency="normal"
    else
      message="Phase 2 training stopped before completion. Check the monitor status."
      urgency="critical"
    fi
    command -v notify-send >/dev/null 2>&1 &&
      notify-send --urgency="$urgency" "Isaac Arm RL" "$message" || true
    echo "[monitor] $message"
    exit 0
  fi
  sleep "$interval_seconds"
done
