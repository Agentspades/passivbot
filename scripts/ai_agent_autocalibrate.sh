#!/usr/bin/env bash
set -euo pipefail

PBROOT=/root/ai-trading-server
AUDIT="$PBROOT/passivbot/logs/ai_agent_audit.jsonl"
CAL="$PBROOT/passivbot/logs/ai_agent_calibration.remote.json"
CFG="$PBROOT/pbgui/data/run_v7/botv1/config_run.json"
LOG="$PBROOT/pbgui/data/run_v7/botv1/passivbot.log"
LOCKFILE=/tmp/ai_agent_autocalibrate.lock

exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "[autocal] another run is in progress; exiting" >&2
  exit 0
fi

PYBIN=$(command -v python3 || command -v python)
JQ=$(command -v jq)
# Damping limits: max relative change per run (default 0.20 = 20%)
DAMP_PCT=${DAMP_PCT:-0.20}
# Minimum relative change to trigger restart (default 0.05 = 5%)
MIN_REL_DELTA=${MIN_REL_DELTA:-0.05}
TS=$(date -u +%Y%m%dT%H%M%SZ)
echo "[autocal][$TS] starting autocalibration (last 4h)"

if [ ! -f "$AUDIT" ]; then
  echo "[autocal] audit log not found: $AUDIT" >&2
  exit 1
fi

# 1) Run calibration on last 4 hours window
set +e
"$PYBIN" "$PBROOT/passivbot/scripts/ai_agent_calibrate.py" \
  --audit "$AUDIT" \
  --output "$CAL" \
  --last-hours 4
RC=$?
set -e
if [ $RC -ne 0 ]; then
  echo "[autocal] calibration failed (rc=$RC)" >&2
  exit $RC
fi

# 2) Extract suggested thresholds
ENTER=$($JQ -r '.suggestions.hysteresis.volatility_enter // empty' "$CAL" 2>/dev/null || true)
EXIT=$($JQ -r '.suggestions.hysteresis.volatility_exit // empty' "$CAL" 2>/dev/null || true)
if [ -z "${ENTER:-}" ] || [ -z "${EXIT:-}" ]; then
  echo "[autocal] no suggestions found; skipping patch/restart" >&2
  exit 0
fi

# 3) Sanity bounds and relationship
read -r ENTER_C EXIT_C < <(python3 - "$ENTER" "$EXIT" <<'PY'
import sys
ent=float(sys.argv[1]); ex=float(sys.argv[2])
ent=max(0.001, min(ent, 0.02))
ex=max(0.0005, min(ex, ent*0.95))
print(f"{ent:.6f} {ex:.6f}")
PY
)

# 4) Patch config if changed
CURR_ENTER=$($JQ -r '.live.ai_agent.hysteresis.volatility_enter' "$CFG" 2>/dev/null || echo "")
CURR_EXIT=$($JQ -r '.live.ai_agent.hysteresis.volatility_exit' "$CFG" 2>/dev/null || echo "")

# 4a) Apply damping relative to current values
python3 - "$ENTER_C" "$EXIT_C" "$CURR_ENTER" "$CURR_EXIT" "$DAMP_PCT" <<'PY'
import sys
ent=float(sys.argv[1]); ex=float(sys.argv[2])
curr_ent=float(sys.argv[3]); curr_ex=float(sys.argv[4])
damp=float(sys.argv[5])
def damp_to(curr, target, damp):
    lo=curr*(1.0-damp); hi=curr*(1.0+damp)
    return min(max(target, lo), hi)
ent_d=damp_to(curr_ent, ent, damp)
ex_d=damp_to(curr_ex, ex, damp)
if ex_d>ent_d*0.95:
    ex_d=ent_d*0.95
print(f"{ent_d:.6f} {ex_d:.6f}")
PY
read -r ENTER_D EXIT_D < <(python3 - "$ENTER_C" "$EXIT_C" "$CURR_ENTER" "$CURR_EXIT" "$DAMP_PCT" <<'PY'
import sys
ent=float(sys.argv[1]); ex=float(sys.argv[2])
curr_ent=float(sys.argv[3]); curr_ex=float(sys.argv[4])
damp=float(sys.argv[5])
def damp_to(curr, target, damp):
    lo=curr*(1.0-damp); hi=curr*(1.0+damp)
    return min(max(target, lo), hi)
ent_d=damp_to(curr_ent, ent, damp)
ex_d=damp_to(curr_ex, ex, damp)
if ex_d>ent_d*0.95:
    ex_d=ent_d*0.95
print(f"{ent_d:.6f} {ex_d:.6f}")
PY
)

# 4b) Skip if change is too small (min relative delta)
small_change=true
if [ -n "$CURR_ENTER" ] && awk -v a="$CURR_ENTER" -v b="$ENTER_D" -v t="$MIN_REL_DELTA" 'BEGIN{if(a<=0){exit 1}; if (( (a>b?a-b:b-a) / a )>=t) exit 1; else exit 0}'; then small_change=true; else small_change=false; fi
if [ "$small_change" = true ] && [ -n "$CURR_EXIT" ] && awk -v a="$CURR_EXIT" -v b="$EXIT_D" -v t="$MIN_REL_DELTA" 'BEGIN{if(a<=0){exit 1}; if (( (a>b?a-b:b-a) / a )>=t) exit 1; else exit 0}'; then small_change=true; else small_change=false; fi
if [ "$small_change" = true ]; then
  echo "[autocal] change below MIN_REL_DELTA (enter $CURR_ENTER->${ENTER_D}, exit $CURR_EXIT->${EXIT_D}); skipping restart"
  exit 0
fi

ENTER_NEW="$ENTER_D"
EXIT_NEW="$EXIT_D"

cp "$CFG" "${CFG}.bak.${TS}"
TMP="${CFG}.tmp.${TS}"
$JQ \
  --argjson ent "$ENTER_NEW" \
  --argjson ex "$EXIT_NEW" \
  '.live.ai_agent.hysteresis.volatility_enter=$ent | .live.ai_agent.hysteresis.volatility_exit=$ex' \
  "$CFG" > "$TMP"
mv "$TMP" "$CFG"
echo "[autocal] patched hysteresis: enter=$ENTER_NEW exit=$EXIT_NEW (backup: ${CFG}.bak.${TS})"

# 5) Graceful restart: kill PBRun-managed passivbot; PBRun will respawn
PB_PID=$(pgrep -f "/pb_venv/bin/python -u /root/ai-trading-server/passivbot/src/main.py .*config_run.json" | head -n1 || true)
if [ -n "${PB_PID:-}" ]; then
  echo "[autocal] killing PBRun-managed pid $PB_PID"
  kill "$PB_PID" || true
else
  echo "[autocal] PBRun-managed process not found; searching any passivbot main.py"
  ANY_PID=$(pgrep -f "/pb_venv/bin/python .*src/main.py .*config_run.json" | head -n1 || true)
  if [ -n "${ANY_PID:-}" ]; then
    echo "[autocal] killing pid $ANY_PID"
    kill "$ANY_PID" || true
  fi
fi

# 6) Wait up to 45s for passivbot to resume and acknowledge new thresholds
for i in $(seq 1 45); do
  sleep 1
  if grep -q "vol_enter=${ENTER_NEW}" "$LOG" && grep -q "vol_exit=${EXIT_NEW}" "$LOG"; then
    echo "[autocal] verified new thresholds in log"
    exit 0
  fi
  pgrep -f "/pb_venv/bin/python .*src/main.py .*config_run.json" >/dev/null 2>&1 || true
done

echo "[autocal] WARNING: did not verify thresholds in log within timeout" >&2
exit 0
