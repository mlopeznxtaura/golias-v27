#!/bin/bash
set -e
PID=$(pgrep -f "golias-v27/live/dashboard.py" | head -1)
echo "PID=$PID"
if [ -n "$PID" ]; then
  sudo tr '\0' '\n' < "/proc/$PID/environ" | grep -E '^(WATSONX|IF_)' | sed 's/=.*/=SET/'
fi
cd /opt/golias-v27
eval "$(sudo grep -E '^(IF_|WATSONX_)' /run/golias/secrets.env 2>/dev/null | sed 's/^/export /')"
eval "$(sudo grep -E '^(IF_|WATSONX_)' /opt/golias/config.env 2>/dev/null | sed 's/^/export /')"
/opt/golias/.venv/bin/python <<'PY'
import os, sys
sys.path.insert(0, "core")
print("KEY", bool(os.environ.get("WATSONX_API_KEY")))
print("PROJ", (os.environ.get("WATSONX_PROJECT_ID") or "")[:8])
try:
    from watsonx_if import call_m1
    from if_sidecars import build_if_payload
    p = build_if_payload(0.47, 0.73, "who are you", 0.65, 4.2, 0.55, 0.99)
    r = call_m1(p)
    print("M1 OK", r.get("exploration", "")[:80])
except Exception as e:
    print("ERR", type(e).__name__, e)
PY
