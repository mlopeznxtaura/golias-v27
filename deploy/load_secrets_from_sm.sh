#!/bin/bash
# GPU-side: refresh /run/golias/secrets.env from IBM Secrets Manager (no secrets in git/env.sh).
set -euo pipefail

CONFIG="${GOLIAS_CONFIG_FILE:-/opt/golias/config.env}"
SECRETS_OUT="${GOLIAS_SECRETS_FILE:-/run/golias/secrets.env}"
ROOT="${GOLIAS_ROOT:-/opt/golias-v27}"
VENV="${GOLIAS_VENV:-/opt/golias/.venv/bin/python}"

mkdir -p "$(dirname "$SECRETS_OUT")"
[[ -f "$CONFIG" ]] && set -a && source "$CONFIG" && set +a

if [[ -z "${IBM_SECRETS_MANAGER_URL:-}" ]]; then
  echo "[secrets] IBM_SECRETS_MANAGER_URL not set in $CONFIG — skip SM refresh"
  exit 0
fi

# Bootstrap reader key: file synced once from SM (chmod 600), not in env.sh
BOOT="/opt/golias/.ibm_cloud_api_key"
if [[ -f "$BOOT" ]]; then
  export IBM_CLOUD_API_KEY="$(cat "$BOOT")"
fi

"$VENV" - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, os.environ.get("GOLIAS_ROOT", "/opt/golias-v27") + "/core")
from secrets_loader import fetch_secrets_from_sm, _manifest

out = Path(os.environ.get("GOLIAS_SECRETS_FILE", "/run/golias/secrets.env"))
fetched = fetch_secrets_from_sm()
if not fetched:
    print("[secrets] nothing fetched from SM", flush=True)
    sys.exit(0)
lines = [f'{k}={v}' for k, v in fetched.items()]
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(out, 0o600)
print(f"[secrets] wrote {len(fetched)} keys to {out}", flush=True)
PY

chmod 600 "$SECRETS_OUT" 2>/dev/null || true
