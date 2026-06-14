#!/bin/bash
# gh auth from IBM SM secrets (loaded into /run/golias/secrets.env).
set -euo pipefail
CONFIG="${GOLIAS_CONFIG_FILE:-/opt/golias/config.env}"
SECRETS="${GOLIAS_SECRETS_FILE:-/run/golias/secrets.env}"
[[ -f "$CONFIG" ]] && set -a && source "$CONFIG" && set +a
[[ -f "$SECRETS" ]] && set -a && source "$SECRETS" && set +a

if ! command -v gh >/dev/null; then
  echo "Install gh first: deploy/setup_github_ssh.sh"
  exit 1
fi

TOKEN="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [[ -z "$TOKEN" ]]; then
  echo "ERROR: GH_TOKEN not in $SECRETS — run sync_all_sm_to_gpu.ps1"
  exit 1
fi

echo "$TOKEN" | gh auth login --with-token
gh config set git_protocol ssh
gh auth status

# Non-secret defaults only (org comes from SM via secrets.env)
: "${GOLIAS_GH_VISIBILITY:=public}"
: "${GOLIAS_GH_PUBLISH:=1}"
echo "gh ready — org=\${GOLIAS_GH_ORG:-unset} visibility=$GOLIAS_GH_VISIBILITY"
