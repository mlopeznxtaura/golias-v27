#!/bin/bash
# One-time: move secrets from legacy env.sh → /run/golias/secrets.env (then strip env.sh).
set -euo pipefail
LEGACY=/opt/golias/env.sh
SECRETS=/run/golias/secrets.env
CONFIG=/opt/golias/config.env

sudo mkdir -p /run/golias
sudo rm -f "$SECRETS"

tmp=$(mktemp)
for key in IBM_CLOUD_API_KEY WATSONX_API_KEY WATSONX_PROJECT_ID GH_TOKEN GITHUB_TOKEN HF_TOKEN GOLIAS_INTERNAL_KEY GOLIAS_GH_ORG; do
  line=$(sudo grep -m1 -E "^${key}=" "$LEGACY" 2>/dev/null || true)
  if [[ -n "$line" ]]; then
    echo "$line" >> "$tmp"
  fi
done

if [[ ! -s "$tmp" ]]; then
  echo "ERROR: no secrets found in $LEGACY" >&2
  rm -f "$tmp"
  exit 1
fi

sudo mv "$tmp" "$SECRETS"
sudo chmod 600 "$SECRETS"

if [[ ! -f "$CONFIG" ]]; then
  sudo tee "$CONFIG" >/dev/null <<'CFG'
GOLIAS_ROOT=/opt/golias-v27
GOLIAS_CONFIG_FILE=/opt/golias/config.env
GOLIAS_SECRETS_FILE=/run/golias/secrets.env
IF_BACKEND=watsonx
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_MODEL=ibm/granite-3-8b-instruct
GOLIAS_GH_VISIBILITY=public
GOLIAS_GH_PUBLISH=1
CFG
fi

sudo sed -i '/^GH_TOKEN=/d;/^GITHUB_TOKEN=/d;/^WATSONX_/d;/^HF_TOKEN=/d;/^GOLIAS_INTERNAL_KEY=/d;/^IBM_CLOUD_API_KEY=/d;/^GOLIAS_GH_ORG=/d' "$LEGACY"
echo "migrated $(sudo wc -l < "$SECRETS") keys → $SECRETS; stripped secrets from $LEGACY"
