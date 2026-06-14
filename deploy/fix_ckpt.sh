#!/bin/bash
set -e
# Point live inference at checkpoints/latest.txt (newest trained .pt)
ROOT=/opt/golias-v27
LATEST="$ROOT/checkpoints/latest.txt"
if [[ -f "$LATEST" ]]; then
  CKPT=$(cat "$LATEST")
else
  CKPT=$(ls -t "$ROOT"/checkpoints/goliasv-*.pt "$ROOT"/goliasv*.pt 2>/dev/null | head -1)
  CKPT="${CKPT:-$ROOT/goliasv27.pt}"
fi
echo "Using checkpoint: $CKPT"
if sudo grep -q '^GOLIAS_CKPT=' /opt/golias/env.sh 2>/dev/null; then
  sudo sed -i "s|^GOLIAS_CKPT=.*|GOLIAS_CKPT=$CKPT|" /opt/golias/env.sh
else
  echo "GOLIAS_CKPT=$CKPT" | sudo tee -a /opt/golias/env.sh
fi
# Remove dangerous overwrite of output onto resume
sudo sed -i '/^GOLIAS_OUTPUT=.*goliasv27\.pt/d' /opt/golias/env.sh 2>/dev/null || true
sudo systemctl restart goliasv27-dash
sleep 2
KEY=$(sudo grep GOLIAS_INTERNAL_KEY /run/golias/secrets.env 2>/dev/null | cut -d= -f2-)
[[ -z "$KEY" ]] && KEY=$(sudo grep GOLIAS_INTERNAL_KEY /opt/golias/env.sh 2>/dev/null | cut -d= -f2-)
curl -s http://127.0.0.1:8080/info -H "X-Golias-Key: $KEY" | python3 -c "import sys,json; d=json.load(sys.stdin); print('ckpt', d.get('ckpt_path'))"
