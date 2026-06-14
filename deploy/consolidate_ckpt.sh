#!/bin/bash
# One canonical checkpoint — archive duplicates, fix pointers + env.
set -euo pipefail
ROOT=/opt/golias-v27
ARCH="$ROOT/checkpoints/archive"
mkdir -p "$ARCH"

# Pick winner: latest pointer, else newest mtime among real weights
WINNER=""
if [[ -f "$ROOT/checkpoints/latest.txt" ]]; then
  WINNER=$(cat "$ROOT/checkpoints/latest.txt")
fi
if [[ -z "$WINNER" || ! -f "$WINNER" ]]; then
  WINNER=$(ls -t "$ROOT"/checkpoints/goliasv-*.pt "$ROOT"/goliasv28.pt "$ROOT"/goliasv27.pt 2>/dev/null | head -1)
fi
if [[ -z "$WINNER" || ! -f "$WINNER" ]]; then
  echo "ERROR: no checkpoint found under $ROOT"
  exit 1
fi

echo "Canonical winner: $WINNER"
# Stable alias for inference + GitHub naming
if [[ "$WINNER" != "$ROOT/goliasv28.pt" ]]; then
  sudo cp -f "$WINNER" "$ROOT/goliasv28.pt"
fi
WINNER="$ROOT/goliasv28.pt"

echo "$WINNER" | sudo tee "$ROOT/checkpoints/latest.txt" >/dev/null
sudo ln -sf "$WINNER" "$ROOT/checkpoints/latest.pt"
sudo ln -sf "$WINNER" "$ROOT/latest.pt"

stamp=$(date -u +%Y%m%d-%H%M%S)
for stale in "$ROOT/goliasv27.pt"; do
  if [[ -f "$stale" && "$stale" != "$WINNER" ]]; then
    sudo mv "$stale" "$ARCH/goliasv27-stale-${stamp}.pt"
    echo "Archived stale: $stale"
  fi
done

ENV=/opt/golias/env.sh
sudo sed -i '/^GOLIAS_OUTPUT=/d' "$ENV" 2>/dev/null || true
sudo sed -i '/^GOLIAS_RESUME=/d' "$ENV" 2>/dev/null || true
if sudo grep -q '^GOLIAS_CKPT=' "$ENV" 2>/dev/null; then
  sudo sed -i "s|^GOLIAS_CKPT=.*|GOLIAS_CKPT=$WINNER|" "$ENV"
else
  echo "GOLIAS_CKPT=$WINNER" | sudo tee -a "$ENV" >/dev/null
fi

sudo systemctl restart goliasv27-dash
sleep 2
KEY=$(sudo grep GOLIAS_INTERNAL_KEY "$ENV" | cut -d= -f2-)
curl -s "http://127.0.0.1:8080/info" -H "X-Golias-Key: $KEY" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('live ckpt:', d.get('ckpt_path'))
print('exists:', d.get('ckpt_exists'))
"
