#!/bin/bash
# Post-train: fix perms, point latest, publish public repo, restart dashboard.
set -euo pipefail
ROOT=/opt/golias-v27
CKPT="${1:-$ROOT/checkpoints/goliasv-20260613-205716.pt}"

sudo chown -R ubuntu:ubuntu "$ROOT/checkpoints" 2>/dev/null || true
echo "$CKPT" | sudo tee "$ROOT/checkpoints/latest.txt" >/dev/null
sudo ln -sf "$CKPT" "$ROOT/checkpoints/latest.pt"
sudo ln -sf "$CKPT" "$ROOT/latest.pt"
sudo cp -f "$CKPT" "$ROOT/goliasv28.pt"

ENV=/opt/golias/env.sh
if sudo grep -q '^GOLIAS_CKPT=' "$ENV" 2>/dev/null; then
  sudo sed -i "s|^GOLIAS_CKPT=.*|GOLIAS_CKPT=$ROOT/goliasv28.pt|" "$ENV"
fi

set -a
source /opt/golias/config.env 2>/dev/null || true
source /run/golias/secrets.env 2>/dev/null || true
set +a

cd "$ROOT"
/opt/golias/.venv/bin/python - <<PY
import os, sys
sys.path.insert(0, "$ROOT/core")
sys.path.insert(0, "$ROOT/training")
from pathlib import Path
from publish_checkpoint import publish_checkpoint
from checkpoint_registry import append_manifest, write_latest_pointer
ckpt = Path("$CKPT")
write_latest_pointer(Path("$ROOT"), ckpt)
url = publish_checkpoint(ckpt, root=Path("$ROOT"), train_mode="hybrid", metrics_summary="hf avg 0.0026")
append_manifest(Path("$ROOT"), ckpt=ckpt, resume=Path("$ROOT/goliasv28.pt"), mode="hybrid", metrics=[], gh_repo=url)
print("published", url or "(skip)")
PY

sudo systemctl restart goliasv27-dash
sleep 2
KEY=$(grep GOLIAS_INTERNAL_KEY /run/golias/secrets.env | cut -d= -f2-)
curl -s http://127.0.0.1:8080/info -H "X-Golias-Key: $KEY" | python3 -c "import sys,json;d=json.load(sys.stdin);print('ckpt',d.get('ckpt_path'));print('hf_offset',d.get('hf_stream_offset'))"
