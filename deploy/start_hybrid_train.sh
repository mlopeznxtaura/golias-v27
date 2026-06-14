#!/bin/bash
set -e
sudo rsync -a /tmp/golias-sync/core/ /opt/golias-v27/core/
sudo rsync -a /tmp/golias-sync/training/ /opt/golias-v27/training/
sudo rsync -a /tmp/golias-sync/live/ /opt/golias-v27/live/
sudo rsync -a /tmp/golias-sync/data/ /opt/golias-v27/data/
sudo mkdir -p /opt/golias-v27/checkpoints /opt/golias-v27/data/inbox

set -a
[[ -f /opt/golias/config.env ]] && source /opt/golias/config.env
[[ -f /run/golias/secrets.env ]] && source /run/golias/secrets.env
[[ -f /opt/golias/env.sh ]] && eval "$(sudo grep -E '^(GOLIAS_|PORT)' /opt/golias/env.sh 2>/dev/null | grep -v '_KEY\|TOKEN\|WATSONX\|GH_\|HF_' | sed 's/^/export /' || true)"
set +a

export GOLIAS_TRAIN_MODE=hybrid
export GOLIAS_JSONL=/opt/golias-v27/data/goliasv27_corpus.jsonl
export GOLIAS_RESUME="$(python3 -c "import sys; sys.path.insert(0,'/opt/golias-v27/core'); from checkpoint_registry import resolve_active_ckpt; print(resolve_active_ckpt('/opt/golias-v27'))")"
unset GOLIAS_OUTPUT
export GOLIAS_CHECKPOINT=/opt/golias-v27/checkpoints/rolling_checkpoint.pt
export GOLIAS_LOG=/opt/golias-v27/train_hybrid.log
export GOLIAS_JSONL_EPOCHS=3
export GOLIAS_HF_SAMPLES=50000
export GOLIAS_HF_OFFSET=0
export GOLIAS_BATCH=32
export GOLIAS_GH_PUBLISH=1

cd /opt/golias-v27
echo "resume=$GOLIAS_RESUME"
nohup /opt/golias/.venv/bin/python training/train_hybrid.py >> "$GOLIAS_LOG" 2>&1 &
echo "hybrid train pid=$!"
sudo systemctl restart goliasv27-dash
