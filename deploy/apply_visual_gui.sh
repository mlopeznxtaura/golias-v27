#!/bin/bash
set -e
sudo cp /tmp/ui_page.py /opt/golias-v27/live/ui_page.py
sudo cp /tmp/dashboard.py /opt/golias-v27/live/dashboard.py
sudo cp /tmp/v27.py /tmp/frame_renderer.py /opt/golias-v27/core/
/opt/golias/.venv/bin/pip install -q Pillow 2>/dev/null || true
sudo systemctl restart goliasv27-dash
sleep 2
curl -s http://127.0.0.1:8080/health
echo
