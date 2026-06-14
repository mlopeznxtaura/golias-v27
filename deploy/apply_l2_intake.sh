#!/bin/bash
set -e
sudo cp /tmp/ui_page.py /opt/golias-v27/live/ui_page.py
sudo cp /tmp/dashboard.py /opt/golias-v27/live/dashboard.py
sudo cp /tmp/v27.py /opt/golias-v27/core/v27.py
sudo cp /tmp/l2_json_intake.py /tmp/normalize_jsonl.py /opt/golias-v27/training/
sudo cp /tmp/l2_4594_corpus.jsonl /tmp/goliasv27_corpus.jsonl /opt/golias-v27/data/
sudo systemctl restart goliasv27-dash
sleep 2
curl -s http://127.0.0.1:8080/health
echo
wc -l /opt/golias-v27/data/goliasv27_corpus.jsonl
