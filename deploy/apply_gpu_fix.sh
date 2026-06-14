#!/bin/bash
set -e
sudo cp /tmp/if_sidecars.py /tmp/watsonx_if.py /opt/golias-v27/core/
sudo cp /tmp/if_rules.py /opt/golias-v27/sidecars/
sudo cp /tmp/dashboard.py /opt/golias-v27/live/
sudo sed -i 's|GOLIAS_CKPT=.*|GOLIAS_CKPT=/opt/golias-v27/goliasv27.pt|' /etc/systemd/system/goliasv27-dash.service
sudo systemctl daemon-reload
sudo systemctl restart goliasv27-dash
sleep 2
curl -s http://127.0.0.1:8080/info
echo
curl -s -X POST http://127.0.0.1:8080/forward -H 'Content-Type: application/json' -d '{"geometry":0.47,"binary":0.73,"language":"who are you?","m1":4.2,"m2":0.55,"m3":0.99,"V":0.58,"if7":0.5}' | head -c 1200
