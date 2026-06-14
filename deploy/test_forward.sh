#!/bin/bash
KEY=$(sudo grep GOLIAS_INTERNAL_KEY /run/golias/secrets.env 2>/dev/null | cut -d= -f2-)
[[ -z "$KEY" ]] && KEY=$(sudo grep GOLIAS_INTERNAL_KEY /opt/golias/env.sh 2>/dev/null | cut -d= -f2-)
curl -s http://127.0.0.1:8080/info -H "X-Golias-Key: $KEY"
echo
curl -s -X POST http://127.0.0.1:8080/forward -H "X-Golias-Key: $KEY" -H "Content-Type: application/json" -d '{"geometry":0.47,"binary":0.73,"language":"who are you?","m1":4.2,"m2":0.55,"m3":0.99,"V":0.58,"if7":0.5}'
