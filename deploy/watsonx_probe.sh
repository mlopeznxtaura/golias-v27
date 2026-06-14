#!/bin/bash
eval "$(sudo grep -E '^(IF_|WATSONX_)' /run/golias/secrets.env 2>/dev/null | sed 's/^/export /')"
eval "$(sudo grep -E '^(IF_|WATSONX_)' /opt/golias/config.env 2>/dev/null | sed 's/^/export /')"
/opt/golias/.venv/bin/python <<'PY'
import json, os, urllib.error, urllib.parse, urllib.request

api = os.environ["WATSONX_API_KEY"]
proj = os.environ["WATSONX_PROJECT_ID"]
base = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
model = os.environ.get("WATSONX_MODEL", "ibm/granite-3-8b-instruct")

data = urllib.parse.urlencode({
    "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
    "apikey": api,
}).encode()
req = urllib.request.Request("https://iam.cloud.ibm.com/identity/token", data=data, method="POST")
req.add_header("Content-Type", "application/x-www-form-urlencoded")
tok = json.loads(urllib.request.urlopen(req, timeout=10).read())["access_token"]
print("token ok")

body = json.dumps({
    "model_id": model,
    "project_id": proj,
    "input": "hello",
    "parameters": {"max_new_tokens": 8},
}).encode()
req = urllib.request.Request(f"{base}/ml/v1/text/generation?version=2023-05-29", data=body, method="POST")
req.add_header("Authorization", f"Bearer {tok}")
req.add_header("Content-Type", "application/json")
try:
    print(urllib.request.urlopen(req, timeout=15).read()[:500])
except urllib.error.HTTPError as e:
    print("HTTP", e.code)
    print(e.read().decode()[:1200])
PY
