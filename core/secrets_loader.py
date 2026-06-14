"""Load Golias credentials from IBM Secrets Manager at runtime.

Requires (non-secret bootstrap in /opt/golias/config.env):
  IBM_SECRETS_MANAGER_URL

Requires (from /run/golias/secrets.env or SM):
  IBM_CLOUD_API_KEY — reader key with Secrets Manager access

Never hardcode org names, API keys, or SM instance URLs in application code.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any

_MANIFEST = Path(__file__).resolve().parent.parent / "deploy" / "sm_manifest.json"
_SECRETS_FILE = Path(os.environ.get("GOLIAS_SECRETS_FILE", "/run/golias/secrets.env"))
_CONFIG_FILE = Path(os.environ.get("GOLIAS_CONFIG_FILE", "/opt/golias/config.env"))
_LOADED = False


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _iam_token(api_key: str) -> str:
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": api_key,
    }).encode()
    req = urllib.request.Request(
        "https://iam.cloud.ibm.com/identity/token",
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())["access_token"]


def _sm_get_payload(sm_url: str, token: str, secret_name: str, group: str = "default") -> str:
    base = sm_url.rstrip("/")
    q = urllib.parse.urlencode({
        "name": secret_name,
        "secret_type": "arbitrary",
        "secret_group_name": group,
    })
    url = f"{base}/api/v2/secrets?{q}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        data = json.loads(resp.read())
    secrets = data.get("secrets") or []
    if not secrets:
        raise RuntimeError(f"SM secret not found: {secret_name}")
    sid = secrets[0]["id"]
    vurl = f"{base}/api/v2/secrets/{sid}/versions/latest"
    req2 = urllib.request.Request(vurl, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urllib.request.urlopen(req2, timeout=20) as resp2:
        ver = json.loads(resp2.read())
    payload = ver.get("payload") or (ver.get("data") or {}).get("payload")
    if not payload:
        raise RuntimeError(f"empty payload for {secret_name}")
    return str(payload)


@lru_cache(maxsize=1)
def _manifest() -> dict[str, Any]:
    if _MANIFEST.is_file():
        return json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return {"secrets": [], "config": {}, "secret_group": "default"}


def fetch_secrets_from_sm() -> dict[str, str]:
    """Pull all manifest secrets from IBM SM. Returns env key → value."""
    sm_url = os.environ.get("IBM_SECRETS_MANAGER_URL", "").strip()
    api_key = os.environ.get("IBM_CLOUD_API_KEY", "").strip()
    if not sm_url or not api_key:
        return {}
    token = _iam_token(api_key)
    mf = _manifest()
    group = mf.get("secret_group", "default")
    out: dict[str, str] = {}
    for row in mf.get("secrets", []):
        name = row["sm_name"]
        env_key = row["env"]
        try:
            out[env_key] = _sm_get_payload(sm_url, token, name, group)
        except (urllib.error.URLError, RuntimeError, KeyError) as ex:
            print(f"[secrets] skip {name}: {ex}", flush=True)
    return out


def ensure_secrets(*, refresh_sm: bool = False) -> None:
    """Load config + secrets into os.environ. Idempotent."""
    global _LOADED
    if _LOADED and not refresh_sm:
        return

    _load_dotenv(_CONFIG_FILE)
    _load_dotenv(_SECRETS_FILE)

    for key, val in (_manifest().get("config") or {}).items():
        if key not in os.environ:
            os.environ[key] = str(val)

    if refresh_sm or not os.environ.get("GH_TOKEN") or not os.environ.get("WATSONX_API_KEY"):
        fetched = fetch_secrets_from_sm()
        for k, v in fetched.items():
            os.environ[k] = v
        if fetched.get("GH_TOKEN"):
            os.environ["GITHUB_TOKEN"] = fetched["GH_TOKEN"]

    _LOADED = True


def github_token() -> str:
    ensure_secrets()
    return os.environ.get("GH_TOKEN", "") or os.environ.get("GITHUB_TOKEN", "")
