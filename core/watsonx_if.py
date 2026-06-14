"""Watsonx.ai clients for M1/M2/M3 input functions."""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from if_sidecars import SidecarFallbackError

def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)

_TOKEN: str | None = None

SYSTEM_M1 = (
    "You are M1, Golias dreamer. Generate exploratory continuation of the physics state. "
    "Ignore efficiency. Reply JSON only: {\"exploration\": \"...\", \"explore_scalar\": 0.0-1.0}"
)
SYSTEM_M2 = (
    "You are M2, efficiency zealot. Score compute waste 0-1 as c_comp_proxy. "
    "Set halt true if c_comp_proxy > tau. Reply JSON only: "
    "{\"efficiency\": 0.0-1.0, \"halt\": bool, \"c_comp_proxy\": float}"
)
SYSTEM_M3 = (
    "You are M3, arbiter. Balance M1 exploration vs M2 efficiency. "
    "Reply JSON only: {\"meta\": 0.0-1.0, \"arbitration\": \"...\", \"chosen_path\": \"explore|efficient\"}"
)


def _get_token() -> str:
    global _TOKEN
    if _TOKEN:
        return _TOKEN
    if not _env("WATSONX_API_KEY"):
        raise SidecarFallbackError("WATSONX_API_KEY not set")
    url = "https://iam.cloud.ibm.com/identity/token"
    data = urllib.parse.urlencode({
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
        "apikey": _env("WATSONX_API_KEY"),
    }).encode()
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            _TOKEN = json.loads(resp.read())["access_token"]
            return _TOKEN
    except Exception as ex:
        raise SidecarFallbackError(f"iam token: {ex}") from ex


def _generate(system: str, user: str, temperature: float, max_tokens: int, timeout: float) -> str:
    project_id = _env("WATSONX_PROJECT_ID")
    if not project_id:
        raise SidecarFallbackError("WATSONX_PROJECT_ID not set")
    token = _get_token()
    base_url = _env("WATSONX_URL", "https://us-south.ml.cloud.ibm.com").rstrip("/")
    model_id = _env("WATSONX_MODEL", "ibm/granite-3-8b-instruct")
    url = f"{base_url}/ml/v1/text/generation?version=2023-05-29"
    body = json.dumps({
        "model_id": model_id,
        "project_id": project_id,
        "input": f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>\n",
        "parameters": {
            "decoding_method": "greedy" if temperature < 0.2 else "sample",
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "min_new_tokens": 1,
        },
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as ex:
        detail = ""
        try:
            detail = ex.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise SidecarFallbackError(f"watsonx http {ex.code}: {detail}") from ex
    except (urllib.error.URLError, TimeoutError) as ex:
        raise SidecarFallbackError(str(ex)) from ex

    results = data.get("results") or []
    if not results:
        raise SidecarFallbackError("empty watsonx response")
    return str(results[0].get("generated_text", ""))


def _parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _user_payload(payload: dict[str, Any]) -> str:
    return json.dumps({
        "geometry": payload.get("geometry"),
        "binary": payload.get("binary"),
        "language": (payload.get("language") or "")[:400],
        "tau": payload.get("tau"),
        "m1": payload.get("m1"),
        "m2": payload.get("m2"),
        "m3": payload.get("m3"),
        "prior": payload.get("prior", {}),
    })


def call_m1(payload: dict[str, Any]) -> dict[str, Any]:
    m1 = float(payload.get("m1", 4.2))
    temp = min(1.5, 0.9 + m1 / 10.0)
    raw = _generate(SYSTEM_M1, _user_payload(payload), temp, 512, 8.0)
    parsed = _parse_json(raw)
    exploration = parsed.get("exploration") or raw.strip()[:500]
    explore_scalar = float(parsed.get("explore_scalar", min(1.0, m1 / 10.0)))
    return {
        "exploration": exploration,
        "explore_scalar": explore_scalar,
        "tokens_used": len(raw.split()),
        "raw": raw[:200],
    }


def call_m2(payload: dict[str, Any]) -> dict[str, Any]:
    tau = float(payload.get("tau", 0.3))
    m1 = float(payload.get("m1", 4.2))
    m2 = float(payload.get("m2", 0.55))
    raw = _generate(SYSTEM_M2, _user_payload(payload), 0.1, 128, 5.0)
    parsed = _parse_json(raw)
    c_comp = float(parsed.get("c_comp_proxy", max(0, (m1 / 10.0) - m2)))
    halt = bool(parsed.get("halt", c_comp > tau))
    efficiency = float(parsed.get("efficiency", m2))
    halt_reason = "c_comp_gt_tau" if halt and c_comp > tau else ("model_halt" if halt else None)
    return {
        "efficiency": efficiency,
        "halt": halt,
        "halt_reason": halt_reason,
        "c_comp_proxy": round(c_comp, 4),
        "raw": raw[:200],
    }


def call_m3(payload: dict[str, Any]) -> dict[str, Any]:
    m3 = float(payload.get("m3", 0.99))
    raw = _generate(SYSTEM_M3, _user_payload(payload), 0.3, 256, 5.0)
    parsed = _parse_json(raw)
    prior = payload.get("prior") or {}
    m2_out = prior.get("m2_out") or {}
    chosen = parsed.get("chosen_path", "efficient" if m2_out.get("halt") else "explore")
    meta = float(parsed.get("meta", m3))
    meta = max(0.0, min(1.0, meta))
    return {
        "meta": meta,
        "arbitration": str(parsed.get("arbitration", raw.strip()[:300])),
        "chosen_path": chosen,
        "raw": raw[:200],
    }
