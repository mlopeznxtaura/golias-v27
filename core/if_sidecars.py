"""M1/M2/M3 input-function sidecars — contracts, CE fallback, tensor bridge."""
from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np
import torch

IF_BACKEND = os.environ.get("IF_BACKEND", "watsonx").lower()
FALLBACK_URLS = {
    "m1": os.environ.get("IF_FALLBACK_M1", ""),
    "m2": os.environ.get("IF_FALLBACK_M2", ""),
    "m3": os.environ.get("IF_FALLBACK_M3", ""),
}


class SidecarFallbackError(Exception):
    """Watsonx failed; caller should use CE fallback."""


def build_if_payload(
    geometry: float,
    binary: float,
    language: str,
    tau: float,
    m1: float,
    m2: float,
    m3: float,
    v: float = 0.58,
    if7: float = 0.5,
    prior: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    return {
        "geometry": float(geometry),
        "binary": float(binary),
        "language": str(language or ""),
        "tau": float(tau),
        "m1": float(m1),
        "m2": float(m2),
        "m3": float(m3),
        "v": float(v),
        "if7": float(if7),
        "prior": prior or {},
    }


def _hash_vec(text: str, dim: int) -> torch.Tensor:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.tile(arr, (dim + len(arr) - 1) // len(arr))[:dim]
    return torch.from_numpy(arr.astype(np.float32))


def _text_scalar(text: str, salt: str) -> float:
    h = int(hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:8], 16)
    return (h % 10000) / 10000.0


def encode_sidecar_vectors(
    m1_slider: float,
    m2_slider: float,
    m3_slider: float,
    geometry: float,
    binary: float,
    language: str,
    v: float,
    if7: float,
    sidecar_m1: dict[str, Any],
    sidecar_m2: dict[str, Any],
    sidecar_m3: dict[str, Any],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map slider + sidecar outputs into Golias m1/m2/m3 tensors."""
    g, b = float(geometry), float(binary)
    lang = str(language or "").strip()
    explore_text = str(sidecar_m1.get("exploration", lang))
    fused_lang = explore_text if explore_text else lang

    l = _text_scalar(fused_lang[:512], "l")
    m1l = math.copysign(math.log1p(abs(m1_slider)), m1_slider) / 10.0

    m1v = _hash_vec(fused_lang or "empty", 96).unsqueeze(0)
    m1v[:, :32] = b
    m1v[:, 32:64] = l
    m1v[:, 64:96] = float(sidecar_m1.get("explore_scalar", m1l))

    words = fused_lang.lower().split()
    m2_raw = torch.tensor(
        [
            [
                len(fused_lang) / 512.0,
                len(words) / 64.0,
                sum(len(w) for w in words) / max(len(fused_lang), 1),
                b,
                g,
                l,
                v,
                if7,
                float(sidecar_m2.get("efficiency", m2_slider)),
                float(sidecar_m2.get("c_comp_proxy", 0)),
            ]
            + [hash(w) % 1000 / 1000.0 for w in (words[:14] + [""] * 14)[:14]]
        ],
        dtype=torch.float32,
    )[:, :24]
    m2v = torch.cat([_hash_vec(fused_lang + ":m2pad", 128).unsqueeze(0), m2_raw], dim=-1)

    meta = float(sidecar_m3.get("meta", m3_slider))
    base = np.array(
        [m1l, m2_slider, meta, b, g, l, v, if7, 0.0, (m1l + m2_slider + meta) / 3, v * if7],
        dtype=np.float32,
    )
    m3v = torch.zeros(1, 352)
    for j, val in enumerate(base):
        freq = (j + 1) * np.pi / len(base)
        m3v[0] += float(val) * torch.sin(torch.arange(352, dtype=torch.float32) * freq / 352.0 * 2 * np.pi)
    m3v = m3v / (m3v.abs().max() + 1e-8)
    arb = str(sidecar_m3.get("arbitration", ""))[:96]
    if arb:
        m3v[:, :96] = 0.7 * m3v[:, :96] + 0.3 * _hash_vec(arb, 96)

    return m1v.to(device), m2v.to(device), m3v.to(device)


def _ce_post(url: str, payload: dict[str, Any], timeout: float = 10.0) -> dict[str, Any]:
    if not url:
        raise SidecarFallbackError("no CE fallback URL configured")
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/invoke",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as ex:
        raise SidecarFallbackError(str(ex)) from ex


def _call_if(role: str, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Try Watsonx then CE fallback. Returns (result, backend)."""
    if IF_BACKEND == "local":
        import sys
        from pathlib import Path

        sidecars = Path(__file__).resolve().parent.parent / "sidecars"
        if str(sidecars) not in sys.path:
            sys.path.insert(0, str(sidecars))
        from if_rules import rule_response  # noqa: WPS433

        return rule_response(role, payload), "local"

    if IF_BACKEND == "watsonx":
        from watsonx_if import call_m1, call_m2, call_m3  # noqa: WPS433

        callers = {"m1": call_m1, "m2": call_m2, "m3": call_m3}
        try:
            return callers[role](payload), "watsonx"
        except SidecarFallbackError:
            pass

    url = FALLBACK_URLS.get(role, "")
    if url:
        try:
            return _ce_post(url, payload), "ce-fallback"
        except SidecarFallbackError:
            pass

    import sys
    from pathlib import Path

    sidecars = Path(__file__).resolve().parent.parent / "sidecars"
    if str(sidecars) not in sys.path:
        sys.path.insert(0, str(sidecars))
    from if_rules import rule_response  # noqa: WPS433

    return rule_response(role, payload), "local-fallback"


def run_sidecar_pipeline(payload: dict[str, Any]) -> dict[str, Any]:
    """M1 → M2 → M3 on-demand; returns sidecar outputs + backends."""
    backends: dict[str, str] = {}
    m1_out, backends["m1"] = _call_if("m1", payload)

    p2 = {**payload, "prior": {**payload.get("prior", {}), "m1_out": m1_out}}
    m2_out, backends["m2"] = _call_if("m2", p2)

    p3 = {**p2, "prior": {**p2["prior"], "m2_out": m2_out}}
    m3_out, backends["m3"] = _call_if("m3", p3)

    return {
        "m1": m1_out,
        "m2": m2_out,
        "m3": m3_out,
        "backends": backends,
        "halt": bool(m2_out.get("halt", False)),
        "c_comp_proxy": float(m2_out.get("c_comp_proxy", 0)),
    }
