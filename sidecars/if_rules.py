"""Deterministic M1/M2/M3 rules — CE fallback and IF_BACKEND=local."""
from __future__ import annotations

from typing import Any


def rule_response(role: str, payload: dict[str, Any]) -> dict[str, Any]:
    m1 = float(payload.get("m1", 4.2))
    m2 = float(payload.get("m2", 0.55))
    m3 = float(payload.get("m3", 0.99))
    tau = float(payload.get("tau", 0.3))
    lang = str(payload.get("language", ""))[:400]
    prior = payload.get("prior") or {}

    if role == "m1":
        return {
            "exploration": f"[M1 explore={m1}] Continue: {lang or 'scalar dynamics'}",
            "explore_scalar": min(1.0, m1 / 10.0),
            "tokens_used": 0,
        }

    if role == "m2":
        c_comp = max(0.0, (m1 / 10.0) - m2)
        halt = c_comp > tau
        halt_reason = "c_comp_gt_tau" if halt else None
        return {
            "efficiency": m2,
            "halt": halt,
            "halt_reason": halt_reason,
            "c_comp_proxy": round(c_comp, 4),
        }

    m1_out = prior.get("m1_out") or {}
    m2_out = prior.get("m2_out") or {}
    halt = bool(m2_out.get("halt", False))
    return {
        "meta": min(1.0, m3 * 0.85),
        "arbitration": f"M1={m1_out.get('explore_scalar', '?')} M2 halt={halt}",
        "chosen_path": "efficient" if halt else "explore",
    }
