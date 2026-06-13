"""Golias / NextAura v27 — IF state machine with τ invariant.

Input order: geometry → binary → language
Outputs: of₁ next_frame (224-d), of₂ explanation (language token)
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Optional

import numpy as np
import torch

OF1_DIM = 224
BASE_LR = 1e-4


def compute_tau(geometry: float, binary: float, language: float) -> float:
    """τ from triangulation of GEO, BIN, LNG — single decision invariant."""
    g, b, l = float(geometry), float(binary), float(language)
    tri = (g + b + l) / 3.0
    # Harmonic blend: penalize dead language channel
    denom = g + b + max(l, 1e-6)
    harmonic = (3.0 * g * b * max(l, 0.01)) ** (1.0 / 3.0) if denom > 0 else 0.0
    tau = 0.5 * tri + 0.5 * harmonic
    return float(max(0.0, min(1.0, tau)))


def lr_from_tau(tau: float) -> float:
    return BASE_LR * tau


def encode_if_record(
    geometry: float,
    binary: float,
    language: str,
    triangulation: Optional[float] = None,
) -> dict[str, Any]:
    """Build JSONL-compatible record for streaming encoder (geometry → binary → language)."""
    g, b = float(geometry), float(binary)
    lang = str(language or "")
    tri = float(triangulation if triangulation is not None else (g + b) / 2)
    return {
        "geometry": g,
        "binary": b,
        "language": lang,
        "triangulation": tri,
        "next_frame": g,  # placeholder until model predicts
        "next_token": "",
    }


def project_outputs(
    pred: torch.Tensor,
    decode: torch.Tensor,
    comp_score: torch.Tensor,
    tau: float,
    geometry: float,
    binary: float,
    language: str,
) -> dict[str, Any]:
    """of₁ = F_pred, of₂ = F_explain; RL context on mismatch."""
    p = pred.detach().float().view(-1)
    d = decode.detach().float().view(-1)

    if p.numel() >= OF1_DIM:
        of1 = p[:OF1_DIM].cpu().numpy().tolist()
    else:
        of1 = torch.nn.functional.pad(p, (0, OF1_DIM - p.numel())).cpu().numpy().tolist()

    of1_scalar = float(np.mean(of1))
    next_frame_pred = of1_scalar
    mismatch = abs(next_frame_pred - geometry) > 0.05

    top = torch.topk(d.abs(), min(8, d.numel()))
    tokens = [f"d{int(i)}" for i in top.indices.tolist()]
    if language.strip():
        of2 = f"{language.strip()[:120]} → " + " ".join(tokens)
    else:
        of2 = " ".join(tokens)

    rl_context = ""
    if mismatch:
        rl_context = (
            f"Prediction {next_frame_pred:.4f} vs geometry {geometry:.4f}. "
            f"Retune: increase language weight; τ={tau:.3f}."
        )

    comp = float(comp_score.view(-1)[0].item())
    halt = comp > tau

    return {
        "of1_next_frame": of1,
        "of1_scalar": round(next_frame_pred, 6),
        "of2_explanation": of2,
        "rl_language_context": rl_context,
        "mismatch": mismatch,
        "tau": round(tau, 6),
        "triangulation": round((geometry + binary) / 2, 6),
        "c_comp": round(comp, 6),
        "halt": halt,
        "export_eligible": tau > 0.85,
        "active_query": tau < 0.30,
        "sidecar_reject": tau < 0.20,
    }


def explanation_from_jsonl_next_token(next_token: str, of1_scalar: float) -> str:
    nt = (next_token or "").strip()
    if not nt:
        return f"next_frame={of1_scalar:.4f}"
    return f"{nt} (pred={of1_scalar:.4f})"
