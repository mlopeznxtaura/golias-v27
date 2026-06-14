"""Golias / NextAura v27 — IF state machine with τ invariant.

Input order: geometry → binary → language
Outputs: of₁ next_frame (224-d), of₂ explanation (language token)
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Optional

import numpy as np
import torch

from dimensions import OF1_VIS as OF1_DIM
BASE_LR = 1e-4


def language_scalar_from_text(text: str) -> float:
    """LNG channel strength from text (0–1); avoids dead semantic τ."""
    t = (text or "").strip()
    if not t:
        return 0.075
    h = hashlib.sha256(t.encode("utf-8")).digest()
    base = int.from_bytes(h[:4], "big") % 10000 / 10000.0
    density = min(1.0, len(t) / 512.0)
    return float(max(0.05, min(1.0, 0.6 * base + 0.4 * density)))


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


def compute_output_alignment(
    language: str,
    geometry: float,
    binary: float,
    of1_scalar: float,
    of1_next_frame: list[float],
    tau: float,
    *,
    m1_exploration: str = "",
    decode_tokens: str = "",
) -> dict[str, Any]:
    """
    Language context judges whether visual next-frame and language outputs agree.
    Mirrors training triangulation: geometry ↔ binary ↔ language must cohere.
    """
    lang_blob = f"{language} {m1_exploration} {decode_tokens}".lower()
    vec = np.asarray(of1_next_frame[:OF1_DIM], dtype=np.float32)
    pred_scalar = float(of1_scalar)
    delta_g = pred_scalar - float(geometry)
    frame_energy = float(np.std(vec)) if vec.size else 0.0
    lang_strength = language_scalar_from_text(language)

    score = 1.0
    notes: list[str] = []

    direction_rules = [
        (r"\bleft\b", -0.04, "LEFT"),
        (r"\bright\b", 0.04, "RIGHT"),
        (r"\bup\b", -0.02, "UP"),
        (r"\bdown\b", 0.02, "DOWN"),
    ]
    for pattern, expected_sign, label in direction_rules:
        if re.search(pattern, lang_blob):
            if expected_sign < 0 and delta_g > 0.015:
                score -= 0.35
                notes.append(f"language says {label} but next-frame geometry rises (ΔG={delta_g:+.4f})")
            elif expected_sign > 0 and delta_g < -0.015:
                score -= 0.35
                notes.append(f"language says {label} but next-frame geometry falls (ΔG={delta_g:+.4f})")

    if lang_strength > 0.25 and frame_energy < 0.0008:
        score -= 0.25
        notes.append("language channel active but frame prediction is flat — visual under-expresses text")

    if not language.strip() and lang_strength < 0.12:
        score -= 0.15
        notes.append("empty language input — cannot verify cross-modal alignment")

    tau_gap = abs(pred_scalar - tau)
    if tau_gap > 0.35:
        score -= 0.1
        notes.append(f"next-frame scalar {pred_scalar:.4f} diverges from τ={tau:.4f}")

    b = float(binary)
    if b > 0.7 and frame_energy < 0.001:
        score -= 0.1
        notes.append("high binary flags but low frame dynamics")

    score = float(max(0.0, min(1.0, score)))
    aligned = score >= 0.62

    if aligned and not notes:
        explanation = (
            f"ALIGNED (score={score:.2f}): language intent, geometry ΔG={delta_g:+.4f}, "
            f"and next-frame latent (σ={frame_energy:.4f}) are coherent at τ={tau:.3f}."
        )
    elif aligned:
        explanation = f"ALIGNED (score={score:.2f}) with caveats: " + "; ".join(notes)
    else:
        explanation = f"MISALIGNED (score={score:.2f}): " + ("; ".join(notes) if notes else "channels disagree")

    return {
        "outputs_aligned": aligned,
        "alignment_score": round(score, 4),
        "alignment_explanation": explanation,
        "alignment_context": {
            "geometry": round(geometry, 4),
            "binary": round(binary, 4),
            "delta_geometry": round(delta_g, 4),
            "frame_energy": round(frame_energy, 6),
            "language_strength": round(lang_strength, 4),
            "tau": round(tau, 4),
        },
    }


def project_outputs(
    pred: torch.Tensor,
    decode: torch.Tensor,
    comp_score: torch.Tensor,
    tau: float,
    geometry: float,
    binary: float,
    language: str,
    *,
    m1_exploration: str = "",
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
    decode_language = " ".join(tokens)

    lang_in = language.strip()[:120]
    if lang_in:
        of2_language = lang_in
        if m1_exploration.strip():
            of2_language = f"{of2_language} | M1: {m1_exploration.strip()[:400]}"
        of2 = f"{of2_language} → {decode_language}"
    else:
        of2_language = m1_exploration.strip()[:400] or decode_language
        of2 = of2_language if not decode_language else f"{of2_language} → {decode_language}"

    # Next-frame summary: scalar + compact preview of 224-d vector
    preview_n = min(12, len(of1))
    next_frame_preview = [round(v, 6) for v in of1[:preview_n]]

    rl_context = ""
    if mismatch:
        rl_context = (
            f"Prediction {next_frame_pred:.4f} vs geometry {geometry:.4f}. "
            f"Retune: increase language weight; τ={tau:.3f}."
        )

    comp = float(comp_score.view(-1)[0].item())
    halt = comp > tau

    alignment = compute_output_alignment(
        language, geometry, binary, next_frame_pred, of1, tau,
        m1_exploration=m1_exploration,
        decode_tokens=decode_language,
    )

    return {
        "of1_next_frame": of1,
        "of1_scalar": round(next_frame_pred, 6),
        "next_frame_scalar": round(next_frame_pred, 6),
        "next_frame_preview": next_frame_preview,
        "of2_explanation": of2,
        "of2_language": of2_language,
        "of2_decode_tokens": decode_language,
        "rl_language_context": rl_context,
        "mismatch": mismatch or not alignment["outputs_aligned"],
        **alignment,
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
