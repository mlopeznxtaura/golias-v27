"""Language-first intake: scalars live inside language, not separate form fields.

Training rows use:
  geometry G=… binary B=… triangulation T=… language: <body>
M1/M2/M3/V/if7 are likewise expressed in language when present.
"""
from __future__ import annotations

import hashlib
import re
from typing import Any, Optional

from v27 import language_scalar_from_text

_SCALAR_PATTERNS = {
    "geometry": [
        re.compile(r"\bgeometry\s*G\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\bG\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "binary": [
        re.compile(r"\bbinary\s*B\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\bB\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "triangulation": [
        re.compile(r"\btriangulation\s*T\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\bT\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "m1": [
        re.compile(r"\bM1(?:\s+explore)?\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\bexplore\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "m2": [
        re.compile(r"\bM2(?:\s+effic(?:iency)?)?\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\beffic(?:iency)?\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "m3": [
        re.compile(r"\bM3(?:\s+meta)?\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\bmeta\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "v": [
        re.compile(r"\bV(?:\s+judge)?\s*=\s*([0-9]*\.?[0-9]+)", re.I),
        re.compile(r"\bjudge\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
    "if7": [
        re.compile(r"\bif7\s*=\s*([0-9]*\.?[0-9]+)", re.I),
    ],
}

_DOMAIN_BINARY = {
    "coding": 0.89,
    "code": 0.85,
    "python": 0.88,
    "robot": 0.75,
    "block": 0.72,
    "red": 0.71,
    "move": 0.70,
    "spatial": 0.73,
    "science": 0.35,
    "math": 0.40,
    "finance": 0.16,
    "crime": 0.05,
}


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _hash_scalar(text: str, salt: str = "") -> float:
    h = hashlib.sha256((salt + text).encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") % 10000) / 10000.0


def _first_match(patterns: list[re.Pattern[str]], text: str) -> Optional[float]:
    for pat in patterns:
        m = pat.search(text)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                continue
    return None


def _semantic_body(text: str) -> str:
    """Plain language body after optional 'language:' marker."""
    t = (text or "").strip()
    if not t:
        return ""
    m = re.search(r"\blanguage\s*:\s*(.+)$", t, re.I | re.S)
    if m:
        return m.group(1).strip()
    # Strip scalar header lines if user pasted a full corpus row
    if re.search(r"\bgeometry\s+G\s*=", t, re.I):
        m2 = re.search(r"\blanguage\s*:\s*(.+)$", t, re.I | re.S)
        return m2.group(1).strip() if m2 else t
    return t


def _infer_geometry(body: str, full: str) -> float:
    explicit = _first_match(_SCALAR_PATTERNS["geometry"], full)
    if explicit is not None:
        return _clamp01(explicit)
    blob = f"{body} {full}".lower()
    base = _hash_scalar(body or full, "geometry")
    if re.search(r"\bleft\b", blob):
        return _clamp01(0.48 + 0.08 * base)
    if re.search(r"\bright\b", blob):
        return _clamp01(0.55 + 0.10 * base)
    if re.search(r"\bup\b", blob):
        return _clamp01(0.42 + 0.06 * base)
    if re.search(r"\bdown\b", blob):
        return _clamp01(0.58 + 0.06 * base)
    return _clamp01(0.35 + 0.35 * base)


def _infer_binary(body: str, full: str) -> float:
    explicit = _first_match(_SCALAR_PATTERNS["binary"], full)
    if explicit is not None:
        return _clamp01(explicit)
    blob = f"{body} {full}".lower()
    hits = [_DOMAIN_BINARY[k] for k in _DOMAIN_BINARY if k in blob]
    if hits:
        return _clamp01(sum(hits) / len(hits))
    if re.search(r"\b(domain|venue|flags?)=", blob):
        return _clamp01(0.65 + 0.2 * _hash_scalar(body, "binary"))
    return _clamp01(0.55 + 0.35 * _hash_scalar(body or full, "binary"))


def _infer_triangulation(g: float, b: float, body: str, full: str) -> float:
    explicit = _first_match(_SCALAR_PATTERNS["triangulation"], full)
    if explicit is not None:
        return _clamp01(explicit)
    lng = language_scalar_from_text(body or full)
    density = min(1.0, len(body) / 400.0)
    return _clamp01(0.35 * (g + b) / 2 + 0.35 * lng + 0.30 * density)


def _infer_slider(key: str, body: str, full: str, default: float) -> float:
    parsed = _first_match(_SCALAR_PATTERNS[key], full)
    if parsed is not None:
        if key == "m1":
            return float(parsed)
        return _clamp01(parsed)
    blob = body or full
    if key == "m1":
        return 2.5 + 6.0 * _hash_scalar(blob, "m1")
    if key == "m2":
        return _clamp01(0.35 + 0.5 * _hash_scalar(blob, "m2"))
    if key == "m3":
        return _clamp01(0.7 + 0.25 * _hash_scalar(blob, "m3"))
    if key == "v":
        return _clamp01(0.4 + 0.45 * _hash_scalar(blob, "v"))
    if key == "if7":
        return _clamp01(0.25 + 0.5 * _hash_scalar(blob, "if7"))
    return default


def canonical_language(
    body: str,
    geometry: float,
    binary: float,
    triangulation: float,
    *,
    m1: float,
    m2: float,
    m3: float,
    v: float,
    if7: float,
) -> str:
    """Corpus-shaped record: scalars named in language, semantics in language: block."""
    body = (body or "").strip()
    flags = "interactive_query"
    if re.search(r"\b(code|python|function|class)\b", body, re.I):
        flags = "domain=Coding, code_domain"
    elif re.search(r"\b(block|move|robot|arm|spatial)\b", body, re.I):
        flags = "domain=Real_World_Execution, spatial_command"
    return (
        f"geometry G={geometry:.4f} anchors live query. "
        f"binary B={binary:.4f} encodes {flags}. "
        f"triangulation T={triangulation:.4f} measures cross-modal coherence. "
        f"M1 explore={m1:.2f} M2 effic={m2:.2f} M3 meta={m3:.2f} "
        f"V judge={v:.2f} if7={if7:.2f}. "
        f"language: {body}"
    )


def resolve_language_input(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Accept {language: "..."} or legacy numeric fields.
    Returns decoded scalars + canonical language string for model/sidecars.
    """
    raw = str(
        payload.get("language")
        or payload.get("question")
        or payload.get("text")
        or ""
    ).strip()

    # Legacy API: numbers passed separately — fold into language header
    if not raw and any(k in payload for k in ("geometry", "binary", "m1")):
        g = float(payload.get("geometry", 0.47))
        b = float(payload.get("binary", 0.73))
        tri = payload.get("triangulation")
        tri_f = float(tri) if tri is not None else (g + b) / 2
        m1 = float(payload.get("m1", 4.2))
        m2 = float(payload.get("m2", 0.55))
        m3 = float(payload.get("m3", 0.99))
        v = float(payload.get("V", payload.get("v", 0.58)))
        if7 = float(payload.get("if7", 0.5))
        body = str(payload.get("language_body", "legacy numeric payload"))
        raw = canonical_language(body, g, b, tri_f, m1=m1, m2=m2, m3=m3, v=v, if7=if7)

    body = _semantic_body(raw)
    g = _infer_geometry(body, raw)
    b = _infer_binary(body, raw)
    tri = _infer_triangulation(g, b, body, raw)
    m1 = _infer_slider("m1", body, raw, 4.2)
    m2 = _infer_slider("m2", body, raw, 0.55)
    m3 = _infer_slider("m3", body, raw, 0.99)
    v = _infer_slider("v", body, raw, 0.58)
    if7 = _infer_slider("if7", body, raw, 0.5)

    # Allow explicit API overrides only when language is empty (migration)
    if not body and payload.get("geometry") is not None:
        g = float(payload["geometry"])
        b = float(payload.get("binary", b))
        tri = float(payload.get("triangulation", tri))

    lang = canonical_language(body or raw, g, b, tri, m1=m1, m2=m2, m3=m3, v=v, if7=if7)

    return {
        "geometry": round(g, 4),
        "binary": round(b, 4),
        "triangulation": round(tri, 4),
        "language": lang,
        "language_body": body or raw,
        "m1": round(m1, 4),
        "m2": round(m2, 4),
        "m3": round(m3, 4),
        "V": round(v, 4),
        "if7": round(if7, 4),
        "decoded_from": "language",
    }
