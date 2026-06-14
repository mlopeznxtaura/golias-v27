"""Normalize any JSONL row → geometry → binary → language training record."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Optional, Union

_SCALAR_KEYS = frozenset({"geometry", "binary", "language"})


def _hash_scalar(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") % 10000) / 10000.0


_CATEGORY_BITS = {
    "Scientific_Research": 0.0,
    "Mathematics_Physics": 0.125,
    "Coding": 0.25,
    "Language_Linguistics": 0.375,
    "Human_Behavior_Social_Sciences": 0.5,
    "Engineering_Applied": 0.625,
    "Real_World_Execution": 0.75,
    "Domain_Specific_Misc": 0.875,
    "Reasoning_CoT_Dataset": 0.9,
}


def _l2_triangulation(g: float, b: float, text: str, category: str, dep_count: int) -> float:
    cat_prior = _CATEGORY_BITS.get(category, 0.5)
    density = min(1.0, len(text) / 1200.0)
    dep_signal = min(1.0, dep_count / 5.0)
    return float(max(0.0, min(1.0, 0.35 * (g + b) / 2 + 0.25 * cat_prior + 0.25 * density + 0.15 * dep_signal)))


def _numeric_id(value: Any, fallback: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return fallback


def _binary_flags(category: str, source: str, dep_count: int) -> str:
    flags = []
    if category:
        flags.append(f"domain={category}")
    if source:
        flags.append(f"venue={source}")
    flags.append("has_dependencies" if dep_count else "standalone")
    if category == "Coding":
        flags.append("code_domain")
    if category == "Scientific_Research":
        flags.append("peer_reviewed_science")
    if dep_count >= 3:
        flags.append("multi_hop_graph")
    return ", ".join(flags)


def _l2_language_explanation(
    g: float,
    b: float,
    tri: float,
    id_: int,
    total: int,
    category: str,
    source: str,
    dep_count: int,
    text: str,
) -> str:
    """Natural-language training row: geometry → binary → language content."""
    flags = _binary_flags(category, source, dep_count)
    body = text[:3200].strip()
    return (
        f"geometry G={g:.4f} anchors record {id_} of {total} "
        f"({100 * g:.1f}% through the {category or 'general'} stream). "
        f"binary B={b:.4f} encodes {flags}. "
        f"triangulation T={tri:.4f} measures cross-modal coherence "
        f"between position, domain flags, and text density. "
        f"language: {body}"
    )


def _l2_next_frame_explanation(
    nf: float,
    next_rec: Optional[dict[str, Any]],
    prev_category: str,
    prev_g: float,
) -> str:
    """Language supervision for next frame — not a raw scalar label."""
    if next_rec is None:
        return (
            f"Next frame G={nf:.4f} closes the sequence after {prev_category} "
            f"at G={prev_g:.4f}; halt boundary."
        )
    next_id = _numeric_id(next_rec.get("id"), 0)
    next_cat = str(next_rec.get("category") or "Domain_Specific_Misc")
    next_src = str(next_rec.get("source") or "unknown")
    next_snip = str(next_rec.get("text") or "")[:280].strip()
    delta = nf - prev_g
    return (
        f"Next frame G={nf:.4f} (ΔG={delta:+.4f}) advances to record {next_id}: "
        f"binary shifts into {next_cat} via {next_src}. "
        f"Predicted transition from {prev_category} → {next_cat}. "
        f"Upcoming language: {next_snip}"
    )


def normalize_l2_modular_record(
    rec: dict[str, Any],
    *,
    next_rec: Optional[dict[str, Any]] = None,
    total: int = 1,
) -> dict[str, Any]:
    """Map modularfunctionloop row → geometry → binary → language training record."""
    id_ = _numeric_id(rec.get("id"), 1)
    text = str(rec.get("text") or "").strip()
    category = str(rec.get("category") or "Domain_Specific_Misc").strip()
    source = str(rec.get("source") or "").strip()
    deps = rec.get("dependencies") or []
    dep_count = len(deps) if isinstance(deps, list) else 0

    denom = max(total - 1, 1)
    g = (id_ - 1) / denom
    b = _hash_scalar(f"{category}|{source}|deps={dep_count}")
    tri = _l2_triangulation(g, b, text, category, dep_count)

    language = _l2_language_explanation(
        g, b, tri, id_, total, category, source, dep_count, text
    )

    if next_rec is not None:
        next_id = _numeric_id(next_rec.get("id"), id_ + 1)
        nf = (next_id - 1) / denom
    else:
        nf = min(1.0, id_ / denom)

    nt = _l2_next_frame_explanation(nf, next_rec, category, g)

    return {
        "geometry": round(g, 4),
        "binary": round(b, 4),
        "language": language,
        "triangulation": round(tri, 4),
        "next_frame": round(nf, 4),
        "next_token": nt,
        "source_id": str(rec.get("id", id_)),
        "source_file": "l2_modular",
    }


def _triangulation(g: float, b: float, explanation: str, why_valid: str) -> float:
    """Cross-modal coherence proxy for doctrine rows."""
    le, lw = len(explanation), len(why_valid)
    len_ratio = min(le, lw) / max(le, lw, 1)
    word_overlap = len(set(explanation.lower().split()) & set(why_valid.lower().split()))
    overlap = word_overlap / max(len(set(explanation.split())), 1)
    return float(max(0.0, min(1.0, 0.4 * (g + b) / 2 + 0.3 * len_ratio + 0.3 * overlap)))


def normalize_record(rec: dict[str, Any], *, next_rec: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Map doctrine, HF scalar pack, or legacy rows to unified training schema."""
    if _SCALAR_KEYS.issubset(rec.keys()):
        g = float(rec["geometry"])
        b = float(rec["binary"])
        lang = str(rec["language"])
        tri = float(rec.get("triangulation", (g + b) / 2))
        nf = float(rec.get("next_frame", g))
        nt = str(rec.get("next_token", ""))
        return {
            "geometry": g,
            "binary": b,
            "language": lang,
            "triangulation": tri,
            "next_frame": nf,
            "next_token": nt,
        }

    if "text" in rec and ("category" in rec or "source" in rec):
        raise ValueError(
            "L2 modular records must be normalized via normalize_l2_modular_record() "
            "with sequence context (use l2_json_intake.py)"
        )

    if "explanation" in rec:
        id_ = int(rec.get("id", 1))
        topic = str(rec.get("topic", ""))
        explanation = str(rec["explanation"])
        why_valid = str(rec.get("why_valid", ""))
        n = 40.0
        g = (id_ - 1) / max(n - 1, 1)
        b = _hash_scalar(why_valid)
        tri = _triangulation(g, b, explanation, why_valid)
        if next_rec and "id" in next_rec:
            nf = int(next_rec["id"]) / n
        else:
            nf = min(1.0, (id_ + 1) / n)
        return {
            "geometry": round(g, 4),
            "binary": round(b, 4),
            "language": (
                f"geometry G={g:.4f} frames doctrine item {id_}. "
                f"binary B={b:.4f} fingerprints validation rationale. "
                f"triangulation T={round(tri, 4):.4f} links explanation to why_valid. "
                f"[{topic}] {explanation} "
                f"Why valid: {why_valid}"
            ),
            "triangulation": round(tri, 4),
            "next_frame": round(nf, 4),
            "next_token": (
                f"Next frame G={round(nf, 4):.4f}: advance to "
                f"{str((next_rec or {}).get('topic', topic))}. "
                f"{why_valid[:200]}"
            ),
        }

    raise ValueError(f"Unrecognized JSONL record keys: {list(rec.keys())}")


def iter_normalized(path: Union[str, Path]) -> Iterator[dict[str, Any]]:
    path = Path(path)
    raw = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))
    for i, rec in enumerate(raw):
        nxt = raw[i + 1] if i + 1 < len(raw) else None
        yield normalize_record(rec, next_rec=nxt)


def merge_corpora(
    paths: list[Union[str, Path]],
    output: Union[str, Path],
) -> int:
    """Write merged normalized JSONL; returns line count."""
    output = Path(output)
    count = 0
    with output.open("w", encoding="utf-8") as out:
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            for rec in iter_normalized(p):
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count


def build_default_corpus(root: Union[str, Path]) -> Path:
    root = Path(root)
    out = root / "data" / "goliasv27_corpus.jsonl"
    sources = [
        root / "data" / "architecture_doctrine.jsonl",
        root / "data" / "goliasv11_corpus.jsonl",
        root / "data" / "l2_4594_corpus.jsonl",
    ]
    merge_corpora(sources, out)
    return out
