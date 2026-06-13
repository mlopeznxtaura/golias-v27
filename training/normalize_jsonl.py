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
            "language": f"[{topic}] {explanation}",
            "triangulation": round(tri, 4),
            "next_frame": round(nf, 4),
            "next_token": topic or why_valid[:120],
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
    ]
    merge_corpora(sources, out)
    return out
