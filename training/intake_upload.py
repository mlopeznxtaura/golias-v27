"""Ingest dropped JSON / JSONL datasets into the training corpus."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from l2_json_intake import convert_l2_json
from normalize_jsonl import merge_corpora, normalize_record


def _parse_upload(raw: bytes, filename: str) -> list[dict[str, Any]]:
    text = raw.decode("utf-8-sig", errors="replace").strip()
    if not text:
        return []

    lower = filename.lower()
    if lower.endswith(".jsonl"):
        rows = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Unsupported JSON shape in {filename}")


def _is_l2_modular(rows: list[dict]) -> bool:
    if not rows:
        return False
    sample = rows[0]
    return "text" in sample and ("category" in sample or "source" in sample)


def ingest_bytes(
    raw: bytes,
    filename: str,
    *,
    root: Path,
    merge_master: bool = True,
) -> dict[str, Any]:
    """Save upload, normalize, append to drop corpus, optionally rebuild master corpus."""
    root = Path(root)
    inbox = root / "data" / "inbox"
    inbox.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in filename)[:120]
    saved = inbox / f"{stamp}_{safe}"
    saved.write_bytes(raw)

    rows = _parse_upload(raw, filename)
    drop_corpus = root / "data" / "drop_corpus.jsonl"
    drop_corpus.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    if _is_l2_modular(rows):
        l2_path = inbox / f"{stamp}_l2.json"
        l2_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        added = convert_l2_json(l2_path, drop_corpus, append=True)
    else:
        with drop_corpus.open("a", encoding="utf-8") as out:
            for i, rec in enumerate(rows):
                nxt = rows[i + 1] if i + 1 < len(rows) else None
                try:
                    norm = normalize_record(rec, next_rec=nxt)
                except ValueError:
                    # Already normalized row
                    if all(k in rec for k in ("geometry", "binary", "language", "next_frame")):
                        norm = rec
                    else:
                        raise
                out.write(json.dumps(norm, ensure_ascii=False) + "\n")
                added += 1

    master_lines = None
    if merge_master:
        master = root / "data" / "goliasv27_corpus.jsonl"
        sources = [
            root / "data" / "architecture_doctrine.jsonl",
            root / "data" / "goliasv11_corpus.jsonl",
            root / "data" / "l2_4594_corpus.jsonl",
            drop_corpus,
        ]
        master_lines = merge_corpora([p for p in sources if p.exists()], master)

    return {
        "saved": str(saved),
        "filename": filename,
        "records_added": added,
        "drop_corpus": str(drop_corpus),
        "master_corpus_lines": master_lines,
        "master_corpus": str(root / "data" / "goliasv27_corpus.jsonl"),
    }
