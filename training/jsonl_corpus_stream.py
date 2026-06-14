"""Inline JSONL corpus → geometry → binary → language → train batches.

Each line:
  geometry, binary, language (inputs, in order)
  triangulation (aux)
  next_frame (supervision target)
  next_token (RL / language correction when mismatch)
"""
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dimensions import D_M1, D_M2_RAW, D_M2, D_M3, D_PRED as D_OUT


def _hash_text(text: str, dim: int) -> np.ndarray:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
    arr = (arr / 127.5) - 1.0
    arr = np.tile(arr, (dim + len(arr) - 1) // len(arr))[:dim]
    return arr.astype(np.float32)


def _expand_base(base: np.ndarray, dim: int) -> np.ndarray:
    positions = np.arange(dim, dtype=np.float32)
    out = np.zeros(dim, dtype=np.float32)
    for j, val in enumerate(base):
        freq = (j + 1) * math.pi / max(len(base), 1)
        out += float(val) * np.sin(freq * positions / dim * 2 * math.pi)
    mx = np.abs(out).max()
    return (out / (mx + 1e-8)).astype(np.float32)


from normalize_jsonl import normalize_record  # noqa: E402


def encode_jsonl_record(rec: dict):
    """Map one JSONL row to model inputs + targets (geometry → binary → language)."""
    rec = normalize_record(rec)
    g = float(rec["geometry"])
    b = float(rec["binary"])
    lang = str(rec.get("language", ""))
    tri = float(rec.get("triangulation", (g + b) / 2))
    nf = float(rec["next_frame"])
    nt = str(rec.get("next_token", ""))

    # 1 geometry → m2
    m2_raw = np.array([g, tri, b, nf, g * b, tri * nf, g - b, b - g] * 3, dtype=np.float32)[:D_M2_RAW]
    m2_pad = _hash_text(f"geom|{g:.4f}|{tri:.4f}", 128)
    m2 = np.concatenate([m2_pad, m2_raw]).astype(np.float32)

    # 2 binary → m1
    m1 = _hash_text(f"bin|{b:.4f}", D_M1)
    m1[:32] = b
    m1[32:64] = tri

    # 3 language → m3
    m3_base = np.array([g, b, tri, nf, len(lang) / 512.0, lang.count(" ") / 64.0], dtype=np.float32)
    m3 = _expand_base(m3_base, D_M3)
    lang_vec = _hash_text(lang, 96)
    m3[:96] = 0.7 * m3[:96] + 0.3 * lang_vec

    # Targets
    next_frame_tgt = _expand_base(np.array([nf, g, b, tri, nf - g], dtype=np.float32), D_OUT)
    next_token_tgt = _expand_base(
        np.array([_hash_text(nt, 8)[:8].mean(), b, g, nf, tri], dtype=np.float32), D_OUT
    )
    rl_active = float(abs(nf - g) > 0.05 or len(nt) > 0)

    return m1, m2, m3, next_frame_tgt, next_token_tgt, rl_active


def iter_jsonl_batches(
    path: Union[str, Path],
    batch_size: int = 64,
    shuffle: bool = True,
    limit: Optional[int] = None,
):
    """Yield (m1, m2, m3, next_frame, next_token, rl_mask) tensors per batch."""
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break

    if shuffle:
        rng = np.random.default_rng(int(os.environ.get("GOLIAS_SEED", "11")))
        rng.shuffle(rows)

    buf = []
    for rec in rows:
        buf.append(encode_jsonl_record(rec))
        if len(buf) >= batch_size:
            yield _stack_batch(buf)
            buf = []
    if buf:
        yield _stack_batch(buf)


def _stack_batch(buf):
    m1, m2, m3, nf, nt, rl = zip(*buf)
    return (
        torch.from_numpy(np.stack(m1).astype(np.float32)),
        torch.from_numpy(np.stack(m2).astype(np.float32)),
        torch.from_numpy(np.stack(m3).astype(np.float32)),
        torch.from_numpy(np.stack(nf).astype(np.float32)),
        torch.from_numpy(np.stack(nt).astype(np.float32)),
        torch.tensor(rl, dtype=torch.float32),
    )
