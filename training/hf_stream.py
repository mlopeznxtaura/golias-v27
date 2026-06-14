"""Stream PhysicalAI HF → language JSONL records → training batches."""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dimensions import DEFAULT_BATCH
from jsonl_corpus_stream import encode_jsonl_record, _stack_batch

PHYSICALAI = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"
BUFFER_SIZE = int(os.environ.get("GOLIAS_BUFFER", "4096"))


def _hash_scalar(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") % 10000) / 10000.0


def _geometry_from_pose(pose_world2cam: list) -> float:
    pose = np.array(pose_world2cam, dtype=np.float64)
    trans = pose[:3, 3] if pose.ndim == 2 else pose[:3]
    return float(np.tanh(np.linalg.norm(trans) / 5.0) * 0.5 + 0.5)


def physicalai_to_record(
    sample: dict[str, Any],
    *,
    prev_geometry: Optional[float] = None,
    index: int = 0,
) -> dict[str, Any]:
    """HF row → language-rich training record (geometry → binary → language)."""
    j = sample.get("json") or sample
    key = str(sample.get("__key__", j.get("camera_name", "scene")))
    cam = j["camera"]
    pose = cam["pose_world2cam"]
    g = _geometry_from_pose(pose)
    fc = int(j.get("frame_count", 1))
    cam_name = str(j.get("camera_name", "camera"))
    b = float(min(1.0, (fc % 100) / 100.0 + 0.25 * (1 if fc > 30 else 0)))
    tri = float((g + b + _hash_scalar(key)) / 3)
    pg = prev_geometry if prev_geometry is not None else g
    nf = float(max(0.0, min(1.0, g + 0.05 * (g - pg))))

    language = (
        f"geometry G={g:.4f} from PhysicalAI scene {key} frame {fc}. "
        f"binary B={b:.4f} flags camera={cam_name}, "
        f"{'long_sequence' if fc > 30 else 'short_sequence'}. "
        f"triangulation T={tri:.4f} links pose to scene identity. "
        f"language: world-model camera view {cam_name} at frame {fc}."
    )
    next_token = (
        f"Next frame G={nf:.4f} (ΔG={nf - g:+.4f}): advance PhysicalAI scene "
        f"{key} — predict camera motion for frame {fc + 1}."
    )
    return {
        "geometry": round(g, 4),
        "binary": round(b, 4),
        "language": language,
        "triangulation": round(tri, 4),
        "next_frame": round(nf, 4),
        "next_token": next_token,
    }


def _load_hf_stream(offset: int = 0):
    from datasets import Features, Sequence, Value, load_dataset

    token = os.environ.get("HF_TOKEN", "")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    features = Features({
        "json": {
            "camera": {
                "distortion": Sequence(Value("float64")),
                "focal_length": Sequence(Sequence(Value("float64"))),
                "pose_world2cam": Sequence(Sequence(Value("float64"))),
                "principal_point": Sequence(Sequence(Value("float64"))),
                "skew": Value("float64"),
            },
            "camera_name": Value("string"),
            "frame_count": Value("int64"),
            "usd_transform": Sequence(Sequence(Value("float64"))),
        },
        "__key__": Value("string"),
        "__url__": Value("string"),
    })
    print(f"Connecting HF stream: {PHYSICALAI} (offset={offset:,})...", flush=True)
    try:
        ds = load_dataset(PHYSICALAI, split="train", streaming=True, features=features)
    except Exception:
        ds = load_dataset(PHYSICALAI, split="train", streaming=True)
    print("HF stream connected.", flush=True)
    return ds


def iter_hf_batches(
    *,
    batch_size: int = DEFAULT_BATCH,
    max_samples: Optional[int] = None,
    offset: int = 0,
) -> Iterator[tuple]:
    """Yield training batches from HF with same schema as JSONL encoder."""
    ds = _load_hf_stream(offset)
    skipped, total, prev_g, buf = 0, 0, None, []
    errors = 0
    if offset > 0:
        print(
            f"  Fast-forwarding HF stream by {offset:,} samples "
            f"(resume cursor — samples are skipped intentionally, not errors)",
            flush=True,
        )

    for sample in ds:
        if max_samples is not None and total >= max_samples:
            break
        if skipped < offset:
            skipped += 1
            if skipped % 5000 == 0:
                print(f"  seek {skipped:,}/{offset:,} into PhysicalAI train split", flush=True)
            continue
        try:
            rec = physicalai_to_record(sample, prev_geometry=prev_g, index=total)
            prev_g = rec["geometry"]
            buf.append(encode_jsonl_record(rec))
            total += 1
        except Exception as ex:
            errors += 1
            if errors <= 3:
                print(f"  [HF skip] {type(ex).__name__}: {ex}", flush=True)
            continue

        if total == 1 or total % 2000 == 0:
            print(f"  HF encoded {total:,} samples", flush=True)
        if len(buf) >= batch_size:
            yield _stack_batch(buf)
            buf = []

    if buf:
        yield _stack_batch(buf)
    print(f"HF stream done: {total:,} samples ({errors} skips)", flush=True)

