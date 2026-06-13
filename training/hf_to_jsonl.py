"""Hugging Face dataset → inline JSONL scalar packs (one line per frame).

Runs offline once per dataset. Training consumes JSONL directly — no pixel decode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize_jsonl import normalize_record

PHYSICALAI = "nvidia/PhysicalAI-WorldModel-Synthetic-Physical-Interaction-Scenes"


def _hash_scalar(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(h[:4], "big") % 10000) / 10000.0


def _geometry_from_pose(pose_world2cam: list) -> float:
    pose = np.array(pose_world2cam, dtype=np.float64)
    trans = pose[:3, 3] if pose.ndim == 2 else pose[:3]
    return float(np.tanh(np.linalg.norm(trans) / 5.0) * 0.5 + 0.5)


def physicalai_sample_to_record(sample: dict[str, Any], prev_geometry: Optional[float] = None) -> dict:
    j = sample.get("json") or sample
    cam = j["camera"]
    pose = cam["pose_world2cam"]
    g = _geometry_from_pose(pose)
    fc = float(j.get("frame_count", 1))
    b = float(min(1.0, (fc % 100) / 100.0 + 0.25 * (1 if fc > 30 else 0)))
    lang = str(j.get("camera_name") or sample.get("__key__", "scene"))
    tri = float((g + b + _hash_scalar(lang)) / 3)
    nf = g + 0.05 * (g - (prev_geometry if prev_geometry is not None else g))
    nf = float(max(0.0, min(1.0, nf)))
    return normalize_record({
        "geometry": round(g, 4),
        "binary": round(b, 4),
        "language": lang,
        "triangulation": round(tri, 4),
        "next_frame": round(nf, 4),
        "next_token": f"frame_{int(fc)}",
    })


def text_label_sample_to_record(sample: dict[str, Any], text_field: str, label_field: str) -> dict:
    label = str(sample.get(label_field, sample.get("label", "")))
    text = str(sample.get(text_field, label))
    g = _hash_scalar(label or text)
    b = float(len(label.split()) % 2)
    tri = float((g + b + _hash_scalar(text)) / 3)
    direction = 1.0 if "left" in label.lower() else (-1.0 if "right" in label.lower() else 0.0)
    nf = float(max(0.0, min(1.0, g + 0.08 * direction)))
    return normalize_record({
        "geometry": round(g, 4),
        "binary": round(b, 4),
        "language": text or label,
        "triangulation": round(tri, 4),
        "next_frame": round(nf, 4),
        "next_token": label[:120],
    })


def convert(
    dataset: str,
    output: Path,
    *,
    split: str = "train",
    max_samples: int = 0,
    text_field: str = "sentence",
    label_field: str = "label",
    streaming: bool = True,
) -> int:
    from datasets import Features, Sequence, Value, load_dataset

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    prev_g: Optional[float] = None

    if dataset == PHYSICALAI or "PhysicalAI" in dataset:
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
        token = os.environ.get("HF_TOKEN", "")
        if token:
            os.environ["HF_TOKEN"] = token
        ds = load_dataset(dataset, split=split, streaming=streaming, features=features)
        iterator = ds
        writer = output.open("w", encoding="utf-8")
        try:
            for sample in iterator:
                try:
                    rec = physicalai_sample_to_record(sample, prev_g)
                    prev_g = rec["geometry"]
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                except Exception:
                    continue
                if max_samples and count >= max_samples:
                    break
                if count and count % 5000 == 0:
                    print(f"  {count:,} scalar packs written", flush=True)
        finally:
            writer.close()
    else:
        ds = load_dataset(dataset, split=split, streaming=streaming)
        writer = output.open("w", encoding="utf-8")
        try:
            for sample in ds:
                try:
                    rec = text_label_sample_to_record(sample, text_field, label_field)
                    writer.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    count += 1
                except Exception:
                    continue
                if max_samples and count >= max_samples:
                    break
                if count and count % 5000 == 0:
                    print(f"  {count:,} scalar packs written", flush=True)
        finally:
            writer.close()

    print(f"DONE {count:,} lines -> {output}", flush=True)
    return count


def main():
    p = argparse.ArgumentParser(description="HF dataset → JSONL scalar packs")
    p.add_argument("--dataset", default=PHYSICALAI)
    p.add_argument("--split", default="train")
    p.add_argument("--output", default=str(ROOT / "data" / "hf_scalars.jsonl"))
    p.add_argument("--max-samples", type=int, default=0, help="0 = unlimited")
    p.add_argument("--text-field", default="sentence")
    p.add_argument("--label-field", default="label")
    args = p.parse_args()
    convert(
        args.dataset,
        Path(args.output),
        split=args.split,
        max_samples=args.max_samples,
        text_field=args.text_field,
        label_field=args.label_field,
    )


if __name__ == "__main__":
    main()
