"""Hybrid retrain: JSONL corpus (language explanations) + HF PhysicalAI stream.

GOLIAS_TRAIN_MODE=jsonl|hf|hybrid (default hybrid)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from checkpoint_io import GRAD_CLIP, save_checkpoint, save_model
from dimensions import D_PRED, DEFAULT_BATCH, DEFAULT_EPOCHS_JSONL, DEFAULT_HF_SAMPLES
from hf_stream import iter_hf_batches
from jsonl_corpus_stream import iter_jsonl_batches
from publish_checkpoint import publish_checkpoint
from checkpoint_registry import (
    advance_hf_offset,
    append_manifest,
    read_hf_offset,
    resolve_train_output,
    resolve_train_resume,
    write_latest_pointer,
)

BATCH_SIZE = int(os.environ.get("GOLIAS_BATCH", str(DEFAULT_BATCH)))
JSONL_EPOCHS = int(os.environ.get("GOLIAS_JSONL_EPOCHS", str(DEFAULT_EPOCHS_JSONL)))
HF_SAMPLES = int(os.environ.get("GOLIAS_HF_SAMPLES", str(DEFAULT_HF_SAMPLES)))
HF_OFFSET = int(os.environ.get("GOLIAS_HF_OFFSET", "0"))
LR = float(os.environ.get("GOLIAS_LR", "5e-5"))
MODE = os.environ.get("GOLIAS_TRAIN_MODE", "hybrid").lower()
JSONL_PATH = os.environ.get("GOLIAS_JSONL", str(ROOT / "data" / "goliasv27_corpus.jsonl"))
CHECKPOINT = os.environ.get("GOLIAS_CHECKPOINT", str(ROOT / "checkpoints" / "rolling_checkpoint.pt"))
RL_WEIGHT = float(os.environ.get("GOLIAS_RL_WEIGHT", "0.5"))
LOG_PATH = os.environ.get("GOLIAS_LOG", str(ROOT / "train.jsonl.log"))


def _run_batches(model, opt, device, batches, phase: str, metrics: list, step: list):
    ep_loss, n = 0.0, 0
    for m1, m2, m3, nf_tgt, nt_tgt, rl_mask in batches:
        m1, m2, m3 = m1.to(device), m2.to(device), m3.to(device)
        nf_tgt, nt_tgt, rl_mask = nf_tgt.to(device), nt_tgt.to(device), rl_mask.to(device)
        out = model(m1, m2, m3)
        l_frame = F.mse_loss(out["pred"], nf_tgt)
        l_token = F.mse_loss(out["decode"][:, :D_PRED], nt_tgt)
        rl = rl_mask.mean().clamp(min=1e-6)
        loss = l_frame + RL_WEIGHT * rl * l_token + 0.01 * out["comp_score"].abs().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        ep_loss += loss.item()
        n += 1
        step[0] += 1
        if step[0] % 10 == 0:
            print(
                f"  [{phase}] step {step[0]} L_frame={l_frame.item():.4f} L_token={l_token.item():.4f}",
                flush=True,
            )
    avg = ep_loss / max(n, 1)
    metrics.append({"phase": phase, "loss": avg, "batches": n})
    print(f"  [{phase}] avg loss={avg:.4f} ({n} batches)", flush=True)
    return avg


def train():
    from goliasv7_torch import GoliasV7Torch

    try:
        sys.path.insert(0, str(ROOT / "core"))
        from secrets_loader import ensure_secrets
        ensure_secrets()
    except ImportError:
        pass

    # Strip dangerous env inherited from systemd/env.sh
    for key in ("GOLIAS_OUTPUT",):
        val = os.environ.get(key, "")
        if val and Path(val).name in ("goliasv27.pt", "goliasv28.pt", "goliasv11.pt"):
            os.environ.pop(key, None)

    resume_path = resolve_train_resume(ROOT)
    output_env = os.environ.get("GOLIAS_OUTPUT", "")
    output_path = resolve_train_output(ROOT, resume_path, output_env)
    hf_offset = HF_OFFSET if HF_OFFSET > 0 else read_hf_offset(ROOT, 70000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print(f"GOLIAS HYBRID RETRAIN mode={MODE}", flush=True)
    print(f"  JSONL: {JSONL_PATH}", flush=True)
    print(f"  HF: PhysicalAI samples={HF_SAMPLES} stream_offset={hf_offset:,}", flush=True)
    print(f"    (offset fast-forwards the HF stream — not failed samples)", flush=True)
    print(f"  D_PRED={D_PRED} | batch={BATCH_SIZE}", flush=True)
    print(f"  resume: {resume_path}", flush=True)
    print(f"  output: {output_path}", flush=True)
    if output_path.resolve() == resume_path.resolve():
        raise RuntimeError("refusing to overwrite resume checkpoint — set GOLIAS_OUTPUT")
    print(f"  device: {device}", flush=True)
    print("=" * 70, flush=True)

    model = GoliasV7Torch()
    if resume_path.is_file():
        model.load_checkpoint(str(resume_path))
        print(f"Loaded warm-started weights from {resume_path}", flush=True)
    model = model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    metrics: list = []
    step = [0]

    if MODE in ("jsonl", "hybrid"):
        for epoch in range(JSONL_EPOCHS):
            print(f"\n--- JSONL epoch {epoch + 1}/{JSONL_EPOCHS} ---", flush=True)
            batches = iter_jsonl_batches(JSONL_PATH, batch_size=BATCH_SIZE, shuffle=True)
            _run_batches(model, opt, device, batches, f"jsonl_e{epoch+1}", metrics, step)
            save_checkpoint(model, CHECKPOINT, metrics, epoch + 1)

    if MODE in ("hf", "hybrid"):
        print(f"\n--- HF PhysicalAI stream ({HF_SAMPLES:,} samples) ---", flush=True)
        batches = iter_hf_batches(
            batch_size=BATCH_SIZE,
            max_samples=HF_SAMPLES,
            offset=hf_offset,
        )
        _run_batches(model, opt, device, batches, "hf_stream", metrics, step)
        save_checkpoint(model, CHECKPOINT, metrics, JSONL_EPOCHS + 1)
        new_offset = advance_hf_offset(ROOT, HF_SAMPLES, hf_offset)
        print(f"  HF offset advanced: {hf_offset:,} → {new_offset:,} for next run", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, str(output_path), metrics, f"golias hybrid {MODE}")
    write_latest_pointer(ROOT, output_path)
    gh_url = publish_checkpoint(
        output_path,
        root=ROOT,
        train_mode=MODE,
        metrics_summary=str(metrics[-1] if metrics else {}),
    )
    append_manifest(
        ROOT,
        ckpt=output_path,
        resume=resume_path,
        mode=MODE,
        metrics=metrics,
        gh_repo=gh_url,
    )
    print(f"DONE -> {output_path}", flush=True)
    print(f"LATEST -> {output_path} (pointer updated)", flush=True)


if __name__ == "__main__":
    train()
