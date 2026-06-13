"""Fine-tune goliasv11 from inline JSONL — v27 corpus replay loop."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from checkpoint_io import GRAD_CLIP, save_checkpoint, save_model
from jsonl_corpus_stream import D_OUT, iter_jsonl_batches

BATCH_SIZE = int(os.environ.get("GOLIAS_BATCH", "16"))
NUM_EPOCHS = int(os.environ.get("GOLIAS_EPOCHS", "20"))
LR = float(os.environ.get("GOLIAS_LR", "5e-5"))
JSONL_PATH = os.environ.get("GOLIAS_JSONL", str(ROOT / "data" / "goliasv27_corpus.jsonl"))
RESUME = os.environ.get("GOLIAS_RESUME", str(ROOT / "goliasv11.pt"))
OUTPUT = os.environ.get("GOLIAS_OUTPUT", str(ROOT / "goliasv27.pt"))
CHECKPOINT = str(ROOT / "goliasv27_checkpoint.pt")
RL_WEIGHT = float(os.environ.get("GOLIAS_RL_WEIGHT", "0.5"))
LOG_PATH = os.environ.get("GOLIAS_LOG", str(ROOT / "train.jsonl.log"))


def train():
    from goliasv7_torch import GoliasV7Torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70, flush=True)
    print("v27 JSONL corpus replay (geometry → binary → language)", flush=True)
    print(f"  corpus: {JSONL_PATH}", flush=True)
    print(f"  resume: {RESUME}", flush=True)
    print(f"  device: {device}", flush=True)
    print("=" * 70, flush=True)

    model = GoliasV7Torch()
    if os.path.isfile(RESUME):
        model.load_checkpoint(RESUME)
    model = model.to(device)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    metrics, step = [], 0

    for epoch in range(NUM_EPOCHS):
        ep_loss, n = 0.0, 0
        for m1, m2, m3, nf_tgt, nt_tgt, rl_mask in iter_jsonl_batches(
            JSONL_PATH, batch_size=BATCH_SIZE, shuffle=True
        ):
            m1, m2, m3 = m1.to(device), m2.to(device), m3.to(device)
            nf_tgt, nt_tgt, rl_mask = nf_tgt.to(device), nt_tgt.to(device), rl_mask.to(device)
            out = model(m1, m2, m3)
            l_frame = F.mse_loss(out["pred"], nf_tgt)
            l_token = F.mse_loss(out["decode"][:, :D_OUT], nt_tgt)
            rl = rl_mask.mean().clamp(min=1e-6)
            loss = l_frame + RL_WEIGHT * rl * l_token + 0.01 * out["comp_score"].abs().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            ep_loss += loss.item()
            n += 1
            step += 1
            if step % 5 == 0:
                print(
                    f"  step {step} L_frame={l_frame.item():.4f} L_token={l_token.item():.4f}",
                    flush=True,
                )
        avg = ep_loss / max(n, 1)
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}: loss={avg:.4f}", flush=True)
        metrics.append({"epoch": epoch + 1, "loss": avg})
        save_checkpoint(model, CHECKPOINT, metrics, epoch + 1)

    save_model(model, OUTPUT, metrics, "goliasv27 JSONL")
    print(f"DONE -> {OUTPUT}", flush=True)


if __name__ == "__main__":
    train()
