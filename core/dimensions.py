"""Canonical Golias v27 tensor dimensions — single source for train + inference."""

# Model I/O
D_M1, D_M2_RAW, D_M2, D_M3 = 96, 24, 152, 352
D_PRED = 512          # mlp_pred output / JSONL next_frame target
D_DECODE = 1536         # decode_lm output
OF1_VIS = 224           # visualized next-frame slice (14×16 heatmap)
OF1_VIS_GRID = (14, 16)

# Training defaults (override via env)
DEFAULT_BATCH = 32
DEFAULT_EPOCHS_JSONL = 3
DEFAULT_HF_SAMPLES = 50_000
