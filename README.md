# Golias v27 — NextAura live application

Intelligence as a **τ-controlled state machine**. Inputs in order: **geometry → binary → language**. Outputs: **of₁** next-frame (224-d) and **of₂** explanation; RL language context on mismatch.

Base checkpoint: [goliasv11](https://github.com/mlopeznxtaura/goliasv11) — download `goliasv11.pt` to repo root before running.

## Quick start (local)

```bash
# 1. Place checkpoint
curl -L -o goliasv11.pt https://github.com/mlopeznxtaura/goliasv11/raw/main/goliasv11.pt

# 2. GPU dashboard
pip install -r requirements.txt
python live/dashboard.py

# 3. JSONL corpus replay (105-line self-critique corpus in data/)
python training/train_from_jsonl.py
```

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/forward` | POST | Full v27: τ, of₁, of₂, RL context |
| `/judge` | POST | `V` human score → `Adapt(θ,V)` signal |
| `/train/jsonl` | POST | Start corpus replay fine-tune (v11→v27) |
| `/log` | GET | Training log poll |

POST body:

```json
{"geometry":0.47,"binary":0.73,"language":"...","triangulation":0.68,"V":0.58}
```

## Architecture

- **τ** — triangulation invariant from GEO + BIN + LNG; controls halt, LR, export eligibility
- **Corpus** — inline `.jsonl` per line (`data/goliasv11_corpus.jsonl`); immediate batch training, no nested HF `.json` buffer
- **CE deploy** — `Dockerfile.proxy` on Code Engine → `GOLIAS_GPU_URL` private path

## IBM hybrid deploy

```bash
docker build -f Dockerfile.gpu -t golias-v27-gpu .
docker build -f Dockerfile.proxy -t golias-v27-proxy .
```

Set `GOLIAS_INTERNAL_KEY`, `GOLIAS_GPU_URL` on CE; `GOLIAS_CKPT=/app/goliasv11.pt` on GPU.
