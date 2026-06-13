# Golias v27 — public ledger

**This repo is the public ledger** — architecture doctrine, self-critique corpus, and scalar-pack schema. It is not the runtime.

| What | Where |
|------|--------|
| **Live app** | IBM Cloud — GPU L40S + Code Engine hybrid |
| **Public URL** | https://golias-live.2b02drai9gwy.us-east.codeengine.appdomain.cloud |
| **Checkpoint** | [goliasv11](https://github.com/mlopeznxtaura/goliasv11) (`goliasv11.pt`) |
| **Private ops** | IBM GPU `/opt/golias-v27`, RAWgolias deploy scripts |

## Ledger files

| File | Lines | Role |
|------|-------|------|
| `data/architecture_doctrine.jsonl` | 40 | HF→JSONL pipeline, encoding-not-decoding, τ, world model |
| `data/goliasv11_corpus.jsonl` | 105 | Self-critique from v11 checkpoint metrics |
| `data/goliasv27_corpus.jsonl` | 145 | Merged training corpus (doctrine + critique) |

Each ledger line is either **doctrine** (`id`, `topic`, `explanation`, `why_valid`) or **scalar pack** (`geometry`, `binary`, `language`, `triangulation`, `next_frame`, `next_token`).

## Architecture (v27)

Intelligence is a **τ-controlled state machine**. Input order: **geometry → binary → language**. Outputs: **of₁** next-frame (224-d), **of₂** explanation. Training on scalar packs — no raw video decode.

τ ∈ [0,1] from GEO + BIN + LNG controls halt, LR, export eligibility, active query.

## IBM runtime (not this repo)

On GPU (`goliasv11`, `150.239.211.245`):

- `goliasv27-dash` → `/opt/golias-v27/live/dashboard.py` :8080
- CE `golias-live` proxies via private path → GPU
- Corpus replay: `GOLIAS_JSONL=/opt/golias-v27/data/goliasv27_corpus.jsonl`

`live/` and `training/` in this repo mirror the IBM deploy for audit; production runs on cloud only.

## Rebuild merged corpus

```bash
python training/build_corpus.py   # → data/goliasv27_corpus.jsonl
```

## HF → JSONL (offline ingest, run on IBM GPU)

```bash
python training/hf_to_jsonl.py --dataset nvidia/PhysicalAI-... --max-samples 10000
```

One-time scalar extraction; training consumes JSONL lines immediately.
