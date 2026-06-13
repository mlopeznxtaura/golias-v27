# Golias v27 — public ledger

**This repo is the public ledger** — architecture doctrine, self-critique corpus, and scalar-pack schema. It is not the runtime.

| What | Where |
|------|--------|
| **Live app** | IBM Cloud — GPU L40S + Code Engine hybrid |
| **Public URL** | https://golias-live.2b02drai9gwy.us-east.codeengine.appdomain.cloud |
| **Checkpoint** | [goliasv11](https://github.com/mlopeznxtaura/goliasv11) (`goliasv11.pt`) |
| **Private ops** | IBM GPU `/opt/golias-v27`, `deploy/deploy_if_sidecars.ps1` |

## Ledger files

| File | Lines | Role |
|------|-------|------|
| `data/architecture_doctrine.jsonl` | 40 | HF→JSONL pipeline, encoding-not-decoding, τ, world model |
| `data/goliasv11_corpus.jsonl` | 105 | Self-critique from v11 checkpoint metrics |
| `data/goliasv27_corpus.jsonl` | 145 | Merged training corpus (doctrine + critique) |

## Architecture (v27)

Intelligence is a **τ-controlled state machine**. Input order: **geometry → binary → language**. Outputs: **of₁** next-frame (224-d), **of₂** explanation.

**M1/M2/M3 sidecars** (interim, until weights baked in):

| IF | Role | Primary | Fallback |
|----|------|---------|----------|
| M1 | Dreamer / explore | Watsonx | CE `golias-if-m1` (min-scale 0) |
| M2 | Efficiency / halt | Watsonx | CE `golias-if-m2` |
| M3 | Arbitration | Watsonx | CE `golias-if-m3` |

### IF request (all sidecars)

```json
{
  "geometry": 0.47, "binary": 0.73, "language": "...",
  "tau": 0.30, "m1": 4.2, "m2": 0.55, "m3": 0.99,
  "prior": { "m1_out": {}, "m2_out": {} }
}
```

### IF responses

- **M1:** `{ "exploration", "explore_scalar", "tokens_used" }`
- **M2:** `{ "efficiency", "halt", "c_comp_proxy" }`
- **M3:** `{ "meta", "arbitration", "chosen_path": "explore|efficient" }`

### `IF_BACKEND` modes (GPU env)

| Value | Behavior |
|-------|----------|
| `watsonx` | Watsonx first; CE fallback on timeout/429/5xx; **local-fallback** if CE URLs unset |
| `local` | Rule-based only (no API calls) — for baked-in checkpoint testing |
| (fallback URLs) | `IF_FALLBACK_M1/M2/M3` → CE app `/invoke` |

GPU secrets: `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, `WATSONX_URL`, `WATSONX_MODEL`.

## IBM runtime

- `goliasv27-dash` → `/opt/golias-v27/live/dashboard.py` :8080
- CE `golias-live` → v27 UI + proxy to GPU
- CE `golias-if-m1/m2/m3` → scale-to-zero fallback only

```powershell
.\deploy\deploy_if_sidecars.ps1
```

## Corpus / HF ingest

```bash
python training/build_corpus.py
python training/hf_to_jsonl.py --dataset nvidia/PhysicalAI-... --max-samples 10000
```
