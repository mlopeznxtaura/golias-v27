"""Build merged v27 training corpus from doctrine + self-critique JSONL."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from normalize_jsonl import build_default_corpus

if __name__ == "__main__":
    out = build_default_corpus(ROOT)
    n = sum(1 for _ in out.open(encoding="utf-8"))
    print(f"Built {n} lines -> {out}")
