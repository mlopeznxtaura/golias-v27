"""Convert L2 modularfunctionloop JSON arrays → Golias v27 training JSONL.

Input schema (4594.json style):
  id, text, source, url, timestamp, category, dependencies

Output schema (one JSONL line per record):
  geometry, binary, language, triangulation, next_frame, next_token
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

from normalize_jsonl import merge_corpora, normalize_l2_modular_record


def load_json_array(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not text:
        return []
    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [r for r in parsed if isinstance(r, dict)]
    if isinstance(parsed, dict):
        return [parsed]
    raise ValueError(f"Expected JSON array or object in {path}")


def iter_l2_training_rows(records: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    total = len(records)
    for i, rec in enumerate(records):
        nxt = records[i + 1] if i + 1 < total else None
        yield normalize_l2_modular_record(rec, next_rec=nxt, total=total)


def convert_l2_json(
    input_path: Path,
    output_path: Path,
    *,
    append: bool = False,
) -> int:
    records = load_json_array(input_path)
    if not records:
        raise ValueError(f"No records in {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with output_path.open(mode, encoding="utf-8") as out:
        for row in iter_l2_training_rows(records):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="L2 modular JSON → Golias v27 JSONL")
    parser.add_argument(
        "--input",
        default=r"D:\NextAura\L2EVAL\OpenSourceWorldModel-headless\v2L2\modularfunctionloop\4594.json",
        help="Source JSON array (modularfunctionloop)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL (default: data/l2_4594_corpus.jsonl under repo root)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Rebuild goliasv27_corpus.jsonl including L2 shard",
    )
    parser.add_argument("--append", action="store_true", help="Append to output instead of overwrite")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else root / "data" / "l2_4594_corpus.jsonl"

    n = convert_l2_json(input_path, output_path, append=args.append)
    print(f"Converted {n} rows -> {output_path}")

    if args.merge:
        merged = root / "data" / "goliasv27_corpus.jsonl"
        sources = [
            root / "data" / "architecture_doctrine.jsonl",
            root / "data" / "goliasv11_corpus.jsonl",
            output_path,
        ]
        total = merge_corpora(sources, merged)
        print(f"Merged corpus: {total} lines -> {merged}")


if __name__ == "__main__":
    main()
