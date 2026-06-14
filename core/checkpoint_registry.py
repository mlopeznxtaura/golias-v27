"""Versioned checkpoint paths, latest-pointer, HF stream offset tracking."""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

CKPT_GLOB = "golias*.pt"
POINTER = "checkpoints/latest.txt"
MANIFEST = "checkpoints/manifest.jsonl"
HF_OFFSET_FILE = "checkpoints/hf_offset.txt"


def _ckpt_dir(root: Path) -> Path:
    d = root / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def list_checkpoints(root: Path) -> list[Path]:
    root = Path(root)
    found: list[Path] = []
    for pattern in (CKPT_GLOB, "checkpoints/" + CKPT_GLOB):
        found.extend(root.glob(pattern))
    # De-dupe; ignore epoch checkpoint files
    uniq = {
        p.resolve(): p
        for p in found
        if p.is_file() and "_checkpoint" not in p.name
    }
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def read_hf_offset(root: Path, default: int = 0) -> int:
    path = Path(root) / HF_OFFSET_FILE
    if not path.is_file():
        return default
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return default


def write_hf_offset(root: Path, offset: int) -> None:
    path = _ckpt_dir(Path(root)) / Path(HF_OFFSET_FILE).name
    path.write_text(str(int(offset)), encoding="utf-8")


def advance_hf_offset(root: Path, trained_samples: int, base_offset: int) -> int:
    """After an HF run, bump stored offset so the next run continues forward."""
    nxt = int(base_offset) + int(trained_samples)
    write_hf_offset(root, nxt)
    return nxt


def write_latest_pointer(root: Path, ckpt_path: Path) -> Path:
    root = Path(root)
    ckpt_dir = _ckpt_dir(root)
    ckpt_path = Path(ckpt_path).resolve()
    ptr = ckpt_dir / "latest.txt"

    def _write_pointer(path: Path) -> None:
        path.write_text(str(ckpt_path), encoding="utf-8")

    try:
        _write_pointer(ptr)
    except PermissionError:
        import subprocess
        subprocess.run(
            ["sudo", "tee", str(ptr)],
            input=str(ckpt_path),
            text=True,
            capture_output=True,
            check=True,
        )
        subprocess.run(["sudo", "chown", f"{os.environ.get('USER', 'ubuntu')}:{os.environ.get('USER', 'ubuntu')}", str(ptr)], check=False)

    for link in (ckpt_dir / "latest.pt", root / "latest.pt"):
        try:
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(ckpt_path)
        except PermissionError:
            import subprocess
            subprocess.run(["sudo", "rm", "-f", str(link)], check=False)
            subprocess.run(["sudo", "ln", "-sf", str(ckpt_path), str(link)], check=True)
    return ptr


def read_latest_pointer(root: Path) -> Optional[Path]:
    root = Path(root)
    for candidate in (
        root / "checkpoints" / "latest.pt",
        root / "latest.pt",
        root / POINTER,
    ):
        if candidate.is_symlink():
            target = candidate.resolve()
            if target.is_file():
                return target
        if candidate.is_file() and candidate.suffix == ".txt":
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                p = Path(text)
                if p.is_file():
                    return p
    return None


def next_versioned_path(root: Path, *, prefix: str = "goliasv") -> Path:
    """Unique output path: checkpoints/goliasv-20260611-183500.pt"""
    root = Path(root)
    out_dir = _ckpt_dir(root)
    name = f"{prefix}-{stamp()}.pt"
    return out_dir / name


LEGACY_BLOCKED = frozenset({
    "goliasv27.pt",
    "goliasv11.pt",
    "goliasv28.pt",
    "latest.pt",
})


def _is_blocked_output(path: Path) -> bool:
    """Root-level legacy names must never receive training output."""
    path = Path(path)
    if path.name in LEGACY_BLOCKED and path.parent.name != "checkpoints":
        return True
    if path.name.endswith("_checkpoint.pt"):
        return True
    return False


def resolve_active_ckpt(root: Path) -> Path:
    """Load order: latest pointer → newest on disk → GOLIAS_CKPT env (legacy)."""
    root = Path(root)
    ptr = read_latest_pointer(root)
    if ptr is not None:
        return ptr
    found = list_checkpoints(root)
    if found:
        return found[0]
    env = os.environ.get("GOLIAS_CKPT", "").strip()
    if env:
        p = Path(env)
        if p.is_file():
            return p
    for legacy in (root / "goliasv28.pt", root / "goliasv27.pt", root / "goliasv11.pt"):
        if legacy.is_file() and not legacy.is_symlink():
            return legacy
    return root / "goliasv28.pt"


def resolve_train_resume(root: Path) -> Path:
    """Resume always from latest pointer unless GOLIAS_FORCE_RESUME=1."""
    root = Path(root)
    if os.environ.get("GOLIAS_FORCE_RESUME", "").lower() in ("1", "true", "yes"):
        explicit = os.environ.get("GOLIAS_RESUME", "").strip()
        if explicit:
            p = Path(explicit)
            if p.is_file():
                return p.resolve()
    return resolve_active_ckpt(root).resolve()


def resolve_train_output(root: Path, resume: Path, output_env: str = "") -> Path:
    """Never write final weights onto resume or legacy root-level names."""
    resume = Path(resume).resolve()
    if output_env:
        out = Path(output_env).resolve()
        if out == resume:
            pass
        elif _is_blocked_output(out):
            print(
                f"  [ckpt] ignoring blocked GOLIAS_OUTPUT={out} — using versioned path",
                flush=True,
            )
        elif not str(out).endswith("_checkpoint.pt"):
            return out
    return next_versioned_path(root)


def append_manifest(
    root: Path,
    *,
    ckpt: Path,
    resume: Path,
    mode: str,
    metrics: list,
    gh_repo: str = "",
) -> None:
    root = Path(root)
    manifest = _ckpt_dir(root) / "manifest.jsonl"
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "ckpt": str(ckpt.resolve()),
        "resume": str(resume.resolve()),
        "mode": mode,
        "gh_repo": gh_repo,
        "metrics": metrics[-3:] if metrics else [],
    }
    with manifest.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
