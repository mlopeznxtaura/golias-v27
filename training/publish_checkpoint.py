"""Publish a single .pt checkpoint to a new GitHub repo (one repo per save)."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from checkpoint_registry import stamp


def _gh_env() -> dict[str, str]:
    env = os.environ.copy()
    token = (
        os.environ.get("GH_TOKEN", "").strip()
        or os.environ.get("GITHUB_TOKEN", "").strip()
    )
    if token:
        env["GH_TOKEN"] = token
        env["GITHUB_TOKEN"] = token
    return env


def _ensure_gh_auth() -> bool:
    """Authenticate gh from GH_TOKEN / GITHUB_TOKEN if not already logged in."""
    if shutil.which("gh") is None:
        return False
    env = _gh_env()
    if _run(["gh", "auth", "status"], Path.cwd(), env=env).returncode == 0:
        return True
    token = env.get("GH_TOKEN", "")
    if not token:
        return False
    proc = subprocess.run(
        ["gh", "auth", "login", "--with-token"],
        input=token,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    return proc.returncode == 0


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        env=env or _gh_env(),
    )


def gh_available() -> bool:
    try:
        from secrets_loader import ensure_secrets
        ensure_secrets()
    except ImportError:
        pass
    return shutil.which("gh") is not None and _ensure_gh_auth()


def publish_checkpoint(
    ckpt_path: Path,
    *,
    root: Path,
    train_mode: str = "hybrid",
    metrics_summary: str = "",
) -> str:
    """Create a fresh GitHub repo containing only the .pt file. Returns repo URL or ''."""
    if os.environ.get("GOLIAS_GH_PUBLISH", "1").lower() in ("0", "false", "no"):
        return ""

    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.is_file():
        print(f"  [gh publish] skip — missing {ckpt_path}", flush=True)
        return ""

    if not gh_available():
        print(
            "  [gh publish] skip — load GH_TOKEN from IBM Secrets Manager "
            "(sync_all_sm_to_gpu.ps1 → /run/golias/secrets.env)",
            flush=True,
        )
        return ""

    org = os.environ.get("GOLIAS_GH_ORG", "").strip()
    visibility = os.environ.get("GOLIAS_GH_VISIBILITY", "public").lower()
    if visibility not in ("public", "private"):
        visibility = "public"

    short = ckpt_path.stem[:32]
    repo_name = f"golias-ckpt-{stamp()}-{short}".lower()
    repo_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in repo_name)[:80]

    with tempfile.TemporaryDirectory(prefix="golias-ckpt-") as tmp:
        work = Path(tmp)
        dest = work / ckpt_path.name
        shutil.copy2(ckpt_path, dest)
        readme = work / "README.md"
        readme.write_text(
            f"# Golias checkpoint\n\n"
            f"- **file**: `{ckpt_path.name}`\n"
            f"- **mode**: {train_mode}\n"
            f"- **metrics**: {metrics_summary or 'n/a'}\n\n"
            f"Model weights only — no training code in this repo.\n",
            encoding="utf-8",
        )
        gitignore = work / ".gitignore"
        gitignore.write_text("*\n!README.md\n!*.pt\n", encoding="utf-8")

        for cmd in (
            ["git", "init", "-b", "main"],
            ["git", "config", "user.email", "golias@checkpoint.local"],
            ["git", "config", "user.name", "Golias Checkpoint"],
            ["git", "add", ckpt_path.name, "README.md", ".gitignore"],
            ["git", "commit", "-m", f"checkpoint {ckpt_path.name}"],
        ):
            r = _run(cmd, work)
            if r.returncode != 0 and "commit" in cmd:
                print(f"  [gh publish] git commit failed: {r.stderr}", flush=True)
                return ""

        full_name = f"{org}/{repo_name}" if org else repo_name
        # Prefer SSH push (GPU deploy key / account SSH key — same as Mac workflow)
        ssh_url = f"git@github.com:{full_name}.git" if org else None
        create = ["gh", "repo", "create", full_name, f"--{visibility}", "--source=.", "--remote=origin", "--push"]
        r = _run(create, work)
        if r.returncode != 0 and not org:
            create = ["gh", "repo", "create", repo_name, f"--{visibility}", "--source=.", "--remote=origin", "--push"]
            r = _run(create, work)
        if r.returncode != 0:
            print(f"  [gh publish] failed: {r.stderr or r.stdout}", flush=True)
            return ""
        if ssh_url:
            _run(["git", "remote", "set-url", "origin", ssh_url], work)

        url = f"https://github.com/{full_name}" if org else ""
        if not url:
            view = _run(["gh", "repo", "view", "--json", "url", "-q", ".url"], work)
            url = view.stdout.strip() if view.returncode == 0 else f"github.com/{repo_name}"

        print(f"  [gh publish] pushed → {url}", flush=True)
        return url
