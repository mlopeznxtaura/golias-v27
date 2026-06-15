"""Golias v27 live dashboard — τ state machine, M1/M2/M3 sidecars, of₁/of₂ outputs."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "sidecars"))

try:
    from secrets_loader import ensure_secrets
    ensure_secrets()
except ImportError:
    pass

from auth import HEADER, auth_ok  # noqa: E402
from checkpoint_registry import list_checkpoints, read_hf_offset, resolve_active_ckpt  # noqa: E402
from goliasv7_torch import GoliasV7Torch  # noqa: E402
from if_sidecars import build_if_payload, encode_sidecar_vectors, run_sidecar_pipeline  # noqa: E402
from ui_page import PAGE  # noqa: E402
from frame_renderer import render_frame_outputs  # noqa: E402
from v27 import compute_tau, language_scalar_from_text, project_outputs  # noqa: E402
from language_record import resolve_language_input  # noqa: E402

try:
    from intake_upload import ingest_bytes
except ImportError:
    ingest_bytes = None  # type: ignore

PORT = int(os.environ.get("PORT", "8080"))
BIND = os.environ.get("GOLIAS_BIND", "0.0.0.0")


def active_ckpt() -> Path:
    """Always resolve freshest checkpoint (pointer → newest file → env)."""
    return resolve_active_ckpt(ROOT)
LOG = Path(os.environ.get("GOLIAS_LOG", ROOT / "train.jsonl.log"))
IF_BACKEND = os.environ.get("IF_BACKEND", "watsonx")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_ckpt_mtime = 0.0
_ckpt_path: Optional[Path] = None
_train_proc = None
_lock = threading.Lock()


def load_model():
    global _model, _ckpt_mtime, _ckpt_path
    ckpt = active_ckpt()
    with _lock:
        if not ckpt.exists():
            return None
        mtime = ckpt.stat().st_mtime
        if _model is not None and mtime == _ckpt_mtime and _ckpt_path == ckpt.resolve():
            return _model
        m = GoliasV7Torch().to(DEVICE)
        m.load_checkpoint(str(ckpt))
        m.eval()
        _model, _ckpt_mtime, _ckpt_path = m, mtime, ckpt.resolve()
        return _model


def _sliders(p: dict) -> dict:
    return {
        "m1": float(p.get("m1", 4.2)),
        "m2": float(p.get("m2", 0.55)),
        "m3": float(p.get("m3", 0.99)),
        "v": float(p.get("V", p.get("v", 0.58))),
        "if7": float(p.get("if7", 0.5)),
    }


def _halt_explanation(sidecars: dict, tau: float) -> str:
    c_comp = float(sidecars.get("c_comp_proxy", 0))
    reason = sidecars.get("halt_reason") or (sidecars.get("m2") or {}).get("halt_reason")
    if reason == "c_comp_gt_tau" or c_comp > tau:
        return f"HALT — C_comp={c_comp} > τ={tau:.4f}"
    return "HALT — M2 efficiency zealot"


def run_v27_forward(p: dict) -> dict:
    ckpt = active_ckpt()
    model = load_model()
    if model is None:
        return {"error": "checkpoint not found", "ckpt": str(ckpt)}

    resolved = resolve_language_input(p)
    g = float(resolved["geometry"])
    b = float(resolved["binary"])
    lang = str(resolved["language"])
    sliders = {
        "m1": float(resolved["m1"]),
        "m2": float(resolved["m2"]),
        "m3": float(resolved["m3"]),
        "v": float(resolved["V"]),
        "if7": float(resolved["if7"]),
    }
    tau = compute_tau(g, b, language_scalar_from_text(lang))

    if_payload = build_if_payload(
        g, b, lang, tau,
        sliders["m1"], sliders["m2"], sliders["m3"],
        sliders["v"], sliders["if7"],
    )

    try:
        sidecars = run_sidecar_pipeline(if_payload)
    except Exception as ex:
        return {"error": f"sidecar pipeline failed: {ex}", "ckpt": ckpt.name}

    if sidecars.get("halt"):
        return {
            "halt": True,
            "halt_source": "m2_sidecar",
            "tau": round(tau, 6),
            "c_comp": sidecars.get("c_comp_proxy"),
            "sidecars": sidecars,
            "ckpt": ckpt.name,
            "device": DEVICE,
            "if_backend": IF_BACKEND,
            "of2_explanation": _halt_explanation(sidecars, tau),
            "of1_scalar": g,
        }

    m1t, m2t, m3t = encode_sidecar_vectors(
        sliders["m1"], sliders["m2"], sliders["m3"],
        g, b, lang, sliders["v"], sliders["if7"],
        sidecars["m1"], sidecars["m2"], sidecars["m3"],
        device=DEVICE,
    )

    model.TAU.data.fill_(tau)
    with torch.no_grad():
        out = model(m1t, m2t, m3t)

    v27 = project_outputs(
        out["pred"], out["decode"], out["comp_score"], tau, g, b, lang,
        m1_exploration=str(sidecars.get("m1", {}).get("exploration", "")),
    )
    frames = render_frame_outputs(g, b, lang, v27.get("of1_next_frame", []))
    v27.update(frames)
    v27["V"] = sliders["v"]
    v27["m1"] = sliders["m1"]
    v27["m2"] = sliders["m2"]
    v27["m3"] = sliders["m3"]
    v27["if7"] = sliders["if7"]
    v27["sidecars"] = sidecars
    v27["if_backend"] = IF_BACKEND
    v27["ckpt"] = ckpt.name
    v27["device"] = DEVICE
    v27["input_record"] = resolved
    v27["geometry"] = g
    v27["binary"] = b
    return v27


def start_jsonl_train():
    global _train_proc
    with _lock:
        if _train_proc and _train_proc.poll() is None:
            return False, "training already running"
        resume = active_ckpt()
        env = os.environ.copy()
        env["GOLIAS_RESUME"] = str(resume)
        env.pop("GOLIAS_OUTPUT", None)
        env.pop("GOLIAS_CKPT", None)  # train uses latest pointer, not stale env
        env["GOLIAS_LOG"] = str(LOG)
        env["GOLIAS_JSONL"] = os.environ.get(
            "GOLIAS_JSONL", str(ROOT / "data" / "goliasv27_corpus.jsonl")
        )
        env["GOLIAS_TRAIN_MODE"] = os.environ.get("GOLIAS_TRAIN_MODE", "hybrid")
        env["GOLIAS_HF_SAMPLES"] = os.environ.get("GOLIAS_HF_SAMPLES", "50000")
        env["GOLIAS_HF_OFFSET"] = os.environ.get("GOLIAS_HF_OFFSET", "0")
        env["GOLIAS_JSONL_EPOCHS"] = os.environ.get("GOLIAS_JSONL_EPOCHS", "3")
        script = ROOT / "training" / "train_hybrid.py"
        _train_proc = subprocess.Popen(
            [sys.executable, str(script)],
            cwd=str(ROOT),
            env=env,
            stdout=open(LOG, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
    return True, f"started pid {_train_proc.pid}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("cache-control", "no-store")
        self.end_headers()
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.wfile.write(body)

    def _authorized(self) -> bool:
        if self.path.split("?", 1)[0] == "/health":
            return True
        return auth_ok({k: v for k, v in self.headers.items()})

    def do_GET(self):
        if not self._authorized():
            self._send(401, "application/json", json.dumps({"error": "unauthorized"}))
            return
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/health":
            self._send(200, "application/json", json.dumps({
                "status": "ok", "arch": "v27", "if_backend": IF_BACKEND,
            }))
        elif path == "/info":
            ckpt = active_ckpt()
            all_ckpts = list_checkpoints(ROOT)
            self._send(200, "application/json", json.dumps({
                "arch": "Golias-NextAura-v27",
                "device": DEVICE,
                "ckpt": ckpt.name,
                "ckpt_path": str(ckpt),
                "ckpt_exists": ckpt.exists(),
                "latest_pointer": str(ROOT / "checkpoints" / "latest.txt"),
                "checkpoints": [str(p) for p in all_ckpts[:8]],
                "hf_stream_offset": read_hf_offset(ROOT, 70000),
                "if_backend": IF_BACKEND,
                "corpus": str(ROOT / "data" / "goliasv27_corpus.jsonl"),
                "drop_corpus": str(ROOT / "data" / "drop_corpus.jsonl"),
                "doctrine": str(ROOT / "data" / "architecture_doctrine.jsonl"),
                "ledger": "public — runtime on IBM Cloud",
                "log": str(LOG),
                "gh_publish": os.environ.get("GOLIAS_GH_PUBLISH", "1"),
            }))
        elif path == "/log":
            from urllib.parse import parse_qs, urlparse
            qs = parse_qs(urlparse(self.path).query)
            try:
                pos = int(qs.get("pos", ["0"])[0])
            except ValueError:
                pos = 0
            lines, new_pos = [], pos
            if LOG.exists():
                with open(LOG, "r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    new_pos = f.tell()
                lines = chunk.splitlines()
            else:
                lines = ["[ready — POST /train/jsonl to start corpus replay]"]
            self._send(200, "application/json", json.dumps({"pos": new_pos, "lines": lines}))
        else:
            self._send(404, "text/plain", "not found")

    def do_POST(self):
        if not self._authorized():
            self._send(401, "application/json", json.dumps({"error": "unauthorized"}))
            return
        path = self.path.split("?", 1)[0]
        n = int(self.headers.get("content-length", 0))
        try:
            p = json.loads(self.rfile.read(n) or "{}")
        except Exception:
            p = {}

        if path in ("/forward", "/infer"):
            self._send(200, "application/json", json.dumps(run_v27_forward(p)))
        elif path == "/ask":
            r = run_v27_forward(p)
            if "error" in r:
                self._send(503, "application/json", json.dumps(r))
                return
            if r.get("halt_source") == "m2_sidecar":
                answer = f"{r.get('of2_explanation', 'HALT — M2 sidecar')}\n"
            else:
                nf = r.get("next_frame_scalar", r.get("of1_scalar"))
                lang = r.get("of2_language") or r.get("of2_explanation", "")
                aligned = r.get("outputs_aligned")
                answer = (
                    f"ALIGNMENT: {r.get('alignment_explanation', aligned)}\n\n"
                    f"NEXT FRAME (of₁) scalar={nf}\n"
                    f"  current_frame_image: {'yes' if r.get('current_frame_image') else 'no'}\n"
                    f"  next_frame_image: {'yes' if r.get('next_frame_image') else 'no'}\n"
                    f"  clip: {'yes' if r.get('next_frame_video') or r.get('frame_sequence') else 'no'}\n\n"
                    f"LANGUAGE (of₂)\n  {lang}\n"
                )
                answer += (
                    f"\nτ={r['tau']} | halt={r.get('halt')}\n"
                    f"of₂ full: {r.get('of2_explanation', '')}\n"
                )
            if r.get("rl_language_context"):
                answer += f"RL: {r['rl_language_context']}\n"
            if r.get("halt"):
                answer += "HALT — C_comp > τ\n"
            sc = r.get("sidecars", {})
            if sc:
                answer += f"Sidecars: {sc.get('backends', {})}\n"
                if sc.get("errors"):
                    answer += f"Sidecar errors: {sc.get('errors')}\n"
            self._send(200, "application/json", json.dumps({"answer": answer, "v27": r}))
        elif path == "/train/jsonl":
            ok, msg = start_jsonl_train()
            code = 200 if ok else 409
            self._send(code, "application/json", json.dumps({"ok": ok, "message": msg}))
        elif path == "/upload/dataset":
            if ingest_bytes is None:
                self._send(500, "application/json", json.dumps({"error": "intake_upload unavailable"}))
                return
            filename = self.headers.get("X-Filename", "upload.json")
            raw = self.rfile.read(n) if n else b""
            auto_train = self.headers.get("X-Auto-Train", "").lower() in ("1", "true", "yes")
            try:
                result = ingest_bytes(raw, filename, root=ROOT, merge_master=True)
            except Exception as ex:
                self._send(400, "application/json", json.dumps({"error": str(ex)}))
                return
            train_msg = ""
            if auto_train:
                ok, train_msg = start_jsonl_train()
                result["train_started"] = ok
                result["train_message"] = train_msg
            self._send(200, "application/json", json.dumps(result))
        elif path == "/judge":
            r = run_v27_forward(p)
            v = float(p.get("V", 0.5))
            r["judge_V"] = v
            r["adapt_signal"] = "retune" if v < 0.65 else "hold"
            self._send(200, "application/json", json.dumps(r))
        else:
            self._send(404, "text/plain", "not found")


def main():
    print(f"v27 dashboard http://{BIND}:{PORT} device={DEVICE} ckpt={active_ckpt()} if={IF_BACKEND}")
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
