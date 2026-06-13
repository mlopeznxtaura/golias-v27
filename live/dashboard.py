"""Golias v27 live dashboard — τ state machine, JSONL corpus training, of₁/of₂ outputs."""
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

from auth import HEADER, auth_ok  # noqa: E402
from goliasv7_torch import GoliasV7Torch  # noqa: E402
from jsonl_corpus_stream import encode_jsonl_record  # noqa: E402
from ui_page import PAGE  # noqa: E402
from v27 import compute_tau, project_outputs  # noqa: E402

PORT = int(os.environ.get("PORT", "8080"))
BIND = os.environ.get("GOLIAS_BIND", "0.0.0.0")
CKPT = Path(os.environ.get("GOLIAS_CKPT", ROOT / "goliasv11.pt"))
LOG = Path(os.environ.get("GOLIAS_LOG", ROOT / "train.jsonl.log"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_ckpt_mtime = 0.0
_train_proc = None
_lock = threading.Lock()


def load_model():
    global _model, _ckpt_mtime
    with _lock:
        if not CKPT.exists():
            return None
        mtime = CKPT.stat().st_mtime
        if _model is not None and mtime == _ckpt_mtime:
            return _model
        m = GoliasV7Torch().to(DEVICE)
        m.load_checkpoint(str(CKPT))
        m.eval()
        _model, _ckpt_mtime = m, mtime
        return _model


def _encode_tensors(geometry: float, binary: float, language: str, triangulation: Optional[float]):
    rec = {
        "geometry": geometry,
        "binary": binary,
        "language": language,
        "triangulation": triangulation if triangulation is not None else (geometry + binary) / 2,
        "next_frame": geometry,
        "next_token": "",
    }
    m1, m2, m3, _, _, _ = encode_jsonl_record(rec)
    return (
        torch.from_numpy(m1).unsqueeze(0).to(DEVICE),
        torch.from_numpy(m2).unsqueeze(0).to(DEVICE),
        torch.from_numpy(m3).unsqueeze(0).to(DEVICE),
    )


def run_v27_forward(p: dict) -> dict:
    model = load_model()
    if model is None:
        return {"error": "checkpoint not found", "ckpt": str(CKPT)}

    g = float(p.get("geometry", 0.47))
    b = float(p.get("binary", 0.73))
    lang = str(p.get("language", p.get("question", "")))
    tri = p.get("triangulation")
    tri_f = float(tri) if tri is not None else None
    tau = compute_tau(g, b, float(p.get("language_scalar", p.get("l", 0.4))))

    model.TAU.data.fill_(tau)
    m1, m2, m3 = _encode_tensors(g, b, lang, tri_f)

    with torch.no_grad():
        out = model(m1, m2, m3)

    v27 = project_outputs(out["pred"], out["decode"], out["comp_score"], tau, g, b, lang)
    v27["V"] = float(p.get("V", p.get("v", 0.58)))
    v27["ckpt"] = CKPT.name
    v27["device"] = DEVICE
    return v27


def start_jsonl_train():
    global _train_proc
    with _lock:
        if _train_proc and _train_proc.poll() is None:
            return False, "training already running"
        env = os.environ.copy()
        env["GOLIAS_RESUME"] = str(CKPT)
        env["GOLIAS_LOG"] = str(LOG)
        env["GOLIAS_OUTPUT"] = str(ROOT / "goliasv27.pt")
        script = ROOT / "training" / "train_from_jsonl.py"
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
            self._send(200, "application/json", json.dumps({"status": "ok", "arch": "v27"}))
        elif path == "/info":
            self._send(200, "application/json", json.dumps({
                "arch": "Golias-NextAura-v27",
                "device": DEVICE,
                "ckpt": CKPT.name,
                "ckpt_exists": CKPT.exists(),
                "corpus": str(ROOT / "data" / "goliasv11_corpus.jsonl"),
                "log": str(LOG),
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

        if path == "/forward":
            self._send(200, "application/json", json.dumps(run_v27_forward(p)))
        elif path == "/infer":
            self._send(200, "application/json", json.dumps(run_v27_forward(p)))
        elif path == "/ask":
            r = run_v27_forward(p)
            if "error" in r:
                self._send(503, "application/json", json.dumps(r))
                return
            answer = (
                f"τ={r['tau']} | of₁ next_frame={r['of1_scalar']}\n"
                f"of₂: {r['of2_explanation']}\n"
            )
            if r.get("rl_language_context"):
                answer += f"RL: {r['rl_language_context']}\n"
            if r.get("halt"):
                answer += "HALT — C_comp > τ\n"
            self._send(200, "application/json", json.dumps({"answer": answer, "v27": r}))
        elif path == "/train/jsonl":
            ok, msg = start_jsonl_train()
            code = 200 if ok else 409
            self._send(code, "application/json", json.dumps({"ok": ok, "message": msg}))
        elif path == "/judge":
            r = run_v27_forward(p)
            v = float(p.get("V", 0.5))
            r["judge_V"] = v
            r["adapt_signal"] = "retune" if v < 0.65 else "hold"
            self._send(200, "application/json", json.dumps(r))
        else:
            self._send(404, "text/plain", "not found")


def main():
    print(f"v27 dashboard http://{BIND}:{PORT} device={DEVICE} ckpt={CKPT}")
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
