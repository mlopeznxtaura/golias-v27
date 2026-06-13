"""Code Engine public HTTPS front — proxies to private GPU v27 backend."""
import json
import os
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from auth import proxy_headers
from ui_page import PAGE

PORT = int(os.environ.get("PORT", "8080"))
GPU_URL = os.environ.get("GOLIAS_GPU_URL", "http://127.0.0.1:8080").rstrip("/")
_p = urlparse(GPU_URL)
GPU_HOST = _p.hostname or "127.0.0.1"
GPU_PORT = _p.port or (443 if _p.scheme == "https" else 80)
GPU_SCHEME = _p.scheme or "http"

POST_PATHS = frozenset({"/infer", "/ask", "/forward", "/judge", "/train/jsonl"})


def _conn():
    if GPU_SCHEME == "https":
        return HTTPSConnection(GPU_HOST, GPU_PORT, timeout=120)
    return HTTPConnection(GPU_HOST, GPU_PORT, timeout=120)


def _gpu_request(method, path, body=None):
    headers = proxy_headers()
    headers["Accept"] = "*/*"
    if body is not None:
        headers["Content-Type"] = "application/json"
        headers["Content-Length"] = str(len(body))
    c = _conn()
    try:
        c.request(method, path, body=body, headers=headers)
        return c.getresponse()
    except Exception:
        c.close()
        raise


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

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", PAGE)
            return
        if path == "/health":
            self._send(200, "application/json", json.dumps({"status": "ok", "role": "ce-proxy-v27"}))
            return
        if path == "/info":
            try:
                resp = _gpu_request("GET", "/info")
                gpu = json.loads(resp.read().decode("utf-8", errors="replace"))
                resp.close()
            except Exception as ex:
                gpu = {"error": str(ex)}
            self._send(200, "application/json", json.dumps({
                "mode": "secure-hybrid-v27",
                "arch": gpu.get("arch", "Golias-NextAura-v27"),
                "gateway": "code-engine-https",
                "gpu_backend": GPU_URL,
                "device": gpu.get("device"),
                "ckpt": gpu.get("ckpt"),
                "ckpt_exists": gpu.get("ckpt_exists"),
                "corpus": gpu.get("corpus"),
                "doctrine": gpu.get("doctrine"),
                "if_backend": gpu.get("if_backend"),
                "log_exists": gpu.get("log_exists"),
            }))
            return
        if path == "/log":
            try:
                q = self.path.split("?", 1)
                gpu_path = "/log" + (f"?{q[1]}" if len(q) > 1 else "")
                resp = _gpu_request("GET", gpu_path)
                data = resp.read()
                self.send_response(resp.status)
                ct = resp.getheader("Content-Type", "application/json")
                self.send_header("content-type", ct)
                self.end_headers()
                self.wfile.write(data)
                resp.close()
            except Exception as ex:
                self._send(502, "application/json", json.dumps({"error": str(ex), "lines": []}))
            return
        self._send(404, "text/plain", "not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if path not in POST_PATHS:
            self._send(404, "text/plain", "not found")
            return
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n) if n else None
        try:
            resp = _gpu_request("POST", path, body=body)
            data = resp.read()
            self.send_response(resp.status)
            ct = resp.getheader("Content-Type", "application/json")
            self.send_header("content-type", ct)
            self.end_headers()
            self.wfile.write(data)
            resp.close()
        except Exception as ex:
            self._send(502, "application/json", json.dumps({"error": str(ex)}))


def main():
    print(f"CE proxy v27 :{PORT} -> {GPU_URL}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
