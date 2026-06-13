"""Shared internal auth between CE proxy and GPU backend."""
import os

INTERNAL_KEY = os.environ.get("GOLIAS_INTERNAL_KEY", "")
HEADER = "X-Golias-Key"


def _header(headers, name: str) -> str:
    low = name.lower()
    for k, v in headers.items():
        if k.lower() == low:
            return v
    return ""


def auth_ok(headers) -> bool:
    if not INTERNAL_KEY:
        return True
    return _header(headers, HEADER) == INTERNAL_KEY


def proxy_headers() -> dict:
    h = {"User-Agent": "Golias-CE-Proxy/1.0"}
    if INTERNAL_KEY:
        h[HEADER] = INTERNAL_KEY
    return h
