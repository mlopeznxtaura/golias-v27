"""Scalar-pack → frame images + short preview clip for live GUI."""
from __future__ import annotations

import base64
import io
import math
from typing import Any, Sequence

import numpy as np

from dimensions import OF1_VIS as OF1_DIM, OF1_VIS_GRID

W, H = OF1_VIS_GRID[1] * 16, OF1_VIS_GRID[0] * 16


def _norm_vec(vec: Sequence[float], dim: int = OF1_DIM) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    arr = arr[:dim]
    mx = float(np.abs(arr).max()) or 1.0
    return arr / mx


def _block_xy(geometry: float, pred_scalar: float, frame_idx: int, total: int) -> tuple[float, float]:
    t = frame_idx / max(total - 1, 1)
    g0 = float(geometry)
    g1 = float(pred_scalar)
    gx = g0 + (g1 - g0) * t
    x = 24 + gx * (W - 48)
    y = H * 0.55 + math.sin(gx * math.pi * 2) * 12
    return x, y


def _scene_svg(
    geometry: float,
    binary: float,
    language: str,
    scalars: Sequence[float],
    *,
    label: str,
    block_xy: tuple[float, float] | None = None,
) -> str:
    vec = _norm_vec(scalars)
    grid = vec.reshape(OF1_VIS_GRID)
    cells = []
    cw, ch = W / OF1_VIS_GRID[1], H / OF1_VIS_GRID[0]
    for r in range(OF1_VIS_GRID[0]):
        for c in range(OF1_VIS_GRID[1]):
            v = float(grid[r, c])
            gray = int((v * 0.5 + 0.5) * 180)
            cells.append(
                f'<rect x="{c*cw:.1f}" y="{r*ch:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                f'fill="rgb({gray},{gray},{max(gray-20,0)})" opacity="0.55"/>'
            )

    bx, by = block_xy if block_xy else _block_xy(geometry, float(np.mean(vec[:8])), 0, 1)
    size = 14 + int(binary * 22)
    lang = (language or "").lower()
    color = "#ef4444"
    if "blue" in lang:
        color = "#3b82f6"
    elif "green" in lang:
        color = "#22c55e"

    arrow = ""
    if "left" in lang:
        arrow = f'<polygon points="{bx-28},{by} {bx-12},{by-10} {bx-12},{by+10}" fill="#fbbf24"/>'
    elif "right" in lang:
        arrow = f'<polygon points="{bx+28},{by} {bx+12},{by-10} {bx+12},{by+10}" fill="#fbbf24"/>'

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
        f'<rect width="{W}" height="{H}" fill="#0b1220"/>'
        + "".join(cells)
        + f'<rect x="{bx-size/2:.1f}" y="{by-size/2:.1f}" width="{size}" height="{size}" '
        f'rx="3" fill="{color}" stroke="#fecaca" stroke-width="1.5"/>'
        + arrow
        + f'<text x="8" y="14" fill="#94a3b8" font-size="10" font-family="monospace">{label}</text>'
        + f'<text x="8" y="{H-6}" fill="#64748b" font-size="9" font-family="monospace">'
        f'G={geometry:.3f} B={binary:.3f}</text></svg>'
    )


def _svg_data_url(svg: str) -> str:
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def _raster_frames_png(
    geometry: float,
    binary: float,
    language: str,
    current_scalars: Sequence[float],
    next_scalars: Sequence[float],
    n_frames: int = 8,
) -> tuple[str, str, str | None, list[str]]:
    pred_scalar = float(np.mean(_norm_vec(next_scalars)[:8]))
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        cur = _svg_data_url(_scene_svg(geometry, binary, language, current_scalars, label="current"))
        nxt = _svg_data_url(_scene_svg(
            geometry, binary, language, next_scalars, label="next frame",
            block_xy=_block_xy(geometry, pred_scalar, 1, 1),
        ))
        seq = [
            _svg_data_url(_scene_svg(
                geometry, binary, language, next_scalars, label=f"t={i}",
                block_xy=_block_xy(geometry, pred_scalar, i, n_frames),
            ))
            for i in range(n_frames)
        ]
        return cur, nxt, None, seq

    def _draw(scalars: Sequence[float], label: str, block_xy: tuple[float, float]) -> Image.Image:
        vec = _norm_vec(scalars)
        grid = vec.reshape(OF1_VIS_GRID)
        img = Image.new("RGB", (W, H), (11, 18, 32))
        px = img.load()
        cw, ch = W / OF1_VIS_GRID[1], H / OF1_VIS_GRID[0]
        for r in range(14):
            for c in range(16):
                v = float(grid[r, c])
                gray = int((v * 0.5 + 0.5) * 180)
                x0, y0 = int(c * cw), int(r * ch)
                x1, y1 = int((c + 1) * cw), int((r + 1) * ch)
                for y in range(y0, min(y1, H)):
                    for x in range(x0, min(x1, W)):
                        px[x, y] = (gray, gray, max(gray - 20, 0))
        draw = ImageDraw.Draw(img)
        bx, by = block_xy
        size = 14 + int(binary * 22)
        lang = (language or "").lower()
        color = (239, 68, 68)
        if "blue" in lang:
            color = (59, 130, 246)
        elif "green" in lang:
            color = (34, 197, 94)
        draw.rounded_rectangle(
            [bx - size / 2, by - size / 2, bx + size / 2, by + size / 2],
            radius=3, fill=color, outline=(254, 202, 202),
        )
        draw.text((8, 6), label, fill=(148, 163, 184))
        draw.text((8, H - 14), f"G={geometry:.3f} B={binary:.3f}", fill=(100, 116, 139))
        return img

    cur_img = _draw(current_scalars, "current", _block_xy(geometry, geometry, 0, 1))
    nxt_img = _draw(next_scalars, "next frame", _block_xy(geometry, pred_scalar, 1, 1))

    def _png_b64(im: Image.Image) -> str:
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    seq_imgs = [
        _draw(next_scalars, f"t={i}", _block_xy(geometry, pred_scalar, i, n_frames))
        for i in range(n_frames)
    ]

    gif_buf = io.BytesIO()
    seq_imgs[0].save(
        gif_buf, format="GIF", save_all=True, append_images=seq_imgs[1:],
        duration=120, loop=0, optimize=False,
    )
    gif_b64 = "data:image/gif;base64," + base64.b64encode(gif_buf.getvalue()).decode("ascii")

    return _png_b64(cur_img), _png_b64(nxt_img), gif_b64, [_png_b64(im) for im in seq_imgs]


def current_state_scalars(geometry: float, binary: float, language: str) -> list[float]:
    lang = (language or "")[:512]
    g, b = float(geometry), float(binary)
    tri = (g + b) / 2
    base = np.array([g, tri, b, g * b, tri, len(lang) / 512.0], dtype=np.float32)
    out = np.zeros(OF1_DIM, dtype=np.float32)
    for j, val in enumerate(base):
        freq = (j + 1) * math.pi / len(base)
        out += val * np.sin(np.arange(OF1_DIM, dtype=np.float32) * freq / OF1_DIM * 2 * math.pi)
    mx = float(np.abs(out).max()) or 1.0
    return (out / mx).tolist()


def render_frame_outputs(
    geometry: float,
    binary: float,
    language: str,
    of1_next_frame: Sequence[float],
) -> dict[str, Any]:
    current = current_state_scalars(geometry, binary, language)
    cur_url, next_url, gif_url, seq = _raster_frames_png(
        geometry, binary, language, current, of1_next_frame,
    )
    return {
        "current_frame_image": cur_url,
        "next_frame_image": next_url,
        "next_frame_video": gif_url,
        "frame_sequence": seq,
        "frame_width": W,
        "frame_height": H,
    }
