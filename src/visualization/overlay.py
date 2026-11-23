"""Draw tracks, IDs and the risk HUD onto frames; write browser-playable H.264 video."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.detection.detector import COCO_ROAD
from src.risk.risk_engine import LEVEL_COLORS

CLASS_COLORS = {0: (255, 180, 60), 1: (60, 220, 255), 2: (90, 230, 90), 3: (255, 90, 200), 5: (60, 160, 255),
                7: (200, 200, 60)}


def _hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return b, g, r


def draw_frame(frame: np.ndarray, rows: np.ndarray, level: str | None = None, score: float | None = None,
               prob: float | None = None, t: float | None = None, highlight_ids: set[int] | None = None) -> np.ndarray:
    """rows: (n, 8) [step, id, cls, conf, x1, y1, x2, y2] normalised."""
    out = frame.copy()
    h, w = out.shape[:2]
    for r in rows:
        x1, y1, x2, y2 = (int(r[4] * w), int(r[5] * h), int(r[6] * w), int(r[7] * h))
        cls, tid = int(r[2]), int(r[1])
        col = CLASS_COLORS.get(cls, (220, 220, 220))
        thick = 3 if highlight_ids and tid in highlight_ids else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        label = f"#{tid} {COCO_ROAD.get(cls, cls)} {r[3]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(out, label, (x1 + 2, max(y1 - 4, th)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (15, 15, 15), 1, cv2.LINE_AA)
    if level is not None:
        col = _hex_bgr(LEVEL_COLORS.get(level, "#888888"))
        cv2.rectangle(out, (0, 0), (w, 34), (20, 20, 28), -1)
        cv2.rectangle(out, (0, 0), (8, 34), col, -1)
        txt = f"SafeDrive AI | {level}"
        if score is not None:
            txt += f" | risk {score:.2f}"
        if prob is not None:
            txt += f" | P(collision) {prob:.2f}"
        if t is not None:
            txt += f" | t={t:.1f}s"
        cv2.putText(out, txt, (16, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        if level in ("WARNING", "HIGH RISK"):
            cv2.rectangle(out, (0, 0), (w - 1, h - 1), col, 6)
    return out


def write_video(frames: list[np.ndarray], path: str | Path, fps: float) -> Path:
    """Write BGR frames as H.264 MP4 (imageio-ffmpeg), falling back to OpenCV mp4v."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=16,
                                ffmpeg_log_level="error") as wr:
            for f in frames:
                wr.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    except Exception:  # noqa: BLE001
        h, w = frames[0].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            vw.write(f)
        vw.release()
    return path
