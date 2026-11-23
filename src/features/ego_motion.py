"""Camera ego-motion estimation between consecutive 10 Hz grid frames (image-plane, no calibration).

Two estimators are computed every step so they can be compared:
  (a) 'homography': KLT corners on the background (detected road users and the bottom 12 % hood band masked out)
      -> pyramidal Lucas-Kanade -> RANSAC homography (8 dof) and RANSAC partial affine (4 dof, for interpretable stats)
  (b) 'flow': dense Farneback flow (160x90) on background pixels -> RANSAC partial affine.
Each transform maps NORMALISED coordinates (x/w, y/h) of frame t-1 to frame t, i.e. where a static background point moves.
Stats are rates per second: dx, dy (frame units/s), zoom ((scale-1)/s, forward motion), rot (rad/s), plus quality.

These are apparent camera motions on the image plane. With a monocular, uncalibrated camera they are not ego speed or yaw rate.
"""
from __future__ import annotations

import cv2
import numpy as np

EGO_H_STATS = ["egoh_dx", "egoh_dy", "egoh_zoom", "egoh_rot", "egoh_inliers", "egoh_reproj"]
EGO_F_STATS = ["egof_dx", "egof_dy", "egof_zoom", "egof_rot"]
# cache layout of the per-step `ego` array (T, 30)
EGO_LAYOUT = {"h_stats": slice(0, 6), "H": slice(6, 15), "f_stats": slice(15, 19), "F": slice(19, 28), "h_valid": 28, "f_valid": 29}
EGO_DIM = 30
IDENTITY = np.eye(3, dtype=np.float64).ravel()


def _norm_transform(M: np.ndarray, w: float, h: float) -> np.ndarray:
    """Pixel-space 3x3 transform -> normalised-coordinate transform."""
    S = np.diag([1.0 / w, 1.0 / h, 1.0])
    return S @ M @ np.linalg.inv(S)


def _affine_stats(A: np.ndarray, w: float, h: float, dt: float) -> np.ndarray:
    """Partial affine (2x3, pixel space) -> dx, dy (normalised/s), zoom (1/s), rot (rad/s)."""
    a, b = A[0, 0], A[1, 0]
    scale = float(np.hypot(a, b))
    rot = float(np.arctan2(b, a))
    # displacement of the image centre
    cx, cy = w / 2, h / 2
    nx, ny = A[0, 0] * cx + A[0, 1] * cy + A[0, 2], A[1, 0] * cx + A[1, 1] * cy + A[1, 2]
    return np.array([(nx - cx) / w / dt, (ny - cy) / h / dt, (scale - 1) / dt, rot / dt], np.float32)


def background_mask(shape: tuple[int, int], boxes: np.ndarray, pad: float = 0.1, hood: float = 0.12) -> np.ndarray:
    h, w = shape
    m = np.full((h, w), 255, np.uint8)
    m[int(h * (1 - hood)):, :] = 0
    for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
        bw, bh = (x2 - x1) * pad, (y2 - y1) * pad
        m[max(int((y1 - bh) * h), 0): int(min((y2 + bh), 1) * h) + 1, max(int((x1 - bw) * w), 0): int(min((x2 + bw), 1) * w) + 1] = 0
    return m


class EgoMotionEstimator:
    def __init__(self, size: tuple[int, int] = (320, 180), max_corners: int = 300):
        self.size = tuple(size)
        self.max_corners = max_corners
        self.prev: np.ndarray | None = None
        self.prev_mask: np.ndarray | None = None

    def reset(self) -> None:
        self.prev = self.prev_mask = None

    def __call__(self, frame_bgr: np.ndarray, boxes_norm: np.ndarray, dt: float, flow_px: np.ndarray | None = None,
                 flow_size: tuple[int, int] | None = None) -> np.ndarray:
        """boxes_norm: (n,4) normalised boxes in the CURRENT frame. flow_px: dense Farneback field (prev->cur) in pixels
        of `flow_size` (optional, from FlowComputer). Returns the (30,) ego vector (identity / zeros when invalid)."""
        w, h = self.size
        g = cv2.cvtColor(cv2.resize(frame_bgr, self.size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        mask = background_mask((h, w), boxes_norm)
        out = np.zeros(EGO_DIM, np.float32)
        out[EGO_LAYOUT["H"]] = IDENTITY
        out[EGO_LAYOUT["F"]] = IDENTITY
        if self.prev is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev, self.max_corners, 0.01, 7, mask=self.prev_mask)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                ok = st.ravel() == 1
                a, b = p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2)
                if len(a) >= 12:
                    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 2.0)
                    A, _ = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                    if H is not None and A is not None and inl is not None and abs(np.linalg.det(H[:2, :2]) - 1) < 0.5:
                        proj = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
                        resid = np.linalg.norm(proj - b, axis=1)[inl.ravel() == 1]
                        out[EGO_LAYOUT["h_stats"]] = np.concatenate([_affine_stats(A, w, h, dt),
                                                                     [inl.mean(), float(np.median(resid)) / w if len(resid) else 0.0]])
                        out[EGO_LAYOUT["H"]] = _norm_transform(H, w, h).ravel()
                        out[EGO_LAYOUT["h_valid"]] = 1.0
            if flow_px is not None and flow_size is not None:
                fw, fh = flow_size
                fm = background_mask((fh, fw), boxes_norm)
                ys, xs = np.mgrid[2:fh:4, 2:fw:4]
                sel = fm[ys, xs] > 0
                if sel.sum() >= 20:
                    src = np.stack([xs[sel], ys[sel]], 1).astype(np.float32)
                    dst = src + flow_px[ys[sel], xs[sel]]
                    A2, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=1.0)
                    if A2 is not None:
                        out[EGO_LAYOUT["f_stats"]] = _affine_stats(A2, fw, fh, dt)
                        M = np.vstack([A2, [0, 0, 1]])
                        out[EGO_LAYOUT["F"]] = _norm_transform(M, fw, fh).ravel()
                        out[EGO_LAYOUT["f_valid"]] = 1.0
        self.prev, self.prev_mask = g, mask
        return out
