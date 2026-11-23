"""Grad-CAM (Selvaraju et al., 2017) through the full SafeDrive model.

The ResNet-18 embedding of the LAST frame of the observation window is recomputed with gradients,
standardised exactly as during training, inserted into the window, and the collision logit of each
ensemble member is back-propagated to the layer4 activations.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

from src.features.visual_features import GradCAMBackbone, preprocess


class GradCAM:
    def __init__(self, predictor, device: str = "cpu"):
        self.p = predictor
        self.device = device
        self.backbone = GradCAMBackbone().to(device).eval()
        self.mv = torch.from_numpy(predictor.scaler.mv).to(device)
        self.sv = torch.from_numpy(predictor.scaler.sv).to(device)

    def __call__(self, frame_bgr: np.ndarray, v_win: np.ndarray, m_win: np.ndarray) -> np.ndarray:
        """frame_bgr: last frame of the window; v_win/m_win: standardised window (T, D). Returns HxW cam in [0,1]."""
        x = preprocess([frame_bgr]).to(self.device)
        acts, pooled = self.backbone(x)
        acts.retain_grad()
        z = (pooled - self.mv) / self.sv
        xv = torch.from_numpy(v_win).to(self.device)[None].clone()
        xm = torch.from_numpy(m_win).to(self.device)[None]
        xv = torch.cat([xv[:, :-1], z[:, None]], 1)
        with torch.backends.cudnn.flags(enabled=False):
            logit = sum(m.eval()(xv, xm).sum() for m in self.p.models) / len(self.p.models)
            logit.backward()
        w = acts.grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * acts).sum(1))[0].detach().cpu().numpy()
        cam = cv2.resize(cam, (frame_bgr.shape[1], frame_bgr.shape[0]))
        return cam / cam.max() if cam.max() > 0 else cam


def overlay_cam(frame_bgr: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(cv2.addWeighted(frame_bgr, 1 - alpha, heat, alpha, 0), cv2.COLOR_BGR2RGB)
