"""Frame-level visual embeddings from an ImageNet-pretrained CNN (torchvision ResNet-18).

The backbone is frozen. `GradCAMBackbone` exposes the last conv block for Grad-CAM.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_backbone() -> nn.Module:
    from torchvision.models import ResNet18_Weights, resnet18
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()
    return m.eval()


def preprocess(frames_bgr: list[np.ndarray], size: int = 224) -> torch.Tensor:
    arr = np.stack([cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
                    for f in frames_bgr]).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


class VisualEncoder:
    dim = 512

    def __init__(self, device: str = "cpu", size: int = 224, batch: int = 32):
        self.device, self.size, self.batch = device, size, batch
        self.model = build_backbone().to(device)
        self.half = device == "cuda"
        if self.half:
            self.model = self.model.half()

    @torch.no_grad()
    def encode(self, frames_bgr: list[np.ndarray]) -> np.ndarray:
        outs = []
        for i in range(0, len(frames_bgr), self.batch):
            x = preprocess(frames_bgr[i:i + self.batch], self.size).to(self.device)
            if self.half:
                x = x.half()
            outs.append(self.model(x).float().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros((0, self.dim), np.float32)


class GradCAMBackbone(nn.Module):
    """ResNet-18 split into conv trunk (-> layer4 activations) and pooling, for Grad-CAM."""

    def __init__(self):
        super().__init__()
        m = build_backbone()
        self.trunk = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2, m.layer3, m.layer4)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.trunk(x)
        return a, self.pool(a).flatten(1)
