"""Temporal collision-anticipation models. All contain dropout so MC-Dropout can be applied."""
from __future__ import annotations

import torch
import torch.nn as nn


class _Proj(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(), nn.Dropout(p))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, p: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T), causal
        y = self.conv(nn.functional.pad(x, (self.pad, 0)))
        return x + self.drop(torch.relu(y))


class CollisionModel(nn.Module):
    """kind: 'frame' (last step only, no temporal context), 'pool' (mean+max temporal pooling),
    'gru', 'lstm', 'tcn'. Inputs can be visual-only, motion-only or both."""

    def __init__(self, kind: str = "gru", use_visual: bool = True, use_motion: bool = True, d_visual: int = 512,
                 d_motion: int = 28, visual_proj: int = 64, motion_proj: int = 32, hidden: int = 64,
                 dropout: float = 0.3, in_dropout: float = 0.0, visual_mode: str = "abs"):
        super().__init__()
        assert use_visual or use_motion
        assert visual_mode in ("abs", "delta")
        self.kind, self.use_visual, self.use_motion, self.visual_mode = kind, use_visual, use_motion, visual_mode
        self.hparams = dict(kind=kind, use_visual=use_visual, use_motion=use_motion, d_visual=d_visual,
                            d_motion=d_motion, visual_proj=visual_proj, motion_proj=motion_proj, hidden=hidden,
                            dropout=dropout, in_dropout=in_dropout, visual_mode=visual_mode)
        self.in_drop = nn.Dropout(in_dropout)  # feature-level input dropout (regularises the 512-d visual input)
        d = 0
        if use_visual:
            self.pv = _Proj(d_visual, visual_proj, dropout)
            d += visual_proj
        if use_motion:
            self.pm = _Proj(d_motion, motion_proj, dropout)
            d += motion_proj
        if kind in ("gru", "lstm"):
            self.rnn = (nn.GRU if kind == "gru" else nn.LSTM)(d, hidden, batch_first=True)
            d_out = hidden
        elif kind == "tcn":
            self.inp = nn.Conv1d(d, hidden, 1)
            self.tcn = nn.Sequential(*[TCNBlock(hidden, 2 ** i, dropout) for i in range(4)])
            d_out = hidden
        elif kind == "pool":
            self.mlp = nn.Sequential(nn.Linear(2 * d, hidden), nn.ReLU())
            d_out = hidden
        elif kind == "frame":
            self.mlp = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
            d_out = hidden
        else:
            raise ValueError(kind)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_out, 1))

    def embed(self, xv: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        parts = []
        if self.use_visual:
            if self.visual_mode == "delta":  # appearance change v_t - v_{t-1}: removes static scene identity
                xv = torch.cat([torch.zeros_like(xv[:, :1]), xv[:, 1:] - xv[:, :-1]], 1)
            parts.append(self.pv(self.in_drop(xv)))
        if self.use_motion:
            parts.append(self.pm(self.in_drop(xm)))
        return torch.cat(parts, -1)

    def forward(self, xv: torch.Tensor, xm: torch.Tensor) -> torch.Tensor:
        z = self.embed(xv, xm)  # (B, T, d)
        if self.kind in ("gru", "lstm"):
            out, _ = self.rnn(z)
            h = out[:, -1]
        elif self.kind == "tcn":
            h = self.tcn(self.inp(z.transpose(1, 2)))[:, :, -1]
        elif self.kind == "pool":
            h = self.mlp(torch.cat([z.mean(1), z.max(1).values], -1))
        else:
            h = self.mlp(z[:, -1])
        return self.head(h).squeeze(-1)


def build_model(hp: dict) -> CollisionModel:
    return CollisionModel(**hp)
