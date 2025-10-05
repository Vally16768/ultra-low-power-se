from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- utils ----------
def _is_mapping(x) -> bool:
    return isinstance(x, dict | SimpleNamespace)


def _get(obj, key: str, default=None):
    parts = key.split(".")
    cur = obj
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, SimpleNamespace) and hasattr(cur, p):
            cur = getattr(cur, p)
        else:
            return default
    return cur


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    out.update({k: v for k, v in b.items() if v is not None})
    return out


# ---------- model ----------
class LSTMEnhancer(nn.Module):
    """
    Ultra-light waveform denoiser:
      [B,1,T] -> Conv1d stack -> (Bi)LSTM pe [B,T,C] -> 1x1 -> Tanh -> (rezidual)
      I/O: [B,1,T] -> [B,1,T]
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        hidden: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        frame_ms: int = 20,
        lookahead_ms: int = 0,
        channels: int = 64,
        residual: bool = True,
    ):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.lookahead_ms = int(lookahead_ms)
        self.residual = bool(residual)

        c = int(channels)
        self.enc = nn.Sequential(
            nn.Conv1d(1, c // 2, kernel_size=7, stride=1, padding=3),
            nn.PReLU(),
            nn.Conv1d(c // 2, c, kernel_size=7, stride=1, padding=3),
            nn.PReLU(),
            nn.Conv1d(c, c, kernel_size=7, stride=1, padding=3),
            nn.PReLU(),
        )

        self.bi = bool(bidirectional)
        self.rnn = nn.LSTM(
            input_size=c,
            hidden_size=int(hidden),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=self.bi,
        )
        rnn_out = int(hidden) * (2 if self.bi else 1)
        self.proj = nn.Conv1d(rnn_out, 1, kernel_size=1)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        h = self.enc(x)  # [B, Cenc, T]
        h_t, _ = self.rnn(h.transpose(1, 2))  # [B, T, Hrnn]
        y_hat = self.act(self.proj(h_t.transpose(1, 2)))  # [B,1,T]
        out = x + y_hat if self.residual else y_hat
        if out.size(-1) != T:  # safety align
            out = out[..., :T] if out.size(-1) > T else F.pad(out, (0, T - out.size(-1)))
        return out


# ---------- builder ----------
def build_model(cfg: Any = None, **overrides) -> nn.Module:
    """
    Acceptă:
      * build_model(cfg) -> citește cfg.model.args + data.sample_rate
      * build_model(**kwargs)
      * build_model(cfg, hidden=..., ...) -> cfg + override
    """
    defaults = dict(
        sample_rate=16000,
        hidden=256,
        num_layers=2,
        bidirectional=True,
        frame_ms=20,
        lookahead_ms=0,
        channels=64,
        residual=True,
    )
    if _is_mapping(cfg):
        args_cfg = _get(cfg, "model.args", {}) or {}
        sr = _get(cfg, "data.sample_rate", None)
        if sr is not None:
            args_cfg = dict(args_cfg)
            args_cfg["sample_rate"] = sr
        args = _merge(defaults, args_cfg)
    else:
        args = dict(defaults)
    if overrides:
        args = _merge(args, overrides)
    return LSTMEnhancer(**args)
