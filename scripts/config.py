from dataclasses import dataclass

@dataclass(frozen=True)
class Defaults:
    sr: int = 16000
    n_fft: int = 512
    hop: int = 128
    ctx: int = 2
    segment_seconds: float = 2.0
    batch_frames: int = 2048
    lr: float = 1e-3
    dropout: float = 0.2
    epochs: int = 10
    steps_per_epoch: int = 200
    val_steps: int = 20
