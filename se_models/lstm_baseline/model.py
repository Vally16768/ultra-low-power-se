# se_models/lstm_baseline/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEnhancer(nn.Module):
    """
    Input:  x [B, 1, T]  (16 kHz)
    Output: y [B, 1, T]
    """

    def __init__(self, hidden=256, num_layers=2, bidirectional=True, frame_ms=20, lookahead_ms=0, sample_rate=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame = int(frame_ms * sample_rate // 1000)  # ex. 320
        self.hop = self.frame // 2  # ex. 160
        self.pad = self.frame // 2

        # Feature encoder (causal-friendly 1D conv)
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),  # down x2
            nn.PReLU(),
        )

        rnn_in = 64
        self.bi = bidirectional
        self.rnn = nn.LSTM(
            input_size=rnn_in,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        rnn_out = hidden * (2 if bidirectional else 1)

        # Decoder back to waveform domain
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(rnn_out, 32, kernel_size=4, stride=2, padding=1),  # up x2
            nn.PReLU(),
            nn.Conv1d(32, 1, kernel_size=9, stride=1, padding=4),
            nn.Tanh(),  # mask-ish output in [-1,1]
        )

    def _framing(self, x):
        # x: [B, 1, T] -> frames [B, C, T'] then permute to [B, T', C]
        # Folosim pur și simplu conv stridat pentru a obține un "T'" temporar.
        return self.enc(x)  # [B, 64, T']

    def forward(self, x):
        B, C, T = x.shape
        h = self._framing(x)  # [B, 64, T']
        h_t = h.transpose(1, 2)  # [B, T', 64] pentru LSTM
        h_t, _ = self.rnn(h_t)  # [B, T', H]
        h = h_t.transpose(1, 2)  # [B, H, T']
        y = self.dec(h)  # [B, 1, T] (după upsample)
        # aliniază exact la T (în caz de off-by-one datorat padding/stride)
        if y.size(-1) != T:
            if y.size(-1) > T:
                y = y[..., :T]
            else:
                y = F.pad(y, (0, T - y.size(-1)))
        return y


def build_model(sample_rate=16000, hidden=256, num_layers=2, bidirectional=True, frame_ms=20, lookahead_ms=0, **kwargs):
    return LSTMEnhancer(
        hidden=hidden, num_layers=num_layers, bidirectional=bidirectional, frame_ms=frame_ms, lookahead_ms=lookahead_ms, sample_rate=sample_rate
    )
