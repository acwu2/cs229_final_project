"""
Frame-level onset classifier (Approach A baseline).

Architecture
------------
   raw audio  ──►  [frozen wav2vec2]  ──►  (B, T, 768)
                                              │
                        interpolate to grid   │
                                              ▼
                                        (B, S, 768)     S = num grid slots
                                              │
                               ┌──────────────┘
                               ▼
                          MLP / BiLSTM head
                               │
                               ▼
                         (B, S, 1)   logits  →  BCE loss
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.feature_extractor import AudioFeatureExtractor


# ---------------------------------------------------------------------------
# MLP classification head
# ---------------------------------------------------------------------------

class MLPHead(nn.Module):
    """Per-frame MLP: (B, T, D) → (B, T, 1)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_d = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        layers.append(nn.Linear(in_d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)          # (B, T)


# ---------------------------------------------------------------------------
# BiLSTM classification head (captures local temporal context)
# ---------------------------------------------------------------------------

class BiLSTMHead(nn.Module):
    """BiLSTM + linear: (B, T, D) → (B, T, 1)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.lstm(x)                     # (B, T, 2*H)
        h = self.dropout(h)
        return self.fc(h).squeeze(-1)           # (B, T)


# ---------------------------------------------------------------------------
# Full model: encoder + head
# ---------------------------------------------------------------------------

class RhythmQuantizer(nn.Module):
    """End-to-end: waveform → onset logits per grid slot."""

    def __init__(
        self,
        encoder_name: str = "facebook/wav2vec2-base",
        freeze_encoder: bool = True,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        head_type: str = "mlp",           # "mlp" or "bilstm"
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.encoder = AudioFeatureExtractor(
            model_name=encoder_name,
            freeze=freeze_encoder,
            sample_rate=sample_rate,
        )

        Head = MLPHead if head_type == "mlp" else BiLSTMHead
        self.head = Head(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        waveforms: torch.Tensor,       # (B, num_audio_samples)
        num_grid_slots: int = 32,
    ) -> torch.Tensor:
        """Return onset logits of shape (B, num_grid_slots)."""
        features = self.encoder(waveforms, num_grid_slots=num_grid_slots)
        logits = self.head(features)    # (B, S)
        return logits
