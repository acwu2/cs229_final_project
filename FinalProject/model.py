"""
BiLSTM models for rhythm quantization.

Given per-note performance features (IOI, duration, velocity, pitch encoding),
predicts rhythm labels:
    • beat index in bar   (0-3)
    • subdivision index   (0-4)
    • duration class      (0-7, from DURATION_VOCAB)

Two model variants:
  1. RhythmQuantizer       — single composite head (160 classes)
  2. MultiHeadRhythmQuantizer — three separate heads (4 + 5 + 8 classes)
"""

import torch
import torch.nn as nn

# ----- label-space constants -----
NUM_BEATS = 4          # beat_index_in_bar: 0-3
NUM_SUBDIVS = 5        # subdivision_index: 0-4  (round can produce 4)
NUM_DUR_CLASSES = 8    # |DURATION_VOCAB|
NUM_COMPOSITE = NUM_BEATS * NUM_SUBDIVS * NUM_DUR_CLASSES  # 160


# ------------------------------------------------------------------
# helpers to encode / decode the composite label
# ------------------------------------------------------------------
def encode_composite(beat: int, subdiv: int, dur_class: int) -> int:
    """(beat, subdiv, dur_class) → single int label."""
    return beat * (NUM_SUBDIVS * NUM_DUR_CLASSES) + subdiv * NUM_DUR_CLASSES + dur_class


def decode_composite(label: int) -> tuple[int, int, int]:
    """Single int label → (beat, subdiv, dur_class)."""
    dur_class = label % NUM_DUR_CLASSES
    remainder = label // NUM_DUR_CLASSES
    subdiv = remainder % NUM_SUBDIVS
    beat = remainder // NUM_SUBDIVS
    return beat, subdiv, dur_class


def targets_to_composite(targets: torch.Tensor) -> torch.Tensor:
    """
    Convert (…, 3) target tensor  →  (…,) composite class indices.
    Expects columns [beat_index, subdivision_index, duration_class_idx].
    """
    beat = targets[..., 0].clamp(0, NUM_BEATS - 1)
    subdiv = targets[..., 1].clamp(0, NUM_SUBDIVS - 1)
    dur = targets[..., 2].clamp(0, NUM_DUR_CLASSES - 1)
    return beat * (NUM_SUBDIVS * NUM_DUR_CLASSES) + subdiv * NUM_DUR_CLASSES + dur


def composite_to_targets(composite: torch.Tensor) -> torch.Tensor:
    """
    Inverse of targets_to_composite.
    (…,) composite → (…, 3) with columns [beat, subdiv, dur_class].
    """
    dur = composite % NUM_DUR_CLASSES
    remainder = composite // NUM_DUR_CLASSES
    subdiv = remainder % NUM_SUBDIVS
    beat = remainder // NUM_SUBDIVS
    return torch.stack([beat, subdiv, dur], dim=-1)


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class RhythmQuantizer(nn.Module):
    """
    BiLSTM encoder → single linear classification head.
    Operates on variable-length note sequences.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = NUM_COMPOSITE,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)  padded input sequences
        lengths : (B,) optional true lengths for packing

        Returns
        -------
        logits : (B, T, num_classes)
        """
        x = self.input_norm(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            h, _ = self.lstm(x)

        logits = self.head(h)  # (B, T, C)
        return logits


# ------------------------------------------------------------------
# Multi-head Model
# ------------------------------------------------------------------
class MultiHeadRhythmQuantizer(nn.Module):
    """
    BiLSTM encoder → three independent classification heads for
    beat (4), subdivision (5), and duration (8).

    This reduces the output space from 160 composite classes to
    4 + 5 + 8 = 17 independent classes, making learning much easier.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        enc_dim = hidden_dim * 2  # bidirectional

        # Shared projection from LSTM output
        self.shared = nn.Sequential(
            nn.Linear(enc_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Independent heads
        self.beat_head = nn.Linear(hidden_dim, NUM_BEATS)
        self.subdiv_head = nn.Linear(hidden_dim, NUM_SUBDIVS)
        self.dur_head = nn.Linear(hidden_dim, NUM_DUR_CLASSES)

    def _encode(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Run LSTM encoder and shared projection. Returns (B, T, hidden_dim)."""
        x = self.input_norm(x)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            h, _ = self.lstm(x)

        return self.shared(h)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x       : (B, T, input_dim)
        lengths : (B,) optional true lengths for packing

        Returns
        -------
        dict with keys 'beat', 'subdiv', 'dur', each (B, T, num_classes_i)
        """
        h = self._encode(x, lengths)
        return {
            "beat": self.beat_head(h),      # (B, T, 4)
            "subdiv": self.subdiv_head(h),  # (B, T, 5)
            "dur": self.dur_head(h),        # (B, T, 8)
        }
