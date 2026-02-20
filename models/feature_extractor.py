"""
Pretrained audio feature extractor (wav2vec2 / MERT).

Wraps a HuggingFace Wav2Vec2-style model and projects its frame-level
hidden states onto a fixed-length grid of rhythmic slots via interpolation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor


class AudioFeatureExtractor(nn.Module):
    """Extract frame-level features from raw waveforms using a pretrained model.

    The pretrained encoder produces features at ~50 Hz (20 ms hop for wav2vec2).
    We linearly interpolate along the time axis to match the number of rhythmic
    grid slots so that each slot gets a feature vector.
    """

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze: bool = True,
        sample_rate: int = 16_000,
    ):
        super().__init__()
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.encoder = Wav2Vec2Model.from_pretrained(model_name)
        self.sample_rate = sample_rate
        self.hidden_size = self.encoder.config.hidden_size  # e.g. 768

        if freeze:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------

    def forward(
        self,
        waveforms: torch.Tensor,           # (B, num_audio_samples)
        num_grid_slots: int | None = None,  # target temporal resolution
    ) -> torch.Tensor:
        """
        Returns
        -------
        features : (B, T, D)  where T = num_grid_slots (or encoder frames if
                   num_grid_slots is None) and D = hidden_size.
        """
        # Wav2Vec2Model expects float tensors on the same device
        # The HF processor is CPU-only; we normalise ourselves on-device.
        with torch.no_grad():
            # Normalise to zero-mean unit-variance per sample (like the processor)
            waveforms = (waveforms - waveforms.mean(dim=-1, keepdim=True)) / (
                waveforms.std(dim=-1, keepdim=True) + 1e-7
            )

        outputs = self.encoder(waveforms, return_dict=True)
        hidden = outputs.last_hidden_state    # (B, T_enc, D)

        if num_grid_slots is not None and hidden.shape[1] != num_grid_slots:
            # Interpolate along time: (B, D, T_enc) → (B, D, num_grid_slots)
            hidden = hidden.permute(0, 2, 1)
            hidden = F.interpolate(
                hidden, size=num_grid_slots, mode="linear", align_corners=False
            )
            hidden = hidden.permute(0, 2, 1)   # back to (B, T, D)

        return hidden
