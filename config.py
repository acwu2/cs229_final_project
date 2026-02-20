"""
Central configuration for the rhythm quantization baseline.
"""
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ---- paths ----
    data_dir: Path = Path("groove_midi")          # root of the Groove MIDI Dataset
    output_dir: Path = Path("outputs")            # checkpoints, logs, etc.

    # ---- audio ----
    sample_rate: int = 16_000                     # wav2vec2 expects 16 kHz
    audio_duration_sec: float = 4.0               # clip length fed to the model
    hop_sec: float = 0.02                         # 20 ms → feature frame rate

    # ---- rhythmic grid ----
    bpm: float = 120.0                            # default BPM (overridden per clip)
    grid_subdivision: int = 16                     # 16th-note grid (per beat)
    # At 120 BPM a 16th note ≈ 0.0625 s.  For a 4-beat bar that gives 16 slots.

    # ---- pretrained encoder ----
    encoder_name: str = "facebook/wav2vec2-base"  # HuggingFace model id
    freeze_encoder: bool = True                   # freeze pretrained weights
    feature_dim: int = 768                        # hidden size of wav2vec2-base

    # ---- classifier head ----
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    num_classes: int = 2                          # 0 = rest, 1 = onset

    # ---- training ----
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    num_epochs: int = 30
    pos_weight: float = 3.0                       # upweight onsets (minority class)
    early_stop_patience: int = 5

    # ---- misc ----
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"                          # will fall back to mps / cpu

    @property
    def grid_interval_sec(self) -> float:
        """Duration of one grid slot in seconds (depends on BPM)."""
        beat_dur = 60.0 / self.bpm
        return beat_dur / (self.grid_subdivision / 4)   # /4 because subdivision is per bar

    @property
    def num_grid_slots(self) -> int:
        """Number of grid slots in the clip."""
        return int(self.audio_duration_sec / self.grid_interval_sec)
