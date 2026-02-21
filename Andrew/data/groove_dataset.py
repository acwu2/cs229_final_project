"""
Dataset & DataLoader for the Groove MIDI Dataset.

The dataset provides:
  - Audio recordings of drum performances (expressive, unquantized timing)
  - Aligned MIDI files with both *expressive* and *quantized* onset annotations

We render the quantized MIDI onto a fixed 16th-note grid to create binary
frame-level labels (onset vs. rest), then pair those labels with the raw
audio waveform that will be fed through a pretrained encoder.

Directory layout expected (after downloading from Magenta):
    groove_midi/
        info.csv                 # metadata
        drummer1/
            session1/
                *.wav
                *.mid
        ...
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Optional heavy imports are deferred so the module can be imported cheaply.
_PRETTY_MIDI = None
_SOUNDFILE = None


def _lazy_import_pretty_midi():
    global _PRETTY_MIDI
    if _PRETTY_MIDI is None:
        import pretty_midi
        _PRETTY_MIDI = pretty_midi
    return _PRETTY_MIDI


def _lazy_import_soundfile():
    global _SOUNDFILE
    if _SOUNDFILE is None:
        import soundfile
        _SOUNDFILE = soundfile
    return _SOUNDFILE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def midi_to_onset_grid(
    midi_path: str | Path,
    bpm: float,
    duration_sec: float,
    grid_subdivision: int = 16,
) -> np.ndarray:
    """Convert a MIDI file to a binary onset vector on a fixed grid.

    Parameters
    ----------
    midi_path : path to a .mid file (the *quantized* version).
    bpm : tempo in beats per minute.
    duration_sec : total duration of the clip.
    grid_subdivision : subdivisions per bar (16 → 16th-note grid in 4/4).

    Returns
    -------
    onsets : np.ndarray of shape (num_slots,) with dtype float32, values {0, 1}.
    """
    pm = _lazy_import_pretty_midi()
    midi = pm.PrettyMIDI(str(midi_path))

    beat_dur = 60.0 / bpm
    slot_dur = beat_dur / (grid_subdivision / 4)  # duration of one grid slot
    num_slots = int(duration_sec / slot_dur)

    onsets = np.zeros(num_slots, dtype=np.float32)

    for instrument in midi.instruments:
        for note in instrument.notes:
            slot_idx = int(round(note.start / slot_dur))
            if 0 <= slot_idx < num_slots:
                onsets[slot_idx] = 1.0

    return onsets


def load_audio_clip(
    audio_path: str | Path,
    sample_rate: int,
    duration_sec: float,
    offset_sec: float = 0.0,
) -> np.ndarray:
    """Load a mono audio clip, zero-padded or truncated to *duration_sec*."""
    sf = _lazy_import_soundfile()
    import librosa

    y, sr = librosa.load(str(audio_path), sr=sample_rate, mono=True,
                         offset=offset_sec, duration=duration_sec)
    target_len = int(sample_rate * duration_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Groove MIDI Pytorch Dataset
# ---------------------------------------------------------------------------

class GrooveMIDIDataset(Dataset):
    """Wraps the Groove MIDI Dataset for the frame-level quantization task.

    Each sample is a dict with:
        waveform  : Tensor  (num_audio_samples,)
        onsets    : Tensor  (num_grid_slots,)  binary labels
        bpm       : float
        file_id   : str     (for debugging)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",            # "train", "validation", or "test"
        sample_rate: int = 16_000,
        duration_sec: float = 4.0,
        grid_subdivision: int = 16,
        default_bpm: float = 120.0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.grid_subdivision = grid_subdivision
        self.default_bpm = default_bpm

        # Fixed grid size based on default BPM so every sample has the same
        # onset-vector length (required for batching with default collate).
        beat_dur = 60.0 / default_bpm
        slot_dur = beat_dur / (grid_subdivision / 4)
        self.num_grid_slots = int(duration_sec / slot_dur)

        self.entries = self._load_metadata(split)
        print(f"[GrooveMIDIDataset] {split}: {len(self.entries)} clips")

    # ---- metadata parsing -------------------------------------------------

    def _load_metadata(self, split: str) -> list[dict]:
        """Parse info.csv and filter by split, keeping only rows with audio."""
        csv_path = self.data_dir / "info.csv"
        entries = []

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("split", "").strip() != split:
                    continue
                audio_fname = row.get("audio_filename", "").strip()
                midi_fname = row.get("midi_filename", "").strip()
                if not audio_fname or not midi_fname:
                    continue

                audio_path = self.data_dir / audio_fname
                midi_path = self.data_dir / midi_fname

                if not audio_path.exists() or not midi_path.exists():
                    continue

                bpm = float(row.get("bpm", self.default_bpm) or self.default_bpm)
                entries.append(dict(
                    audio_path=str(audio_path),
                    midi_path=str(midi_path),
                    bpm=bpm,
                    file_id=audio_fname,
                ))
        return entries

    # ---- __len__ / __getitem__ -------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        waveform = load_audio_clip(
            entry["audio_path"],
            self.sample_rate,
            self.duration_sec,
        )
        onsets = midi_to_onset_grid(
            entry["midi_path"],
            entry["bpm"],
            self.duration_sec,
            self.grid_subdivision,
        )

        # Pad or truncate to the fixed grid length so all samples in a batch
        # have the same size (avoids "Trying to resize storage" in collate).
        T = self.num_grid_slots
        if len(onsets) < T:
            onsets = np.pad(onsets, (0, T - len(onsets)))
        else:
            onsets = onsets[:T]

        return dict(
            waveform=torch.from_numpy(waveform),
            onsets=torch.from_numpy(onsets),
            bpm=entry["bpm"],
            file_id=entry["file_id"],
        )


# ---------------------------------------------------------------------------
# Convenience: build DataLoaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_dir: str | Path,
    batch_size: int = 16,
    sample_rate: int = 16_000,
    duration_sec: float = 4.0,
    grid_subdivision: int = 16,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Return a dict of DataLoaders keyed by split name."""
    loaders = {}
    for split in ("train", "validation", "test"):
        ds = GrooveMIDIDataset(
            data_dir=data_dir,
            split=split,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            grid_subdivision=grid_subdivision,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
