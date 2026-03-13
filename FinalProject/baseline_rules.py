"""
Rules-based rhythm quantization baseline.

Strategy — "snap to nearest grid point":
    1. Beat index in bar:   accumulate IOIs to get a cumulative beat
                            position, then floor(pos) % 4.
    2. Subdivision index:   round the fractional part of the beat
                            position to the nearest 1/4-beat grid
                            (indices 0-4).
    3. Duration class:      snap duration_beats to the closest entry
                            in DURATION_CLASSES.

Evaluated on the exact same dataset / split / metrics used by the
BiLSTM model so the numbers are directly comparable.

Usage
-----
    python baseline_rules.py                       # defaults
    python baseline_rules.py --dataset dataset.pt  # explicit path
"""

from __future__ import annotations

import argparse
import numpy as np
import torch

from model import (
    targets_to_composite,
    composite_to_targets,
    decode_composite,
    NUM_BEATS,
    NUM_SUBDIVS,
    NUM_DUR_CLASSES,
)
from parse_asap_data import DURATION_CLASSES, DURATION_VOCAB, IDX_TO_DURATION


# ------------------------------------------------------------------
# Core rules
# ------------------------------------------------------------------
SUBDIVISIONS = 12          # quarter-beat grid (same as beat_to_bar_position)
SORTED_DUR_KEYS = sorted(DURATION_CLASSES.keys())   # [0.25, 0.5, …, 4.0]


def snap_duration(dur_beats: float) -> int:
    """Return the DURATION_VOCAB index for the closest duration class."""
    closest_key = min(SORTED_DUR_KEYS, key=lambda k: abs(k - dur_beats))
    label = DURATION_CLASSES[closest_key]
    return DURATION_VOCAB[label]


def rules_predict_sequence(inputs: torch.Tensor) -> torch.Tensor:
    """
    Apply deterministic rules to one sequence of performance features.

    Parameters
    ----------
    inputs : (T, 7) float tensor
        Columns: [onset_ioi_beats, duration_beats, velocity,
                  velocity_delta, pc_sin, pc_cos, octave_norm]

    Returns
    -------
    preds : (T, 3) long tensor   [beat_index, subdiv_index, dur_class_idx]
    """
    T = inputs.shape[0]
    ioi = inputs[:, 0].numpy()          # onset_ioi_beats
    dur = inputs[:, 1].numpy()          # duration_beats

    preds = np.zeros((T, 3), dtype=np.int64)

    cum_beat = 0.0                      # running beat position

    for i in range(T):
        cum_beat += ioi[i]

        # --- beat index in bar ---
        beat_idx = int(cum_beat) % NUM_BEATS

        # --- subdivision index ---
        frac = cum_beat - int(cum_beat)
        subdiv = int(round(frac * SUBDIVISIONS))
        subdiv = min(subdiv, NUM_SUBDIVS - 1)

        # --- duration class ---
        dur_cls = snap_duration(float(dur[i]))

        preds[i] = [beat_idx, subdiv, dur_cls]

    return torch.tensor(preds, dtype=torch.long)


# ------------------------------------------------------------------
# Evaluation helpers  (mirror train_model.py metrics)
# ------------------------------------------------------------------
def evaluate(sequences: list[dict]) -> dict[str, float]:
    """Run the rules baseline on a set of sequences and return metrics."""
    total = 0
    beat_correct = 0
    subdiv_correct = 0
    dur_correct = 0
    composite_correct = 0

    for seq in sequences:
        x = seq["inputs"]       # (T, 7)
        y = seq["targets"]      # (T, 3)  [beat, subdiv, dur_cls]

        pred = rules_predict_sequence(x)  # (T, 3)
        T = x.shape[0]
        total += T

        beat_correct += (pred[:, 0] == y[:, 0]).sum().item()
        subdiv_correct += (pred[:, 1] == y[:, 1]).sum().item()
        dur_correct += (pred[:, 2] == y[:, 2]).sum().item()

        # composite accuracy (all three correct simultaneously)
        composite_correct += ((pred == y).all(dim=1)).sum().item()

    n = max(total, 1)
    return {
        "acc": composite_correct / n,
        "beat_acc": beat_correct / n,
        "subdiv_acc": subdiv_correct / n,
        "dur_acc": dur_correct / n,
        "total_notes": total,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Rules-based rhythm quantization baseline")
    parser.add_argument("--dataset", type=str, default="dataset.pt")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- load data ----
    raw = torch.load(args.dataset, weights_only=False)
    sequences = raw["sequences"]
    print(f"Loaded {len(sequences)} sequences from {args.dataset}")

    # ---- same train/val split as train_model.py ----
    np.random.seed(args.seed)
    indices = np.random.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * args.val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]

    # ---- evaluate ----
    print("\n=== Rules-based baseline ===")
    for split_name, split_seqs in [("Train", train_seqs), ("Val", val_seqs)]:
        m = evaluate(split_seqs)
        print(
            f"{split_name:5s}  "
            f"acc={m['acc']:.4f}  "
            f"beat={m['beat_acc']:.4f}  "
            f"subdiv={m['subdiv_acc']:.4f}  "
            f"dur={m['dur_acc']:.4f}  "
            f"({m['total_notes']:,} notes)"
        )

    # ---- sample predictions ----
    print("\n--- Sample predictions (first 10 notes of first val sequence) ---")
    sample = val_seqs[0]
    x = sample["inputs"]
    y = sample["targets"]
    pred = rules_predict_sequence(x)
    for i in range(min(10, x.shape[0])):
        p = pred[i].tolist()
        t = y[i].tolist()
        pred_dur = IDX_TO_DURATION.get(p[2], "?")
        true_dur = IDX_TO_DURATION.get(int(t[2]), "?")
        print(
            f"  Note {i}: "
            f"pred=(beat={p[0]}, sub={p[1]}, dur={pred_dur})  "
            f"true=(beat={int(t[0])}, sub={int(t[1])}, dur={true_dur})"
        )


if __name__ == "__main__":
    main()
