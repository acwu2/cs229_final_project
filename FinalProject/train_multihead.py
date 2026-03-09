"""
Training script for the multi-headed rhythm quantization model.

Instead of predicting 160 composite classes, uses three independent heads
for beat (4), subdivision (5), and duration (8).

Usage
-----
    python train_multihead.py                          # defaults
    python train_multihead.py --epochs 60 --lr 3e-4    # custom
"""

from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from model import (
    MultiHeadRhythmQuantizer,
    decode_composite,
    encode_composite,
    NUM_BEATS,
    NUM_SUBDIVS,
    NUM_DUR_CLASSES,
)
from parse_asap_data import IDX_TO_DURATION


# ===================================================================
# Dataset  (returns 3 separate targets instead of composite)
# ===================================================================
class MultiHeadRhythmDataset(Dataset):
    """
    Each item returns:
        x      : (T, 7) input features
        y_beat : (T,) beat labels  [0..3]
        y_sub  : (T,) subdiv labels [0..4]
        y_dur  : (T,) duration labels [0..7]
    """

    def __init__(self, sequences: list[dict], max_seq_len: int | None = None):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq["inputs"]           # (T, 7)
        y3 = seq["targets"]         # (T, 3) — [beat, subdiv, dur_cls]

        y_beat = y3[:, 0].long().clamp(0, NUM_BEATS - 1)
        y_sub = y3[:, 1].long().clamp(0, NUM_SUBDIVS - 1)
        y_dur = y3[:, 2].long().clamp(0, NUM_DUR_CLASSES - 1)

        if self.max_seq_len is not None:
            x = x[: self.max_seq_len]
            y_beat = y_beat[: self.max_seq_len]
            y_sub = y_sub[: self.max_seq_len]
            y_dur = y_dur[: self.max_seq_len]

        return x, y_beat, y_sub, y_dur


def collate_fn(batch):
    """Pad variable-length sequences; use -100 for padding (ignored by CE)."""
    xs, y_beats, y_subs, y_durs = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])
    max_len = lengths.max().item()

    B = len(xs)
    x_pad = torch.zeros(B, max_len, xs[0].shape[-1])
    beat_pad = torch.full((B, max_len), -100, dtype=torch.long)
    sub_pad = torch.full((B, max_len), -100, dtype=torch.long)
    dur_pad = torch.full((B, max_len), -100, dtype=torch.long)

    for i, (x, yb, ys, yd) in enumerate(zip(xs, y_beats, y_subs, y_durs)):
        T = x.shape[0]
        x_pad[i, :T] = x
        beat_pad[i, :T] = yb
        sub_pad[i, :T] = ys
        dur_pad[i, :T] = yd

    return x_pad, beat_pad, sub_pad, dur_pad, lengths


# ===================================================================
# Metrics
# ===================================================================
@torch.no_grad()
def compute_metrics(
    beat_preds: torch.Tensor,
    sub_preds: torch.Tensor,
    dur_preds: torch.Tensor,
    beat_tgt: torch.Tensor,
    sub_tgt: torch.Tensor,
    dur_tgt: torch.Tensor,
) -> dict[str, float]:
    """Compute per-head and joint accuracy (ignoring -100 padding)."""
    mask = beat_tgt != -100
    total = mask.sum().item()

    beat_correct = ((beat_preds == beat_tgt) & mask).sum().item()
    sub_correct = ((sub_preds == sub_tgt) & mask).sum().item()
    dur_correct = ((dur_preds == dur_tgt) & mask).sum().item()

    all_correct = (
        (beat_preds == beat_tgt) & (sub_preds == sub_tgt) & (dur_preds == dur_tgt) & mask
    ).sum().item()

    n = max(total, 1)
    return {
        "acc": all_correct / n,
        "beat_acc": beat_correct / n,
        "subdiv_acc": sub_correct / n,
        "dur_acc": dur_correct / n,
    }


# ===================================================================
# Train / eval loops
# ===================================================================
_amp_scaler: torch.amp.GradScaler | None = None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    global _amp_scaler
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    if loss_weights is None:
        loss_weights = {"beat": 1.0, "subdiv": 1.0, "dur": 1.0}

    use_amp = device.type == "cuda"
    if use_amp and _amp_scaler is None:
        _amp_scaler = torch.amp.GradScaler()

    total_loss = 0.0
    n_samples = 0

    all_beat_preds: list[torch.Tensor] = []
    all_sub_preds: list[torch.Tensor] = []
    all_dur_preds: list[torch.Tensor] = []
    all_beat_tgt: list[torch.Tensor] = []
    all_sub_tgt: list[torch.Tensor] = []
    all_dur_tgt: list[torch.Tensor] = []

    desc = "Train" if is_train else "Val"
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_pad, beat_tgt, sub_tgt, dur_tgt, lengths in tqdm(loader, desc=desc, leave=False):
            x_pad = x_pad.to(device, non_blocking=True)
            beat_tgt = beat_tgt.to(device, non_blocking=True)
            sub_tgt = sub_tgt.to(device, non_blocking=True)
            dur_tgt = dur_tgt.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            B = x_pad.shape[0]

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(x_pad, lengths)  # dict of (B, T, Ci)

                # Flatten for cross-entropy
                BT = logits["beat"].shape[0] * logits["beat"].shape[1]
                loss_beat = criterion(
                    logits["beat"].reshape(BT, -1), beat_tgt.reshape(BT)
                )
                loss_sub = criterion(
                    logits["subdiv"].reshape(BT, -1), sub_tgt.reshape(BT)
                )
                loss_dur = criterion(
                    logits["dur"].reshape(BT, -1), dur_tgt.reshape(BT)
                )

                loss = (
                    loss_weights["beat"] * loss_beat
                    + loss_weights["subdiv"] * loss_sub
                    + loss_weights["dur"] * loss_dur
                )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    _amp_scaler.scale(loss).backward()
                    _amp_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    _amp_scaler.step(optimizer)
                    _amp_scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * B
            n_samples += B

            # Store preds for end-of-epoch metrics
            all_beat_preds.append(logits["beat"].detach().argmax(dim=-1).cpu())
            all_sub_preds.append(logits["subdiv"].detach().argmax(dim=-1).cpu())
            all_dur_preds.append(logits["dur"].detach().argmax(dim=-1).cpu())
            all_beat_tgt.append(beat_tgt.cpu())
            all_sub_tgt.append(sub_tgt.cpu())
            all_dur_tgt.append(dur_tgt.cpu())

    # Concatenate and compute metrics
    beat_preds = torch.cat(all_beat_preds, 0)
    sub_preds = torch.cat(all_sub_preds, 0)
    dur_preds = torch.cat(all_dur_preds, 0)
    beat_tgt = torch.cat(all_beat_tgt, 0)
    sub_tgt = torch.cat(all_sub_tgt, 0)
    dur_tgt = torch.cat(all_dur_tgt, 0)

    m = compute_metrics(beat_preds, sub_preds, dur_preds, beat_tgt, sub_tgt, dur_tgt)
    m["loss"] = total_loss / max(n_samples, 1)
    return m


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train multi-headed rhythm quantizer")
    parser.add_argument("--dataset", type=str, default="dataset.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    # Loss weights for each head
    parser.add_argument("--w-beat", type=float, default=1.0)
    parser.add_argument("--w-subdiv", type=float, default=1.0)
    parser.add_argument("--w-dur", type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- load data ----
    raw = torch.load(args.dataset, weights_only=False)
    sequences = raw["sequences"]
    duration_vocab = raw["duration_vocab"]
    print(f"Loaded {len(sequences)} sequences from {args.dataset}")
    print(f"Duration vocab: {duration_vocab}")

    total_notes = sum(s["inputs"].shape[0] for s in sequences)
    print(f"Total notes: {total_notes:,}")

    # ---- train / val split ----
    indices = np.random.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * args.val_split))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    print(f"Train sequences: {len(train_idx)}, Val sequences: {len(val_idx)}")

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    train_ds = MultiHeadRhythmDataset(train_seqs, max_seq_len=max_seq_len)
    val_ds = MultiHeadRhythmDataset(val_seqs, max_seq_len=max_seq_len)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # ---- model ----
    model = MultiHeadRhythmQuantizer(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Heads: beat({NUM_BEATS}) + subdiv({NUM_SUBDIVS}) + dur({NUM_DUR_CLASSES})")

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    loss_weights = {"beat": args.w_beat, "subdiv": args.w_subdiv, "dur": args.w_dur}
    print(f"Loss weights: {loss_weights}")

    # ---- output directory ----
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = out / f"rhythm_multihead_{timestamp}.pt"

    # ---- training loop ----
    best_val_acc = -1.0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion, device, optimizer, loss_weights)
        val_m = run_epoch(model, val_loader, criterion, device, loss_weights=loss_weights)

        scheduler.step()
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['acc']:.3f}  "
            f"val_loss={val_m['loss']:.4f}  val_acc={val_m['acc']:.3f}  "
            f"[beat={val_m['beat_acc']:.3f}  sub={val_m['subdiv_acc']:.3f}  "
            f"dur={val_m['dur_acc']:.3f}]  ({elapsed:.1f}s)"
        )

        if val_m["acc"] > best_val_acc:
            best_val_acc = val_m["acc"]
            patience_ctr = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_m,
                "duration_vocab": duration_vocab,
                "model_type": "multihead",
            }, ckpt_path)
            print(f"  -> Saved best checkpoint (val_acc={best_val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {ckpt_path}")

    # ---- quick inference demo ----
    print("\n--- Sample predictions (first 10 notes of first val sequence) ---")
    model.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
    model.eval()
    with torch.no_grad():
        sample = val_seqs[0]
        x = sample["inputs"].unsqueeze(0).to(device)
        y3 = sample["targets"]   # (T, 3)
        logits = model(x)
        beat_preds = logits["beat"][0].argmax(dim=-1).cpu()
        sub_preds = logits["subdiv"][0].argmax(dim=-1).cpu()
        dur_preds = logits["dur"][0].argmax(dim=-1).cpu()

        for i in range(min(10, len(beat_preds))):
            pb, ps, pd = beat_preds[i].item(), sub_preds[i].item(), dur_preds[i].item()
            tb, ts, td = int(y3[i, 0]), int(y3[i, 1]), int(y3[i, 2])
            pred_dur = IDX_TO_DURATION.get(pd, "?")
            true_dur = IDX_TO_DURATION.get(td, "?")
            print(
                f"  Note {i}: "
                f"pred=(beat={pb}, sub={ps}, dur={pred_dur})  "
                f"true=(beat={tb}, sub={ts}, dur={true_dur})"
            )


if __name__ == "__main__":
    main()
