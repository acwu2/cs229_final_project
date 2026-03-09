"""
Training script for the single-headed rhythm quantization model.

Usage
-----
    python train_model.py                          # defaults
    python train_model.py --epochs 50 --lr 3e-4    # custom
    python train_model.py --dataset dataset.pt     # explicit path
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
    RhythmQuantizer,
    targets_to_composite,
    composite_to_targets,
    decode_composite,
    NUM_BEATS,
    NUM_SUBDIVS,
    NUM_DUR_CLASSES,
    NUM_COMPOSITE,
)
from parse_asap_data import IDX_TO_DURATION


# ===================================================================
# Dataset
# ===================================================================
class RhythmDataset(Dataset):
    """
    Wraps the list-of-sequences format produced by create_torch_dataset.py.
    Each item is a single piece/performance.
    """

    def __init__(self, sequences: list[dict], max_seq_len: int | None = None):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq["inputs"]                          # (T, 7)
        y3 = seq["targets"]                        # (T, 3) — [beat, subdiv, dur_cls]
        y = targets_to_composite(y3)               # (T,)
        if self.max_seq_len is not None:
            x = x[: self.max_seq_len]
            y = y[: self.max_seq_len]
        return x, y


def collate_fn(batch):
    """Pad variable-length sequences and build a mask + lengths tensor."""
    xs, ys = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])
    max_len = lengths.max().item()

    B = len(xs)
    x_pad = torch.zeros(B, max_len, xs[0].shape[-1])
    y_pad = torch.full((B, max_len), -100, dtype=torch.long)  # -100 = ignore in CE

    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[0]
        x_pad[i, :T] = x
        y_pad[i, :T] = y

    return x_pad, y_pad, lengths


# ===================================================================
# Metrics
# ===================================================================
@torch.no_grad()
def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
) -> dict[str, float]:
    """
    Compute overall accuracy + per-component accuracy.

    Parameters
    ----------
    logits  : (B, T, C)
    targets : (B, T) composite labels (-100 for padding)
    lengths : (B,)
    """
    preds = logits.argmax(dim=-1)  # (B, T)

    mask = targets != -100
    correct = (preds == targets) & mask
    total = mask.sum().item()

    acc = correct.sum().item() / max(total, 1)

    # Per-component accuracies
    pred_decomp = composite_to_targets(preds)    # (B,T,3)
    tgt_decomp = composite_to_targets(targets.clamp(min=0))  # clamp padding

    beat_correct = ((pred_decomp[..., 0] == tgt_decomp[..., 0]) & mask).sum().item()
    subdiv_correct = ((pred_decomp[..., 1] == tgt_decomp[..., 1]) & mask).sum().item()
    dur_correct = ((pred_decomp[..., 2] == tgt_decomp[..., 2]) & mask).sum().item()

    return {
        "acc": acc,
        "beat_acc": beat_correct / max(total, 1),
        "subdiv_acc": subdiv_correct / max(total, 1),
        "dur_acc": dur_correct / max(total, 1),
    }


# ===================================================================
# Train / eval loops
# ===================================================================
# AMP scaler (only used when CUDA is available)
_amp_scaler: torch.amp.GradScaler | None = None


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    global _amp_scaler
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    use_amp = device.type == "cuda"
    if use_amp and _amp_scaler is None:
        _amp_scaler = torch.amp.GradScaler()

    total_loss = 0.0
    n_samples = 0
    # accumulate predictions & targets to compute metrics once at the end
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_lengths: list[torch.Tensor] = []

    desc = "Train" if is_train else "Val"
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_pad, y_pad, lengths in tqdm(loader, desc=desc, leave=False):
            x_pad = x_pad.to(device, non_blocking=True)
            y_pad = y_pad.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(x_pad, lengths)  # (B, T, C)
                B, T, C = logits.shape
                loss = criterion(logits.reshape(B * T, C), y_pad.reshape(B * T))

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

            # store lightweight CPU tensors for end-of-epoch metrics
            all_preds.append(logits.detach().argmax(dim=-1).cpu())
            all_targets.append(y_pad.cpu())
            all_lengths.append(lengths.cpu())

    # ---- compute metrics once over the whole epoch ----
    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    lengths = torch.cat(all_lengths, dim=0)

    mask = targets != -100
    total = mask.sum().item()
    acc = ((preds == targets) & mask).sum().item() / max(total, 1)

    pred_decomp = composite_to_targets(preds)
    tgt_decomp = composite_to_targets(targets.clamp(min=0))
    beat_acc = ((pred_decomp[..., 0] == tgt_decomp[..., 0]) & mask).sum().item() / max(total, 1)
    subdiv_acc = ((pred_decomp[..., 1] == tgt_decomp[..., 1]) & mask).sum().item() / max(total, 1)
    dur_acc = ((pred_decomp[..., 2] == tgt_decomp[..., 2]) & mask).sum().item() / max(total, 1)

    return {
        "loss": total_loss / max(n_samples, 1),
        "acc": acc,
        "beat_acc": beat_acc,
        "subdiv_acc": subdiv_acc,
        "dur_acc": dur_acc,
    }


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Train single-headed rhythm quantizer")
    parser.add_argument("--dataset", type=str, default="dataset.pt", help="Path to dataset.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=512, help="Truncate sequences to this many notes (0 = no limit)")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes (0 = main process; >0 can hang on macOS)")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    # ---- reproducibility ----
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

    # ---- quick stats ----
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
    train_ds = RhythmDataset(train_seqs, max_seq_len=max_seq_len)
    val_ds = RhythmDataset(val_seqs, max_seq_len=max_seq_len)

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
    model = RhythmQuantizer(
        input_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=NUM_COMPOSITE,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Composite classes: {NUM_COMPOSITE} "
          f"({NUM_BEATS} beats × {NUM_SUBDIVS} subdivs × {NUM_DUR_CLASSES} durations)")

    # ---- optimizer & scheduler ----
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # ---- output directory ----
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = out / f"rhythm_quantizer_{timestamp}.pt"

    # ---- training loop ----
    best_val_acc = -1.0
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion, device, optimizer)
        val_m = run_epoch(model, val_loader, criterion, device)

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
        y3 = sample["targets"]
        logits = model(x)
        preds = logits[0].argmax(dim=-1).cpu()

        for i in range(min(10, len(preds))):
            pred_parts = decode_composite(preds[i].item())
            true_parts = y3[i].tolist()
            pred_dur = IDX_TO_DURATION.get(pred_parts[2], "?")
            true_dur = IDX_TO_DURATION.get(int(true_parts[2]), "?")
            print(
                f"  Note {i}: "
                f"pred=(beat={pred_parts[0]}, sub={pred_parts[1]}, dur={pred_dur})  "
                f"true=(beat={int(true_parts[0])}, sub={int(true_parts[1])}, dur={true_dur})"
            )


if __name__ == "__main__":
    main()
