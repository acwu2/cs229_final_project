"""
Training script for the Transformer rhythm quantization model.
Adds cumulative IOI as an explicit 8th input feature to help beat tracking.

Usage
-----
    python train_transformer.py                        # defaults
    python train_transformer.py --w-beat 2.0           # upweight beat loss
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

from model import TransformerRhythmQuantizer, NUM_BEATS, NUM_SUBDIVS, NUM_DUR_CLASSES
from parse_asap_data import IDX_TO_DURATION

INPUT_DIM = 8   # 7 original + cumulative IOI


# ===================================================================
# Dataset — adds cumulative IOI as 8th feature on the fly
# ===================================================================
class MultiHeadRhythmDataset(Dataset):
    def __init__(self, sequences: list[dict], max_seq_len: int | None = None):
        self.sequences = sequences
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq["inputs"]           # (T, 7)
        y3 = seq["targets"]         # (T, 3)

        # Cumulative IOI: running sum of onset_ioi_beats (col 0)
        cum_ioi = (torch.cumsum(x[:, 0], dim=0) % 4).unsqueeze(1)  # (T, 1)
        x = torch.cat([x, cum_ioi], dim=1)                   # (T, 8)

        y_beat = y3[:, 0].long().clamp(0, NUM_BEATS - 1)
        y_sub  = y3[:, 1].long().clamp(0, NUM_SUBDIVS - 1)
        y_dur  = y3[:, 2].long().clamp(0, NUM_DUR_CLASSES - 1)

        if self.max_seq_len is not None:
            x      = x[: self.max_seq_len]
            y_beat = y_beat[: self.max_seq_len]
            y_sub  = y_sub[: self.max_seq_len]
            y_dur  = y_dur[: self.max_seq_len]

        return x, y_beat, y_sub, y_dur


def collate_fn(batch):
    xs, y_beats, y_subs, y_durs = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs])
    max_len = lengths.max().item()
    B = len(xs)
    x_pad    = torch.zeros(B, max_len, xs[0].shape[-1])
    beat_pad = torch.full((B, max_len), -100, dtype=torch.long)
    sub_pad  = torch.full((B, max_len), -100, dtype=torch.long)
    dur_pad  = torch.full((B, max_len), -100, dtype=torch.long)
    for i, (x, yb, ys, yd) in enumerate(zip(xs, y_beats, y_subs, y_durs)):
        T = x.shape[0]
        x_pad[i, :T]    = x
        beat_pad[i, :T] = yb
        sub_pad[i, :T]  = ys
        dur_pad[i, :T]  = yd
    return x_pad, beat_pad, sub_pad, dur_pad, lengths


# ===================================================================
# Metrics
# ===================================================================
@torch.no_grad()
def compute_metrics(bp, sp, dp, bt, st, dt):
    mask = bt != -100
    n = max(mask.sum().item(), 1)
    return {
        "acc":        ((bp == bt) & (sp == st) & (dp == dt) & mask).sum().item() / n,
        "beat_acc":   ((bp == bt) & mask).sum().item() / n,
        "subdiv_acc": ((sp == st) & mask).sum().item() / n,
        "dur_acc":    ((dp == dt) & mask).sum().item() / n,
    }


# ===================================================================
# Train / eval loops
# ===================================================================
def run_epoch(model, loader, criterion, device, optimizer=None, loss_weights=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    if loss_weights is None:
        loss_weights = {"beat": 1.0, "subdiv": 1.0, "dur": 1.0}

    total_loss, n_samples = 0.0, 0
    all_bp, all_sp, all_dp = [], [], []
    all_bt, all_st, all_dt = [], [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x_pad, beat_tgt, sub_tgt, dur_tgt, lengths in tqdm(
            loader, desc="Train" if is_train else "Val", leave=False
        ):
            x_pad    = x_pad.to(device)
            beat_tgt = beat_tgt.to(device)
            sub_tgt  = sub_tgt.to(device)
            dur_tgt  = dur_tgt.to(device)
            lengths  = lengths.to(device)

            logits = model(x_pad, lengths)
            BT = logits["beat"].shape[0] * logits["beat"].shape[1]

            loss = (
                loss_weights["beat"]   * criterion(logits["beat"].reshape(BT, -1),   beat_tgt.reshape(BT))
                + loss_weights["subdiv"] * criterion(logits["subdiv"].reshape(BT, -1), sub_tgt.reshape(BT))
                + loss_weights["dur"]    * criterion(logits["dur"].reshape(BT, -1),    dur_tgt.reshape(BT))
            )

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x_pad.shape[0]
            n_samples  += x_pad.shape[0]

            all_bp.append(logits["beat"].detach().argmax(-1).cpu())
            all_sp.append(logits["subdiv"].detach().argmax(-1).cpu())
            all_dp.append(logits["dur"].detach().argmax(-1).cpu())
            all_bt.append(beat_tgt.cpu())
            all_st.append(sub_tgt.cpu())
            all_dt.append(dur_tgt.cpu())

    m = compute_metrics(
        torch.cat(all_bp), torch.cat(all_sp), torch.cat(all_dp),
        torch.cat(all_bt), torch.cat(all_st), torch.cat(all_dt),
    )
    m["loss"] = total_loss / max(n_samples, 1)
    return m


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",        type=str,   default="dataset.pt")
    parser.add_argument("--epochs",         type=int,   default=60)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch-size",     type=int,   default=32)
    parser.add_argument("--max-seq-len",    type=int,   default=512)
    parser.add_argument("--num-workers",    type=int,   default=0)
    parser.add_argument("--hidden-dim",     type=int,   default=128)
    parser.add_argument("--num-layers",     type=int,   default=4)
    parser.add_argument("--nhead",          type=int,   default=4)
    parser.add_argument("--dim-feedforward",type=int,   default=512)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--weight-decay",   type=float, default=1e-4)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--val-split",      type=float, default=0.15)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output-dir",     type=str,   default="checkpoints")
    parser.add_argument("--w-beat",         type=float, default=1.0)
    parser.add_argument("--w-subdiv",       type=float, default=1.0)
    parser.add_argument("--w-dur",          type=float, default=1.0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    raw       = torch.load(args.dataset, weights_only=False)
    sequences = raw["sequences"]
    print(f"Loaded {len(sequences)} sequences  |  input_dim={INPUT_DIM} (7 + cumulative IOI)")

    indices  = np.random.permutation(len(sequences))
    n_val    = max(1, int(len(sequences) * args.val_split))
    val_seqs   = [sequences[i] for i in indices[:n_val]]
    train_seqs = [sequences[i] for i in indices[n_val:]]

    max_seq_len = args.max_seq_len if args.max_seq_len > 0 else None
    loader_kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                         num_workers=args.num_workers, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(MultiHeadRhythmDataset(train_seqs, max_seq_len), shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(MultiHeadRhythmDataset(val_seqs,   max_seq_len), shuffle=False, **loader_kwargs)

    model = TransformerRhythmQuantizer(
        input_dim=INPUT_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion    = nn.CrossEntropyLoss(ignore_index=-100)
    loss_weights = {"beat": args.w_beat, "subdiv": args.w_subdiv, "dur": args.w_dur}
    print(f"Loss weights: {loss_weights}")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = out / f"rhythm_transformer_cumioi_{ts}.pt"

    best_val_acc, patience_ctr = -1.0, 0

    for epoch in range(1, args.epochs + 1):
        t0      = time.time()
        train_m = run_epoch(model, train_loader, criterion, device, optimizer, loss_weights)
        val_m   = run_epoch(model, val_loader,   criterion, device, loss_weights=loss_weights)
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_acc={train_m['acc']:.3f}  "
            f"val_loss={val_m['loss']:.4f}  val_acc={val_m['acc']:.3f}  "
            f"[beat={val_m['beat_acc']:.3f}  sub={val_m['subdiv_acc']:.3f}  "
            f"dur={val_m['dur_acc']:.3f}]  ({time.time()-t0:.1f}s)"
        )

        if val_m["acc"] > best_val_acc:
            best_val_acc  = val_m["acc"]
            patience_ctr  = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_metrics": val_m,
                "duration_vocab": raw["duration_vocab"],
                "model_type": "transformer_cumioi",
                "input_dim": INPUT_DIM,
            }, ckpt_path)
            print(f"  -> Saved checkpoint (val_acc={best_val_acc:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nDone. Best val_acc={best_val_acc:.4f}  |  Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()