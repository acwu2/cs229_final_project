"""
Training & evaluation loops for the rhythm quantization baseline.
"""

from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.metrics import frame_metrics, tolerant_f1


# ---------------------------------------------------------------------------
# Single epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_grid_slots: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        waveforms = batch["waveform"].to(device)        # (B, N)
        targets = batch["onsets"].to(device)             # (B, S)

        logits = model(waveforms, num_grid_slots=num_grid_slots)  # (B, S)

        # targets may differ in length from logits if BPM varies;
        # truncate / pad to match
        T = logits.shape[1]
        if targets.shape[1] > T:
            targets = targets[:, :T]
        elif targets.shape[1] < T:
            pad = torch.zeros(targets.shape[0], T - targets.shape[1], device=device)
            targets = torch.cat([targets, pad], dim=1)

        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * waveforms.shape[0]
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    all_logits_arr = np.concatenate(all_logits)
    all_targets_arr = np.concatenate(all_targets)

    metrics = frame_metrics(all_logits_arr, all_targets_arr)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_grid_slots: int,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_logits: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        waveforms = batch["waveform"].to(device)
        targets = batch["onsets"].to(device)

        logits = model(waveforms, num_grid_slots=num_grid_slots)

        T = logits.shape[1]
        if targets.shape[1] > T:
            targets = targets[:, :T]
        elif targets.shape[1] < T:
            pad = torch.zeros(targets.shape[0], T - targets.shape[1], device=device)
            targets = torch.cat([targets, pad], dim=1)

        loss = criterion(logits, targets)
        total_loss += loss.item() * waveforms.shape[0]

        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_logits_arr = np.concatenate(all_logits)
    all_targets_arr = np.concatenate(all_targets)

    metrics = frame_metrics(all_logits_arr, all_targets_arr)
    tol = tolerant_f1(all_logits_arr, all_targets_arr, tolerance=1)
    metrics.update(tol)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


# ---------------------------------------------------------------------------
# Full training routine
# ---------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    pos_weight: float,
    device: torch.device,
    num_grid_slots: int,
    output_dir: Path,
    head_type: str = "mlp",
    early_stop_patience: int = 5,
) -> dict[str, list]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create a timestamped checkpoint name so retraining never overwrites old ones
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"best_model_{head_type}_{timestamp}.pt"

    # Only optimise non-frozen params
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    pw = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    history: dict[str, list] = {"train": [], "val": []}
    best_val_f1 = -1.0
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device, num_grid_slots)
        val_m = evaluate(model, val_loader, criterion, device, num_grid_slots)

        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{num_epochs}  "
            f"train_loss={train_m['loss']:.4f}  train_f1={train_m['f1']:.3f}  "
            f"val_loss={val_m['loss']:.4f}  val_f1={val_m['f1']:.3f}  "
            f"val_f1_tol={val_m['f1_tol']:.3f}  "
            f"({elapsed:.1f}s)"
        )

        # ---- checkpointing & early stop ----
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / ckpt_name)
            print(f"  ✓ Saved best model to {ckpt_name} (val_f1={best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    return history, ckpt_name
