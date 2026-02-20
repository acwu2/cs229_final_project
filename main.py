#!/usr/bin/env python3
"""
main.py – entry point for training & evaluating the rhythm quantization baseline.

Usage
-----
    # Train with default config
    python main.py

    # Override any config field from the CLI
    python main.py --data_dir /path/to/groove_midi --batch_size 32 --head_type bilstm

    # Evaluate a trained checkpoint
    python main.py --eval_only --checkpoint outputs/best_model.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from config import Config
from data.groove_dataset import build_dataloaders
from models.classifier import RhythmQuantizer
from train import train, evaluate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rhythm Quantization Baseline (Approach A)")

    # paths
    p.add_argument("--data_dir", type=str, default=str(Config.data_dir))
    p.add_argument("--output_dir", type=str, default=str(Config.output_dir))

    # model
    p.add_argument("--encoder_name", type=str, default=Config.encoder_name)
    p.add_argument("--freeze_encoder", type=int, default=int(Config.freeze_encoder))
    p.add_argument("--head_type", type=str, default="mlp", choices=["mlp", "bilstm"])
    p.add_argument("--hidden_dim", type=int, default=Config.hidden_dim)
    p.add_argument("--num_layers", type=int, default=Config.num_layers)
    p.add_argument("--dropout", type=float, default=Config.dropout)

    # audio / grid
    p.add_argument("--sample_rate", type=int, default=Config.sample_rate)
    p.add_argument("--audio_duration_sec", type=float, default=Config.audio_duration_sec)
    p.add_argument("--grid_subdivision", type=int, default=Config.grid_subdivision)

    # training
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--learning_rate", type=float, default=Config.learning_rate)
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--num_epochs", type=int, default=Config.num_epochs)
    p.add_argument("--pos_weight", type=float, default=Config.pos_weight)
    p.add_argument("--early_stop_patience", type=int, default=Config.early_stop_patience)

    # eval
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--checkpoint", type=str, default=None)

    # misc
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--num_workers", type=int, default=Config.num_workers)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    loaders = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        sample_rate=args.sample_rate,
        duration_sec=args.audio_duration_sec,
        grid_subdivision=args.grid_subdivision,
        num_workers=args.num_workers,
    )

    # Compute grid slots from default BPM (120) — individual clips may vary
    cfg = Config()
    num_grid_slots = int(args.audio_duration_sec / cfg.grid_interval_sec)
    print(f"Grid slots per clip: {num_grid_slots}  (at {cfg.bpm} BPM, "
          f"{args.grid_subdivision} subdivisions/bar)")

    # ---- model ----
    model = RhythmQuantizer(
        encoder_name=args.encoder_name,
        freeze_encoder=bool(args.freeze_encoder),
        feature_dim=768,  # wav2vec2-base
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_type=args.head_type,
        sample_rate=args.sample_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ---- eval only ----
    if args.eval_only:
        ckpt = args.checkpoint or str(output_dir / "best_model.pt")
        print(f"Loading checkpoint: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=device))

        pw = torch.tensor([args.pos_weight], device=device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        for split in ("validation", "test"):
            if split not in loaders:
                continue
            metrics = evaluate(model, loaders[split], criterion, device, num_grid_slots)
            print(f"\n=== {split.upper()} ===")
            for k, v in metrics.items():
                print(f"  {k:16s}: {v:.4f}")
        return

    # ---- train ----
    history, ckpt_name = train(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        device=device,
        num_grid_slots=num_grid_slots,
        output_dir=output_dir,
        head_type=args.head_type,
        early_stop_patience=args.early_stop_patience,
    )

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # ---- final test evaluation ----
    print("\n" + "=" * 60)
    print(f"Loading best checkpoint for test evaluation: {ckpt_name}")
    model.load_state_dict(torch.load(output_dir / ckpt_name, map_location=device))

    pw = torch.tensor([args.pos_weight], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    test_metrics = evaluate(model, loaders["test"], criterion, device, num_grid_slots)

    print("\n=== TEST ===")
    for k, v in test_metrics.items():
        print(f"  {k:16s}: {v:.4f}")

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
