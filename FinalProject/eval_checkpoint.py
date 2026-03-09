"""Evaluate saved checkpoint on both train and val sets."""
import sys, numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from model import RhythmQuantizer, composite_to_targets, NUM_COMPOSITE
from train_model import RhythmDataset, collate_fn

# Load checkpoint
ckpt = torch.load("checkpoints/rhythm_quantizer_20260220_213513.pt", weights_only=False)
args = ckpt["args"]
print(f"Checkpoint from epoch {ckpt['epoch']}")

# Load dataset
raw = torch.load("dataset.pt", weights_only=False)
sequences = raw["sequences"]
print(f"Total sequences: {len(sequences)}")

# Reproduce same train/val split
np.random.seed(args["seed"])
indices = np.random.permutation(len(sequences))
n_val = max(1, int(len(sequences) * args["val_split"]))
val_idx = indices[:n_val]
train_idx = indices[n_val:]
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

train_seqs = [sequences[i] for i in train_idx]
val_seqs = [sequences[i] for i in val_idx]

max_seq_len = args["max_seq_len"] if args["max_seq_len"] > 0 else None
train_ds = RhythmDataset(train_seqs, max_seq_len=max_seq_len)
val_ds = RhythmDataset(val_seqs, max_seq_len=max_seq_len)

loader_kw = dict(batch_size=args["batch_size"], collate_fn=collate_fn, num_workers=0)
train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)
val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

# Build model & load weights
device = torch.device("cpu")
model = RhythmQuantizer(
    input_dim=7,
    hidden_dim=args["hidden_dim"],
    num_layers=args["num_layers"],
    num_classes=NUM_COMPOSITE,
    dropout=args["dropout"],
).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()


def evaluate(loader, label):
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_pad, y_pad, lengths in loader:
            logits = model(x_pad.to(device), lengths.to(device))
            all_preds.append(logits.argmax(dim=-1).cpu())
            all_targets.append(y_pad.cpu())
    preds = torch.cat(all_preds, 0)
    targets = torch.cat(all_targets, 0)
    mask = targets != -100
    total = mask.sum().item()
    acc = ((preds == targets) & mask).sum().item() / max(total, 1)
    pd = composite_to_targets(preds)
    td = composite_to_targets(targets.clamp(min=0))
    beat_acc = ((pd[..., 0] == td[..., 0]) & mask).sum().item() / max(total, 1)
    subdiv_acc = ((pd[..., 1] == td[..., 1]) & mask).sum().item() / max(total, 1)
    dur_acc = ((pd[..., 2] == td[..., 2]) & mask).sum().item() / max(total, 1)
    print(
        f"{label}:  acc={acc:.4f}  beat_acc={beat_acc:.4f}  "
        f"subdiv_acc={subdiv_acc:.4f}  dur_acc={dur_acc:.4f}"
    )


print()
evaluate(train_loader, "Train")
evaluate(val_loader, "Val  ")
