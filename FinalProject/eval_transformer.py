"""Evaluate a saved Transformer checkpoint on train and val sets.

Usage
-----
    python eval_transformer.py --checkpoint checkpoints/rhythm_transformer_XXXXXXXX.pt
"""
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import TransformerRhythmQuantizer, NUM_BEATS, NUM_SUBDIVS, NUM_DUR_CLASSES
from train_transformer import MultiHeadRhythmDataset, collate_fn
from parse_asap_data import IDX_TO_DURATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dataset.pt")
    cli = parser.parse_args()

    ckpt = torch.load(cli.checkpoint, weights_only=False)
    args = ckpt["args"]
    print(f"Checkpoint from epoch {ckpt['epoch']}")
    print(f"Val metrics at save: {ckpt['val_metrics']}")

    raw = torch.load(cli.dataset, weights_only=False)
    sequences = raw["sequences"]
    print(f"Total sequences: {len(sequences)}")

    np.random.seed(args["seed"])
    indices = np.random.permutation(len(sequences))
    n_val = max(1, int(len(sequences) * args["val_split"]))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    train_seqs = [sequences[i] for i in train_idx]
    val_seqs = [sequences[i] for i in val_idx]

    max_seq_len = args["max_seq_len"] if args["max_seq_len"] > 0 else None
    train_ds = MultiHeadRhythmDataset(train_seqs, max_seq_len=max_seq_len)
    val_ds = MultiHeadRhythmDataset(val_seqs, max_seq_len=max_seq_len)

    loader_kw = dict(batch_size=args["batch_size"], collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    device = torch.device("cpu")
    model = TransformerRhythmQuantizer(
        input_dim=7,
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        nhead=args["nhead"],
        dropout=args["dropout"],
        dim_feedforward=args["dim_feedforward"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def evaluate(loader, label):
        all_bp, all_sp, all_dp = [], [], []
        all_bt, all_st, all_dt = [], [], []
        with torch.no_grad():
            for x_pad, beat_tgt, sub_tgt, dur_tgt, lengths in loader:
                logits = model(x_pad.to(device), lengths.to(device))
                all_bp.append(logits["beat"].argmax(-1).cpu())
                all_sp.append(logits["subdiv"].argmax(-1).cpu())
                all_dp.append(logits["dur"].argmax(-1).cpu())
                all_bt.append(beat_tgt)
                all_st.append(sub_tgt)
                all_dt.append(dur_tgt)

        bp = torch.cat(all_bp); sp = torch.cat(all_sp); dp = torch.cat(all_dp)
        bt = torch.cat(all_bt); st = torch.cat(all_st); dt = torch.cat(all_dt)

        mask = bt != -100
        total = mask.sum().item()
        n = max(total, 1)

        beat_acc = ((bp == bt) & mask).sum().item() / n
        sub_acc = ((sp == st) & mask).sum().item() / n
        dur_acc = ((dp == dt) & mask).sum().item() / n
        joint_acc = ((bp == bt) & (sp == st) & (dp == dt) & mask).sum().item() / n

        print(
            f"{label}:  joint_acc={joint_acc:.4f}  beat_acc={beat_acc:.4f}  "
            f"subdiv_acc={sub_acc:.4f}  dur_acc={dur_acc:.4f}  ({total:,} notes)"
        )

    print()
    evaluate(train_loader, "Train")
    evaluate(val_loader, "Val  ")

    print("\n--- Sample predictions (first 10 notes of first val sequence) ---")
    with torch.no_grad():
        sample = val_seqs[0]
        x = sample["inputs"].unsqueeze(0).to(device)
        y3 = sample["targets"]
        logits = model(x)
        bp = logits["beat"][0].argmax(-1).cpu()
        sp = logits["subdiv"][0].argmax(-1).cpu()
        dp = logits["dur"][0].argmax(-1).cpu()

        for i in range(min(10, len(bp))):
            pb, ps, pd_ = bp[i].item(), sp[i].item(), dp[i].item()
            tb, ts, td = int(y3[i, 0]), int(y3[i, 1]), int(y3[i, 2])
            print(
                f"  Note {i}: "
                f"pred=(beat={pb}, sub={ps}, dur={IDX_TO_DURATION.get(pd_, '?')})  "
                f"true=(beat={tb}, sub={ts}, dur={IDX_TO_DURATION.get(td, '?')})"
            )


if __name__ == "__main__":
    main()