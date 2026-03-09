"""Evaluate a saved multi-head checkpoint on train and val sets."""
import sys, argparse, numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, ".")
from model import MultiHeadRhythmQuantizer, NUM_BEATS, NUM_SUBDIVS, NUM_DUR_CLASSES
from train_multihead import MultiHeadRhythmDataset, collate_fn
from parse_asap_data import IDX_TO_DURATION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="dataset.pt")
    cli = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(cli.checkpoint, weights_only=False)
    args = ckpt["args"]
    print(f"Checkpoint from epoch {ckpt['epoch']}")
    print(f"Val metrics at save: {ckpt['val_metrics']}")

    # Load dataset
    raw = torch.load(cli.dataset, weights_only=False)
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
    train_ds = MultiHeadRhythmDataset(train_seqs, max_seq_len=max_seq_len)
    val_ds = MultiHeadRhythmDataset(val_seqs, max_seq_len=max_seq_len)

    loader_kw = dict(batch_size=args["batch_size"], collate_fn=collate_fn, num_workers=0)
    train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # Build model & load weights
    device = torch.device("cpu")
    model = MultiHeadRhythmQuantizer(
        input_dim=7,
        hidden_dim=args["hidden_dim"],
        num_layers=args["num_layers"],
        dropout=args["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def evaluate(loader, label):
        all_beat_p, all_sub_p, all_dur_p = [], [], []
        all_beat_t, all_sub_t, all_dur_t = [], [], []
        with torch.no_grad():
            for x_pad, beat_tgt, sub_tgt, dur_tgt, lengths in loader:
                logits = model(x_pad.to(device), lengths.to(device))
                all_beat_p.append(logits["beat"].argmax(-1).cpu())
                all_sub_p.append(logits["subdiv"].argmax(-1).cpu())
                all_dur_p.append(logits["dur"].argmax(-1).cpu())
                all_beat_t.append(beat_tgt)
                all_sub_t.append(sub_tgt)
                all_dur_t.append(dur_tgt)

        bp = torch.cat(all_beat_p, 0)
        sp = torch.cat(all_sub_p, 0)
        dp = torch.cat(all_dur_p, 0)
        bt = torch.cat(all_beat_t, 0)
        st = torch.cat(all_sub_t, 0)
        dt = torch.cat(all_dur_t, 0)

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

    # Sample predictions
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
            pb, ps, pd = bp[i].item(), sp[i].item(), dp[i].item()
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
