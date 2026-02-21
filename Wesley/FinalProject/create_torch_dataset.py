import torch
from parse_asap_data import build_dataset, DURATION_VOCAB

def convert_sequence_to_tensors(inputs, targets):
    """
    Converts list-of-dicts → torch tensors
    Returns:
        x: FloatTensor (T, F)
        y: LongTensor (T, 3)
    """

    x_list = []
    y_list = []

    for inp, tgt in zip(inputs, targets):

        # ----- Input features -----
        x_vec = [
            inp["onset_ioi_beats"],
            inp["duration_beats"],
            inp["velocity"],
            inp["velocity_delta_from_prev"],
            inp["pc_sin"],
            inp["pc_cos"],
            inp["octave_norm"],
        ]
        x_list.append(x_vec)

        # ----- Targets -----
        y_vec = [
            tgt["beat_index_in_bar"],                     # 0–3
            tgt["subdivision_index"],                     # 0–3
            DURATION_VOCAB[tgt["duration_class"]]         # categorical index
        ]
        y_list.append(y_vec)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.long)

    return x, y

def build_torch_dataset(inputs_by_piece, targets_by_piece):

    dataset = []

    for key in inputs_by_piece:

        inputs = inputs_by_piece[key]
        targets = targets_by_piece[key]

        if len(inputs) == 0:
            continue

        x, y = convert_sequence_to_tensors(inputs, targets)

        dataset.append({
            "id": key,
            "inputs": x,      # shape (T, 7)
            "targets": y      # shape (T, 3)
        })

    return dataset

# -----------------------------
# CLI usage
# -----------------------------
if __name__ == "__main__":
    import sys
    root = sys.argv[1]

    X, Y = build_dataset(root)
    print("Pieces parsed:", len(X))

    dataset = build_torch_dataset(X, Y)

    torch.save({
        "sequences": dataset,
        "duration_vocab": DURATION_VOCAB
    }, "dataset.pt")

    print("Saved dataset to dataset.pt")