import torch
from parse_asap_data import build_dataset, DURATION_VOCAB, NULL_DURATION_IDX, NULL_SUBDIV

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
        # Remap null tokens to -100 so CrossEntropyLoss(ignore_index=-100) masks them.
        # The sequence start token uses -100 for beat_index_in_bar as well, since
        # it carries no note-level prediction target.
        dur_idx    = DURATION_VOCAB[tgt["duration_class"]]
        subdiv     = tgt["subdivision_index"]
        beat_index = tgt["beat_index_in_bar"]

        y_vec = [
            -100 if beat_index == -100 else beat_index,        # 0–3, or masked (start token)
            -100 if subdiv  == NULL_SUBDIV      else subdiv,   # 0–11, or masked
            -100 if dur_idx == NULL_DURATION_IDX else dur_idx, # 0–10, or masked
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

    quantized_X, quantized_Y = build_dataset(root, isQuantized=True)
    print("Pieces parsed:", len(quantized_X))

    quantized_dataset = build_torch_dataset(quantized_X, quantized_Y)

    torch.save({
        "sequences": quantized_dataset,
        "duration_vocab": DURATION_VOCAB
    }, "quantized_dataset.pt")

    print("Saved dataset to quantized_dataset.pt")