import os
import json
import numpy as np
from pathlib import Path
import partitura
import warnings
# import csv

# -----------------------------
# Utility: grab all the piece directories (those containing .mid files)
# -----------------------------
def find_piece_dirs(root):
    piece_dirs = []
    for path, dirs, files in os.walk(root):
        mids = [f for f in files if f.endswith(".mid")]
        if mids:
            piece_dirs.append(Path(path))
    return piece_dirs


# -----------------------------
# Utility: load beat map json
# -----------------------------
def load_annotations(json_path):
    with open(json_path, "r") as f:
        return json.load(f)


# -----------------------------
# Utility: build time→beat mapping
# -----------------------------
def build_time_to_beat_fn(beat_times):
    beat_times = np.array(beat_times)
    beat_indices = np.arange(len(beat_times))

    def f(t):
        if t <= beat_times[0]:
            return 0.0
        if t >= beat_times[-1]:
            return float(len(beat_times) - 1)

        i = np.searchsorted(beat_times, t) - 1
        t0, t1 = beat_times[i], beat_times[i + 1]
        b0, b1 = beat_indices[i], beat_indices[i + 1]
        return b0 + (t - t0) / (t1 - t0)

    return f


# -----------------------------
# Pitch encoding
# -----------------------------
def encode_pitch(pitch):
    pitch_class = pitch % 12
    octave = pitch // 12
    return {
        "pc_sin": np.sin(2 * np.pi * pitch_class / 12),
        "pc_cos": np.cos(2 * np.pi * pitch_class / 12),
        "octave_norm": octave / 10.0
    }


# -----------------------------
# Duration classification
# -----------------------------
DURATION_CLASSES = {
    0.25: "16th",
    0.5: "8th",
    0.75: "dotted_8th",
    1.0: "quarter",
    1.5: "dotted_quarter",
    2.0: "half",
    3.0: "dotted_half",
    4.0: "whole"
}

# Build duration label → index mapping
DURATION_LABELS = sorted(set(DURATION_CLASSES.values()))
DURATION_VOCAB = {label: idx for idx, label in enumerate(DURATION_LABELS)}

# Optional reverse mapping (useful later for decoding predictions)
IDX_TO_DURATION = {idx: label for label, idx in DURATION_VOCAB.items()}

def quantize_duration(d):
    closest = min(DURATION_CLASSES.keys(), key=lambda x: abs(x - d))
    return DURATION_CLASSES[closest]


# -----------------------------
# Subdivision helper
# -----------------------------
def beat_to_bar_position(beat_float, beats_per_bar=4, subdivs=4):
    beat_index = int(beat_float) % beats_per_bar
    frac = beat_float - int(beat_float)
    subdivision = int(round(frac * subdivs))
    return beat_index, subdivision


# -----------------------------
# Core parser for one performance
# -----------------------------
def parse_performance(piece_root, perf_mid, xml_score, alignment_file, beat_map):

    # Load performance + score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        performance = partitura.load_performance_midi(perf_mid)
        score = partitura.load_musicxml(xml_score)

    # Build time->beat converter
    # print(beat_map.keys())
    
    key = str(perf_mid).split("asap-dataset-main")[-1].lstrip("/")
    if key not in beat_map:
        print(f"    WARNING: No beat map found for {key}, skipping performance")
        return [], []

    beat_times = beat_map[key]["performance_beats"]
    t2b = build_time_to_beat_fn(beat_times)

    # Load alignment TSV
    align = {}
    with open(alignment_file) as f:
        next(f)
        for line in f:
            if len(line.strip().split("\t")) < 3:
                continue
            xml_id, midi_id, *_ , onset = line.strip().split("\t")
            base_xml_id = xml_id.split("-")[0]   # removes -1, -2, etc.
            align[str(midi_id)] = base_xml_id

    # Build score lookup
    score_na = score.note_array()
    score_notes = {n["id"]: n for n in score_na}

    inputs = []
    targets = []

    # Extract performance notes
    notes = performance.note_array()

    prev_velocity = None
    prev_onset_beat = None

    for n in notes:

        onset_sec = n["onset_sec"]
        duration_sec = n["duration_sec"]
        offset_sec = onset_sec + duration_sec
        pitch = n["pitch"]
        velocity = n["velocity"] / 127.0
        midi_id = n["id"]

        if midi_id not in align:
            continue
        xml_id = align[midi_id]
        if xml_id not in score_notes:
            continue

        score_note = score_notes[xml_id]

        onset_beat = t2b(onset_sec)
        offset_beat = t2b(offset_sec)
        duration_beat = offset_beat - onset_beat

        if prev_onset_beat is None:
            ioi = 0
        else:
            ioi = onset_beat - prev_onset_beat

        velocity = n["velocity"] / 127.0
        vel_delta = 0 if prev_velocity is None else velocity - prev_velocity
        pitch_enc = encode_pitch(n["pitch"])

        # -------- INPUTS --------
        inp = {
            "onset_ioi_beats": float(ioi),
            "duration_beats": float(duration_beat),
            "velocity": float(velocity),
            "velocity_delta_from_prev": float(vel_delta),
            **pitch_enc
        }

        # -------- TARGETS --------
        beat_index, subdiv = beat_to_bar_position(score_note["onset_beat"])

        duration_class = quantize_duration(score_note["duration_beat"])
    
        tgt = {
            "beat_index_in_bar": beat_index,
            "subdivision_index": subdiv,
            "duration_class": duration_class,
        }

        inputs.append(inp)
        targets.append(tgt)

        prev_velocity = velocity
        prev_onset_beat = onset_beat

    return inputs, targets


# -----------------------------
# Main dataset builder
# -----------------------------
def build_dataset(dataset_root):
    # parse annotations
    dataset_root = Path(dataset_root)
    annotations = load_annotations(dataset_root / "asap_annotations.json")

    inputs_by_piece = {}
    targets_by_piece = {}

    # locate piece directories in possibly non-uniform file structure
    piece_dirs = find_piece_dirs(dataset_root)

    counter = 0
    errors = 0

    for piece in piece_dirs:

        print(f"Parsing piece: {piece}")
        print(f"  {counter+1}/{len(piece_dirs)}")
        counter += 1

        try:

            # parse ground-truth xml
            xml = piece / "xml_score.musicxml"
            if not xml.exists():
                continue

            # parse performance midis (may be multiple from different performers)
            midis = [m for m in piece.glob("*.mid") if "midi_score" not in m.name]
            for perf in midis:

                # find alignment file
                align_dir = piece / f"{perf.stem}_note_alignments"
                align_file = align_dir / "note_alignment.tsv"
                if not align_file.exists():
                    continue

                print(f"  Parsing performance: {perf.name}")

                inputs, targets = parse_performance(
                    dataset_root,
                    perf,
                    xml,
                    align_file,
                    annotations
                )

                print (f"    Extracted {len(inputs)} notes")

                if inputs:
                    key = f"{piece}_{perf.stem}"
                    inputs_by_piece[key] = inputs
                    targets_by_piece[key] = targets

        except Exception as e:
            print(f"  ERROR parsing {piece.name}: {e}")
            errors += 1
            continue

    print(f"Finished parsing dataset. Successfully parsed {counter - errors} pieces, {errors} errors.")

    return inputs_by_piece, targets_by_piece

# def write_dataset_to_csv(inputs_by_piece, targets_by_piece, output_path):
#     with open(output_path, "w", newline="") as f:
#         writer = csv.writer(f)

#         # Write header
#         writer.writerow(["piece_perf_id", "input", "target"])

#         for key in inputs_by_piece:

#             inputs = inputs_by_piece[key]
#             targets = targets_by_piece[key]

#             for inp, tgt in zip(inputs, targets):
#                 writer.writerow([key, inp, tgt])