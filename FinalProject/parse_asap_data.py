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
# the mapping returns time since first note, normalized by median bpm
# -----------------------------
def build_time_to_beat_fn(beat_times, first_note_time, first_note_beat):
    beat_times = np.array(beat_times)

    beat_intervals = np.diff(beat_times)
    bpm = 60.0 / np.median(beat_intervals)
    t0 = first_note_time

    def f(t):
        return (t - t0) * bpm / 60.0 + first_note_beat

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
#
# Covers standard 16th-note grid values plus 8th-note triplet values.
# A "null" token handles any duration that doesn't snap within tolerance.
#
# 8th-note triplet durations (in quarter-note beats, where one triplet
# 8th = 1/3 beat):
#   triplet_8th        = 1/3  ≈ 0.333
#   triplet_quarter    = 2/3  ≈ 0.667  (two triplet 8ths tied)
#   triplet_half       = 4/3  ≈ 1.333  (four triplet 8ths tied)
# -----------------------------
DURATION_CLASSES = {
    # Standard 16th-note grid
    0.25:          "16th",
    0.5:           "8th",
    0.75:          "dotted_8th",
    1.0:           "quarter",
    1.5:           "dotted_quarter",
    2.0:           "half",
    3.0:           "dotted_half",
    4.0:           "whole",
    # 8th-note triplet grid
    1.0 / 3.0:     "triplet_8th",
    2.0 / 3.0:     "triplet_quarter",
    4.0 / 3.0:     "triplet_half",
}

# Snap tolerance: a duration must be within this many beats of a class value
# to be assigned that class; otherwise it gets the null token.
DURATION_SNAP_TOL = 0.08   # ~5% of a quarter note

NULL_DURATION = "null"

# Build duration label → index mapping (null token gets the last index)
DURATION_LABELS = sorted(set(DURATION_CLASSES.values())) + [NULL_DURATION]
DURATION_VOCAB  = {label: idx for idx, label in enumerate(DURATION_LABELS)}

# Reverse mapping
IDX_TO_DURATION = {idx: label for label, idx in DURATION_VOCAB.items()}

NULL_DURATION_IDX = DURATION_VOCAB[NULL_DURATION]


def quantize_duration(d):
    """
    Snap d (in quarter-note beats) to the nearest entry in DURATION_CLASSES.
    Returns the label string, or NULL_DURATION if nothing is close enough.
    """
    closest_val   = min(DURATION_CLASSES.keys(), key=lambda x: abs(x - d))
    if abs(closest_val - d) <= DURATION_SNAP_TOL:
        return DURATION_CLASSES[closest_val]
    return NULL_DURATION


# -----------------------------
# Subdivision grid
#
# We use 12 subdivisions per beat — the least common multiple of 4 (16th
# grid) and 3 (triplet grid).  This lets both grids snap to exact integer
# positions:
#   16th positions:            0, 3, 6, 9      (step = 12/4 = 3)
#   8th-triplet positions:     0, 4, 8         (step = 12/3 = 4)
#
# A note that doesn't snap within tolerance gets subdivision index NULL_SUBDIV.
# -----------------------------
SUBDIVS_PER_BEAT = 12
SUBDIV_SNAP_TOL  = 0.5   # in 12ths-of-a-beat units (half a grid step)

NULL_SUBDIV = SUBDIVS_PER_BEAT   # index 12 → "doesn't fit either grid"
NUM_SUBDIV_CLASSES = SUBDIVS_PER_BEAT + 1   # 0-11 valid + 12 null


def beat_to_bar_position(beat_float, beats_per_bar=4):
    """
    Returns (beat_index_in_bar, subdivision_index) where subdivision is
    quantised to a 12-subdivision-per-beat grid covering both 16th-note and
    8th-note-triplet positions.  Out-of-grid onsets get subdivision=NULL_SUBDIV.
    """
    beat_index = int(beat_float) % beats_per_bar
    frac       = beat_float - int(beat_float)          # 0.0 – <1.0

    # position on the 12-step grid (real-valued)
    grid_pos      = frac * SUBDIVS_PER_BEAT
    nearest_step  = round(grid_pos)

    if abs(grid_pos - nearest_step) <= SUBDIV_SNAP_TOL:
        subdiv = int(nearest_step) % SUBDIVS_PER_BEAT  # wrap 12 → 0
    else:
        subdiv = NULL_SUBDIV

    return beat_index, subdiv


# -----------------------------
# Build list of 4/4 time ranges from the perf_time_signatures dict.
#
# perf_time_signatures is keyed by annotation time in seconds (as a string)
# → [ts_string, beats_per_bar].  Each entry marks the start of a new time
# signature; it applies until the next entry (or end of piece).
# Returns a list of (start_sec, end_sec_or_None) covering only 4/4 spans.
# -----------------------------
def get_44_time_ranges(time_sigs):
    """
    Returns a list of (start_sec, end_sec_or_None) for every 4/4 section.
    end_sec=None means open-ended (until the last note of the performance).
    """
    sorted_changes = sorted(
        ((float(t_str), ts_str) for t_str, (ts_str, _) in time_sigs.items()),
        key=lambda x: x[0]
    )

    ranges_44 = []
    for i, (start_sec, ts_str) in enumerate(sorted_changes):
        if ts_str != "4/4":
            continue
        end_sec = sorted_changes[i + 1][0] if i + 1 < len(sorted_changes) else None
        ranges_44.append((start_sec, end_sec))

    return ranges_44


# -----------------------------
# Sequence start token
#
# Prepended to every sequence so the model (and the rules baseline) know the
# absolute bar position of the first note without having to guess from IOIs.
#
# Feature layout (7 stored values; the 8th — is_start_token — is injected
# by the Dataset class in place of the cumulative-IOI feature it normally
# appends, since cum_ioi is always 0.0 for this token and the model can use
# the non-zero value as a sentinel):
#   0  onset_ioi_beats          → 0.0  (meaningless for this token)
#   1  duration_beats           → 0.0
#   2  velocity                 → 0.0
#   3  velocity_delta_from_prev → 0.0
#   4  pc_sin                   → sin encoding of start beat index (0-3)
#   5  pc_cos                   → cos encoding of start beat index (0-3)
#   6  octave_norm              → start subdivision / 11.0  (0.0-1.0)
# The 8th column (cum_ioi % 4) will naturally be 0.0 for this token too,
# since its onset_ioi_beats is 0.0.  The model distinguishes it because
# all note-level features are zero and pc_sin/pc_cos encode the bar anchor.
#
# The corresponding target row is all -100 (masked on every head).
# -----------------------------

def build_start_token(first_note_beat):
    """
    Build the sequence-start input dict and its all-masked target dict.

    Parameters
    ----------
    first_note_beat : float
        Score-side onset_beat of the first note in the section.

    Returns
    -------
    start_input  : dict  — 7 named features matching normal note dicts
    start_target : dict  — all values masked (-100 / NULL_DURATION)
    """
    beat_index, subdiv = beat_to_bar_position(first_note_beat)

    # Encode beat index (0-3) on a unit circle, same style as pitch class
    start_input = {
        "onset_ioi_beats":          0.0,
        "duration_beats":           0.0,
        "velocity":                 0.0,
        "velocity_delta_from_prev": 0.0,
        "pc_sin":    np.sin(2 * np.pi * beat_index / 4),
        "pc_cos":    np.cos(2 * np.pi * beat_index / 4),
        # Subdivision normalised to [0, 1]; NULL_SUBDIV (12) maps to ~1.09,
        # distinguishable from valid range, but first notes are always on-grid.
        "octave_norm": subdiv / 11.0,
    }

    # All three prediction heads must ignore this token
    start_target = {
        "beat_index_in_bar": -100,
        "subdivision_index": -100,        # already the sentinel value
        "duration_class":    NULL_DURATION,  # converted to -100 in Dataset
    }

    return start_input, start_target


# -----------------------------
# Core parser for one performance.
# Accepts a list of (start_sec, end_sec) time ranges to restrict extraction.
# Returns one (inputs, targets) pair per contiguous 4/4 section.
# -----------------------------
def parse_performance(piece_root, perf_mid, score, alignment_file, annotations,
                      time_ranges_44=None, isQuantized=False):
    """
    Parameters
    ----------
    time_ranges_44 : list of (start_sec, end_sec_or_None) or None
        Performance-time intervals (seconds) to include.
        end_sec=None means until the last note of the performance.
        If None, all notes are included (original behaviour).

    Returns
    -------
    sections : list of (inputs, targets)
        One entry per contiguous 4/4 section.  Each inputs/targets is a
        list of dicts as before.  If time_ranges_44 is None the list has
        one entry covering the whole performance.

    Notes
    -----
    Notes whose score duration doesn't snap to any known duration class are
    included with duration_class=NULL_DURATION (model should mask/ignore them
    for the duration head but they still contribute sequence context).
    Similarly, onsets that don't snap to the 12-step subdivision grid receive
    subdivision_index=NULL_SUBDIV.
    """

    # Load performance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        performance = partitura.load_performance_midi(perf_mid)

    # Find beat map entry for this performance
    key = str(perf_mid).split("asap-dataset-main")[-1].lstrip("/")
    if key not in annotations:
        print(f"    WARNING: No beat map found for {key}, skipping performance")
        return []

    # Load alignment TSV
    align = {}
    with open(alignment_file) as f:
        next(f)
        for line in f:
            if len(line.strip().split("\t")) < 3:
                continue
            xml_id, midi_id, *_, onset = line.strip().split("\t")
            base_xml_id = xml_id.split("-")[0]   # removes -1, -2, etc.
            align[str(midi_id)] = base_xml_id

    # Build score lookup
    score_na    = score.note_array()
    score_notes = {n["id"]: n for n in score_na}

    notes = performance.note_array()

    # Resolve open-ended ranges using the last note's onset time
    max_onset_sec = max(float(n["onset_sec"]) for n in notes) + 1.0
    if time_ranges_44 is None:
        resolved_ranges = [(0.0, max_onset_sec)]
    else:
        resolved_ranges = [
            (s, max_onset_sec if e is None else e)
            for s, e in time_ranges_44
        ]

    # One result bucket per range; each tracks its own prev state
    sections = {
        i: {
            "inputs": [],
            "targets": [],
            "prev_velocity": None,
            "prev_onset_beat": None,
            "first_valid_note": False,
            "get_beats": None,
        }
        for i in range(len(resolved_ranges))
    }

    for n in notes:
        midi_id = n["id"]
        if midi_id not in align:
            continue
        xml_id = align[midi_id]
        if xml_id not in score_notes:
            continue

        onset_sec = float(n["onset_sec"])

        # Determine which section this note belongs to (at most one)
        sec_idx = None
        for i, (sec_start, sec_end) in enumerate(resolved_ranges):
            if sec_start <= onset_sec < sec_end:
                sec_idx = i
                break
        if sec_idx is None:
            continue

        sec        = sections[sec_idx]
        score_note = score_notes[xml_id]
        note_beat  = float(score_note["onset_beat"])

        # Lazy-init the time→beat function on the first note of each section
        # and prepend the sequence start token anchored to that note's score beat
        if not sec["first_valid_note"]:
            sec["first_valid_note"] = True
            beat_times = annotations[key]["performance_beats"]
            sec["get_beats"] = build_time_to_beat_fn(
                beat_times, onset_sec, note_beat
            )
            start_inp, start_tgt = build_start_token(note_beat)
            sec["inputs"].append(start_inp)
            sec["targets"].append(start_tgt)

        get_beats = sec["get_beats"]

        # Performance-side features
        duration_sec  = n["duration_sec"]
        offset_sec    = onset_sec + duration_sec
        pitch         = n["pitch"]
        velocity      = n["velocity"] / 127.0

        vel_delta = (0 if sec["prev_velocity"] is None
                     else velocity - sec["prev_velocity"])
        pitch_enc = encode_pitch(pitch)

        if not isQuantized:
            onset_beat    = get_beats(onset_sec)
            offset_beat   = get_beats(offset_sec)
            duration_beat = offset_beat - onset_beat
        else:
            onset_beat    = note_beat
            duration_beat = float(score_note["duration_beat"])

        ioi = (0 if sec["prev_onset_beat"] is None
               else onset_beat - sec["prev_onset_beat"])

        # Score-side ground truth (null tokens for out-of-vocab values)
        beat_index, subdiv = beat_to_bar_position(note_beat)
        duration_class     = quantize_duration(score_note["duration_beat"])

        sec["inputs"].append({
            "onset_ioi_beats":          float(ioi),
            "duration_beats":           float(duration_beat),
            "velocity":                 float(velocity),
            "velocity_delta_from_prev": float(vel_delta),
            **pitch_enc,
        })
        sec["targets"].append({
            "beat_index_in_bar": beat_index,
            "subdivision_index": subdiv,       # 0-11, or NULL_SUBDIV (12)
            "duration_class":    duration_class,  # label string, possibly NULL_DURATION
        })

        sec["prev_velocity"]   = velocity
        sec["prev_onset_beat"] = onset_beat

    return [(sec["inputs"], sec["targets"])
            for sec in sections.values()
            if sec["inputs"]]


# -----------------------------
# Main dataset builder
# -----------------------------
def build_dataset(dataset_root, isQuantized=False):
    dataset_root = Path(dataset_root)
    annotations  = load_annotations(dataset_root / "asap_annotations.json")

    inputs_by_piece  = {}
    targets_by_piece = {}

    piece_dirs = find_piece_dirs(dataset_root)

    counter            = 0
    errors             = 0
    counter_44         = 0
    sections_extracted = 0

    for piece in piece_dirs:

        print(f"Parsing piece: {piece}")
        print(f"  {counter+1}/{len(piece_dirs)}")
        counter += 1

        try:
            xml = piece / "xml_score.musicxml"
            if not xml.exists():
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                score = partitura.load_musicxml(xml)

            midis = [m for m in piece.glob("*.mid") if "midi_score" not in m.name]
            if not midis:
                continue

            # Use the first midi to look up the annotations key for time sigs
            ann_key   = str(midis[0]).split("asap-dataset-main")[-1].lstrip("/")
            time_sigs = annotations[ann_key].get("perf_time_signatures", {})
            if not time_sigs:
                print(f"    WARNING: No time signature info for {ann_key}, skipping")
                errors += 1
                continue

            # Find all 4/4 time ranges (in seconds)
            time_ranges_44 = get_44_time_ranges(time_sigs)
            if not time_ranges_44:
                print(f"    WARNING: No 4/4 sections found for {ann_key}, skipping")
                errors += 1
                continue

            counter_44 += 1

            has_non_44 = any(ts != "4/4" for _, (ts, _) in time_sigs.items())
            if has_non_44:
                print(f"    INFO: Mixed metre — extracting {len(time_ranges_44)} "
                      f"4/4 section(s) starting at "
                      f"{[f'{r[0]:.2f}s' for r in time_ranges_44]}")

            # Parse each performance midi
            for perf in midis:
                align_dir  = piece / f"{perf.stem}_note_alignments"
                align_file = align_dir / "note_alignment.tsv"
                if not align_file.exists():
                    print(f"    WARNING: No alignment file for {perf.name}, skipping")
                    errors += 1
                    continue

                print(f"  Parsing performance: {perf.name}")

                section_results = parse_performance(
                    dataset_root,
                    perf,
                    score,
                    align_file,
                    annotations,
                    time_ranges_44=time_ranges_44,
                    isQuantized=isQuantized,
                )

                for sec_idx, (inputs, targets) in enumerate(section_results):
                    if not inputs:
                        continue
                    print(f"    Section {sec_idx}: {len(inputs)} notes")
                    sec_key = f"{piece}_{perf.stem}_sec{sec_idx}"
                    inputs_by_piece[sec_key]  = inputs
                    targets_by_piece[sec_key] = targets
                    sections_extracted += 1

        except Exception as e:
            print(f"  ERROR parsing {piece.name}: {e}")
            errors += 1
            continue

    print(f"\nFinished parsing dataset.")
    print(f"  Pieces visited:             {counter}")
    print(f"  Pieces with ≥1 4/4 section: {counter_44}")
    print(f"  Errors / skipped:           {errors}")
    print(f"  Total sections extracted:   {sections_extracted}")

    return inputs_by_piece, targets_by_piece