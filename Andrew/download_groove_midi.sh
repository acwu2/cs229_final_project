#!/usr/bin/env bash
# download_groove_midi.sh
# Downloads and extracts the Groove MIDI Dataset from Magenta.
#
# Usage:  bash download_groove_midi.sh [target_dir]
#         default target_dir = ./groove_midi

set -euo pipefail

TARGET="${1:-groove_midi}"
URL="https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"
AUDIO_URL="https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip"

echo "==> Downloading Groove MIDI Dataset (full, with audio)..."
echo "    This is ~4.8 GB and may take a while."

mkdir -p "$TARGET"

# Download the full dataset (with audio)
if [ ! -f "$TARGET/groove-v1.0.0.zip" ]; then
    curl -L -o "$TARGET/groove-v1.0.0.zip" "$AUDIO_URL"
else
    echo "    Archive already exists, skipping download."
fi

echo "==> Extracting..."
unzip -q -o "$TARGET/groove-v1.0.0.zip" -d "$TARGET"

# The archive extracts into groove/ — move contents up one level
if [ -d "$TARGET/groove" ]; then
    mv "$TARGET/groove"/* "$TARGET/" 2>/dev/null || true
    rmdir "$TARGET/groove" 2>/dev/null || true
fi

echo "==> Done.  Dataset at: $TARGET/"
echo "    Verify with: ls $TARGET/info.csv"
