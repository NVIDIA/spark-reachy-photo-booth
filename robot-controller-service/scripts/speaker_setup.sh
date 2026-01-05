#!/bin/sh
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# Check for required binaries
BINARIES="sudo alsactl amixer aplay grep sed head"
for BIN in $BINARIES; do
    if ! which $BIN >/dev/null 2>&1; then
        echo "Error: Required binary '$BIN' is not installed or not in PATH."
        exit 1
    fi
done

# List of supported speakers (newline-separated)
SPEAKER_NAMES="Reachy Mini Audio
reSpeaker XVF3800 4-Mic Array"

# Find all available speakers and collect their card numbers
FOUND_CARDS=""

# Save original IFS and set to newline only
OLD_IFS="$IFS"
IFS='
'

for SPEAKER_NAME in $SPEAKER_NAMES; do
    [ -z "$SPEAKER_NAME" ] && continue
    CARD=$(aplay -l | grep -i "$SPEAKER_NAME" | head -n1 | sed -n 's/^card \([0-9]*\):.*/\1/p')
    if [ -n "$CARD" ]; then
        echo "Found speaker: $SPEAKER_NAME (card $CARD)"
        FOUND_CARDS="$FOUND_CARDS $CARD"
    fi
done

# Restore original IFS
IFS="$OLD_IFS"

# Error if no devices found
if [ -z "$FOUND_CARDS" ]; then
    echo "Error: Could not find any supported speaker devices"
    echo "Supported devices:"
    echo "$SPEAKER_NAMES"
    exit 1
fi

# Configure each found device
SUCCESS_COUNT=0
for CARD in $FOUND_CARDS; do
    echo "Configuring card $CARD..."

    # Set PCM-1 volume to max (100%)
    amixer -c "$CARD" set PCM,1 100%
    if [ "$?" -ne 0 ]; then
        echo "Warning: Failed to set PCM,1 volume to 100% for card $CARD"
        continue
    fi

    # Save the current alsamixer settings
    sudo alsactl store "$CARD"
    if [ "$?" -ne 0 ]; then
        echo "Warning: Failed to save alsamixer settings for card $CARD"
        continue
    fi

    echo "Successfully configured card $CARD"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
done

if [ "$SUCCESS_COUNT" -eq 0 ]; then
    echo "Error: Failed to configure any devices"
    exit 1
fi

echo "Speaker setup complete! Configured $SUCCESS_COUNT device(s)."
