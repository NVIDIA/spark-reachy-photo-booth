#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to extract position data from robot controller log CSV files.
Extracts both Message and Final position entries.
"""

import csv
import json
import re
import sys
from pathlib import Path


def extract_positions(
    input_file: str, message_output: str | None = None, final_output: str | None = None
):
    """
    Extract both Message and Final position entries from the log CSV file into
    separate files.

    Args:
        input_file: Path to the input CSV log file
        message_output: Path to the message positions CSV file (defaults to
            input_file with _message_positions suffix)
        final_output: Path to the final positions CSV file (defaults to
            input_file with _final_positions suffix)
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)

    # Generate output filenames if not provided
    if message_output is None:
        message_output = str(
            input_path.parent / f"{input_path.stem}_message_positions.csv"
        )

    if final_output is None:
        final_output = str(input_path.parent / f"{input_path.stem}_final_positions.csv")

    message_positions = []
    final_positions = []

    # Read the CSV file
    with open(input_path) as f:
        reader = csv.DictReader(f)

        for row in reader:
            line = row.get("Line", "")

            # Check for both "Message position:" and "Final position:"
            position_type = None
            pattern = None
            if "Message position:" in line:
                position_type = "message"
                pattern = r"Message position: (\{[^}]*\{[^}]*\}[^}]*\})"
            elif "Final position:" in line:
                position_type = "final"
                pattern = r"Final position: (\{[^}]*\{[^}]*\}[^}]*\})"

            if position_type and pattern:
                # Extract the dictionary part using regex
                match = re.search(pattern, line)
                if match:
                    position_str = match.group(1)
                    try:
                        # Convert string representation to actual dict
                        # Replace single quotes with double quotes for JSON
                        position_str_json = position_str.replace("'", '"')
                        position_dict = json.loads(position_str_json)

                        # Add timestamp info
                        entry = {
                            "timestamp": row.get("Date", ""),
                            "time_ns": row.get("tsNs", ""),
                            "position": position_dict,
                        }

                        # Add to appropriate list based on type
                        if position_type == "message":
                            message_positions.append(entry)
                        else:  # final
                            final_positions.append(entry)

                    except (json.JSONDecodeError, Exception) as e:
                        print(f"Warning: Could not parse {position_type} position: {e}")
                        print(f"  Line: {line}")

    # Function to save positions to CSV
    def save_positions_to_csv(positions, output_file, position_type_name):
        if positions:
            with open(output_file, "w", newline="") as f:
                # Flatten the nested structure for CSV
                fieldnames = [
                    "timestamp",
                    "time_ns",
                    "body_angle",
                    "r_antenna_angle",
                    "l_antenna_angle",
                    "head_position_x",
                    "head_position_y",
                    "head_position_z",
                    "head_rotation_roll",
                    "head_rotation_pitch",
                    "head_rotation_yaw",
                ]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                for entry in positions:
                    pos = entry["position"]
                    flat_row = {
                        "timestamp": entry["timestamp"],
                        "time_ns": entry["time_ns"],
                        "body_angle": pos.get("body_angle", ""),
                        "r_antenna_angle": pos.get("r_antenna_angle", ""),
                        "l_antenna_angle": pos.get("l_antenna_angle", ""),
                        "head_position_x": pos.get("head_position_x", ""),
                        "head_position_y": pos.get("head_position_y", ""),
                        "head_position_z": pos.get("head_position_z", ""),
                        "head_rotation_roll": pos.get("head_rotation", {}).get(
                            "roll", ""
                        ),
                        "head_rotation_pitch": pos.get("head_rotation", {}).get(
                            "pitch", ""
                        ),
                        "head_rotation_yaw": pos.get("head_rotation", {}).get(
                            "yaw", ""
                        ),
                    }
                    writer.writerow(flat_row)

            print(f"Extracted {len(positions)} {position_type_name} positions")
            print(f"  Output saved to: {output_file}")
        else:
            print(f"No {position_type_name} positions found in the input file")

    # Save message positions
    save_positions_to_csv(message_positions, message_output, "message")

    # Save final positions
    save_positions_to_csv(final_positions, final_output, "final")

    # Print summary
    total = len(message_positions) + len(final_positions)
    print(f"\nTotal positions extracted: {total}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python extract_positions.py <input_csv_file> "
            "[message_output_file] [final_output_file]"
        )
        print("\nExample:")
        print("  python extract_positions.py 'logs_data.csv'")
        print(
            "  python extract_positions.py 'logs_data.csv' "
            "'message_pos.csv' 'final_pos.csv'"
        )
        sys.exit(1)

    input_file = sys.argv[1]
    message_output = sys.argv[2] if len(sys.argv) > 2 else None
    final_output = sys.argv[3] if len(sys.argv) > 3 else None

    extract_positions(input_file, message_output, final_output)
