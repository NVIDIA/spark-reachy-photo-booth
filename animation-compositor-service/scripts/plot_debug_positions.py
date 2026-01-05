#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script to plot debug position data from the animation compositor.
This helps visualize and debug fast movements or position anomalies.

Usage:
    python plot_debug_positions.py [csv_file]

If no csv_file is provided, it will use 'debug_positions.csv' by default.
"""

import ast
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Visualization thresholds
MIN_VELOCITY_FOR_HIGHLIGHTING = 1.0  # unit/s - minimum velocity to enable highlighting
MIN_POSITION_JUMP_THRESHOLD = 5.0  # mm - minimum position change to highlight
MIN_ANGLE_JUMP_THRESHOLD = 5.0  # degrees - minimum angle change to highlight
VELOCITY_HIGHLIGHT_RATIO = 0.8  # Highlight velocities above 80% of max
JUMP_HIGHLIGHT_RATIO = 0.5  # Highlight position jumps above 50% of max delta


def load_position_data(csv_file):
    """Load position data from CSV file."""
    data = {
        "time_offset": [],
        "timestamp_ns": [],  # Store original timestamps
        "date": [],  # Store date field from CSV
        "r_antenna_angle": [],
        "l_antenna_angle": [],
        "body_angle": [],
        "head_position_x": [],
        "head_position_y": [],
        "head_position_z": [],
        "head_rotation_roll": [],
        "head_rotation_pitch": [],
        "head_rotation_yaw": [],
    }

    # First pass: collect all timestamps and frame data
    raw_data = []

    with open(csv_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract the Line column which contains the frame data
            line = row.get("Line", "")
            date = row.get("Date", "")

            # Parse the frame data from the Line column
            # Expected format: "Frame: {'body_angle': 0.0, ...}"
            if not line or not line.startswith("Frame:"):
                continue

            # Extract the dictionary part
            match = re.search(r"Frame:\s*(\{.*\})", line)
            if not match:
                continue

            frame_dict_str = match.group(1)

            # Parse the dictionary
            try:
                frame_data = ast.literal_eval(frame_dict_str)
            except (ValueError, SyntaxError):
                continue

            # Get timestamp in nanoseconds
            tsNs = int(row.get("tsNs", 0))
            raw_data.append((tsNs, date, frame_data))

    # Sort by timestamp (oldest first) since logs are in reverse chronological order
    raw_data.sort(key=lambda x: x[0])

    # Find the minimum timestamp to use as start time
    if not raw_data:
        return data

    start_time_ns = raw_data[0][0]

    # Second pass: process sorted data
    for tsNs, date, frame_data in raw_data:
        time_offset_sec = (tsNs - start_time_ns) / 1e9
        data["time_offset"].append(time_offset_sec)
        data["timestamp_ns"].append(tsNs)  # Store original timestamp
        data["date"].append(date)  # Store date field

        # Parse each field, handling missing values
        data["r_antenna_angle"].append(frame_data.get("r_antenna_angle"))
        data["l_antenna_angle"].append(frame_data.get("l_antenna_angle"))
        data["body_angle"].append(frame_data.get("body_angle"))
        data["head_position_x"].append(frame_data.get("head_position_x"))
        data["head_position_y"].append(frame_data.get("head_position_y"))
        data["head_position_z"].append(frame_data.get("head_position_z"))

        # Parse head_rotation if it exists (it's a nested dict)
        head_rotation = frame_data.get("head_rotation", {})
        if isinstance(head_rotation, dict):
            data["head_rotation_roll"].append(head_rotation.get("roll"))
            data["head_rotation_pitch"].append(head_rotation.get("pitch"))
            data["head_rotation_yaw"].append(head_rotation.get("yaw"))
        else:
            data["head_rotation_roll"].append(None)
            data["head_rotation_pitch"].append(None)
            data["head_rotation_yaw"].append(None)

    return data


def plot_positions(data, csv_file):
    """Create plots for all position data."""
    time = data["time_offset"]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Robot Position Debug Data - {Path(csv_file).name}", fontsize=16)

    # Plot each joint/position
    plots = [
        (0, 0, "r_antenna_angle", "Right Antenna Angle", "degrees"),
        (0, 1, "l_antenna_angle", "Left Antenna Angle", "degrees"),
        (0, 2, "body_angle", "Body Angle", "degrees"),
        (1, 0, "head_position_x", "Head Position X", "mm"),
        (1, 1, "head_position_y", "Head Position Y", "mm"),
        (1, 2, "head_position_z", "Head Position Z", "mm"),
        (2, 0, "head_rotation_roll", "Head Rotation Roll", "degrees"),
        (2, 1, "head_rotation_pitch", "Head Rotation Pitch", "degrees"),
        (2, 2, "head_rotation_yaw", "Head Rotation Yaw", "degrees"),
    ]

    for row, col, field, title, unit in plots:
        ax = axes[row, col]
        values = data[field]

        # Filter out None values
        valid_data = [
            (t, v) for t, v in zip(time, values, strict=False) if v is not None
        ]
        if valid_data:
            valid_time, valid_values = zip(*valid_data, strict=False)
            ax.plot(valid_time, valid_values, "b-", linewidth=0.5)
            ax.scatter(valid_time, valid_values, c="r", s=1, alpha=0.5)
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel(f"{title} ({unit})")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Highlight potential sudden changes
            if len(valid_values) > 1:
                deltas = [
                    abs(valid_values[i + 1] - valid_values[i])
                    for i in range(len(valid_values) - 1)
                ]
                if deltas:
                    threshold = max(deltas) * JUMP_HIGHLIGHT_RATIO
                    # Use appropriate minimum jump threshold based on unit type
                    min_jump = (
                        MIN_ANGLE_JUMP_THRESHOLD
                        if unit == "degrees"
                        else MIN_POSITION_JUMP_THRESHOLD
                    )
                    for i, delta in enumerate(deltas):
                        if delta > threshold and delta > min_jump:
                            ax.axvline(
                                valid_time[i],
                                color="orange",
                                alpha=0.3,
                                linestyle="--",
                                linewidth=1,
                            )
        else:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title(f"{title} (No Data)")

    plt.tight_layout()

    # Save figure
    output_file = Path(csv_file).with_suffix(".png")
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")


def plot_velocities(data, csv_file):
    """Create plots showing velocities (rate of change) for each joint."""
    time = data["time_offset"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(
        f"Robot Velocity Debug Data (Rate of Change) - {Path(csv_file).name}",
        fontsize=16,
    )

    plots = [
        (0, 0, "r_antenna_angle", "Right Antenna Velocity", "deg/s"),
        (0, 1, "l_antenna_angle", "Left Antenna Velocity", "deg/s"),
        (0, 2, "body_angle", "Body Angular Velocity", "deg/s"),
        (1, 0, "head_position_x", "Head Velocity X", "mm/s"),
        (1, 1, "head_position_y", "Head Velocity Y", "mm/s"),
        (1, 2, "head_position_z", "Head Velocity Z", "mm/s"),
        (2, 0, "head_rotation_roll", "Head Roll Velocity", "deg/s"),
        (2, 1, "head_rotation_pitch", "Head Pitch Velocity", "deg/s"),
        (2, 2, "head_rotation_yaw", "Head Yaw Velocity", "deg/s"),
    ]

    for row, col, field, title, unit in plots:
        ax = axes[row, col]
        values = data[field]

        # Filter out None values and calculate velocities
        valid_data = [
            (t, v) for t, v in zip(time, values, strict=False) if v is not None
        ]
        if len(valid_data) > 1:
            valid_time, valid_values = zip(*valid_data, strict=False)
            velocities = []
            velocity_times = []

            for i in range(len(valid_values) - 1):
                dt = valid_time[i + 1] - valid_time[i]
                if dt > 0:
                    dv = valid_values[i + 1] - valid_values[i]
                    velocity = dv / dt
                    velocities.append(velocity)
                    velocity_times.append(valid_time[i])

            if velocities:
                ax.plot(velocity_times, velocities, "b-", linewidth=0.5)
                ax.scatter(velocity_times, velocities, c="r", s=1, alpha=0.5)
                ax.set_xlabel("Time (seconds)")
                ax.set_ylabel(f"Velocity ({unit})")
                ax.grid(True, alpha=0.3)

                # Highlight extreme velocities
                abs_velocities = [abs(v) for v in velocities]
                max_vel = max(abs_velocities)
                avg_vel = sum(abs_velocities) / len(abs_velocities)

                # Add stats to title
                ax.set_title(f"{title}\nMax: {max_vel:.2f}, Avg: {avg_vel:.2f} {unit}")

                # Only highlight if there's significant variation
                if max_vel > MIN_VELOCITY_FOR_HIGHLIGHTING:
                    threshold = max_vel * VELOCITY_HIGHLIGHT_RATIO
                    extreme_times = [
                        velocity_times[i]
                        for i, v in enumerate(abs_velocities)
                        if v > threshold
                    ]
                    for t in extreme_times:
                        ax.axvline(
                            t, color="red", alpha=0.2, linestyle="--", linewidth=1
                        )
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{title} (No Data)")

    plt.tight_layout()

    # Save figure
    output_file = (
        Path(csv_file)
        .with_stem(Path(csv_file).stem + "_velocities")
        .with_suffix(".png")
    )
    plt.savefig(output_file, dpi=150)
    print(f"Velocity plot saved to: {output_file}")


def save_highest_velocities(data, csv_file, top_n=10):
    """Save timestamps and values for the highest velocities to a file."""
    time = data["time_offset"]
    timestamps = data["timestamp_ns"]
    dates = data["date"]

    fields = [
        ("r_antenna_angle", "Right Antenna", "deg/s"),
        ("l_antenna_angle", "Left Antenna", "deg/s"),
        ("body_angle", "Body", "deg/s"),
        ("head_position_x", "Head X", "mm/s"),
        ("head_position_y", "Head Y", "mm/s"),
        ("head_position_z", "Head Z", "mm/s"),
        ("head_rotation_roll", "Head Roll", "deg/s"),
        ("head_rotation_pitch", "Head Pitch", "deg/s"),
        ("head_rotation_yaw", "Head Yaw", "deg/s"),
    ]

    # Create output file
    output_file = (
        Path(csv_file)
        .with_stem(Path(csv_file).stem + "_highest_velocities")
        .with_suffix(".txt")
    )

    with open(output_file, "w") as f:
        f.write("=" * 100 + "\n")
        f.write(f"TOP {top_n} HIGHEST VELOCITIES BY FIELD\n")
        f.write(f"Source: {Path(csv_file).name}\n")
        f.write("=" * 100 + "\n")

        for field, name, unit in fields:
            values = data[field]

            # Filter out None values and calculate velocities with indices
            valid_data = []
            for i, (t, ts, dt, v) in enumerate(
                zip(time, timestamps, dates, values, strict=False)
            ):
                if v is not None:
                    valid_data.append((i, t, ts, dt, v))

            if len(valid_data) <= 1:
                continue

            velocity_data = []

            for j in range(len(valid_data) - 1):
                i0, t0, ts0, dt0, v0 = valid_data[j]
                i1, t1, ts1, dt1, v1 = valid_data[j + 1]
                dt = t1 - t0
                if dt > 0:
                    dv = v1 - v0
                    velocity = dv / dt
                    velocity_data.append((abs(velocity), velocity, t0, ts0, dt0, v0))

            if not velocity_data:
                continue

            # Sort by absolute velocity (descending)
            velocity_data.sort(reverse=True, key=lambda x: x[0])

            f.write(f"\n{name} ({unit}):\n")
            f.write(
                f"{'Date':<30} {'Timestamp (ns)':<20} "
                f"{'Time (s)':<12} {'Velocity':<15} {'Value':<15}\n"
            )
            f.write("-" * 100 + "\n")

            for _abs_vel, vel, t, ts, date, val in velocity_data[:top_n]:
                f.write(f"{date:<30} {ts:<20} {t:>11.3f}  {vel:>14.2f}  {val:>14.2f}\n")

    print(f"Highest velocities saved to: {output_file}")
    return output_file


def main():
    # Get CSV file from command line or use default
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "debug_positions.csv"

    csv_path = Path(csv_file)
    if not csv_path.exists():
        print(f"Error: File '{csv_file}' not found!")
        print(f"Usage: {sys.argv[0]} [csv_file]")
        sys.exit(1)

    print(f"Loading data from: {csv_file}")
    data = load_position_data(csv_file)

    num_frames = len(data["time_offset"])
    if num_frames > 0:
        duration = data["time_offset"][-1] - data["time_offset"][0]
        print(f"Loaded {num_frames} frames over {duration:.2f} seconds")
        print(f"Average frame rate: {num_frames / duration:.1f} fps")
    else:
        print("No data found in file!")
        sys.exit(1)

    print("\nGenerating position plots...")
    plot_positions(data, csv_file)

    print("\nGenerating velocity plots (rate of change)...")
    plot_velocities(data, csv_file)

    print("\nSaving highest velocities to file...")
    save_highest_velocities(data, csv_file, top_n=10)

    print(
        "\nDone! Check for sudden spikes in "
        "the velocity plots to identify fast movements."
    )


if __name__ == "__main__":
    main()
