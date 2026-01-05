# Tracker Service

## Overview

The Tracker Service provides real-time person detection and tracking capabilities for robotic systems. It processes camera frames to detect humans, estimate their pose, and track individuals across frames—enabling the robot to understand who is present and where they are located.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run tracker-service/src/main.py
```

- **Person Detection**: Identifies humans in camera frames using deep learning models (Detectron2)
- **Multi-Person Tracking**: Assigns persistent IDs to detected individuals using ByteTrack, maintaining identity even when people briefly leave the frame
- **Pose Estimation**: Extracts 17 body keypoints per person following the COCO convention, enabling skeleton-based analysis of body position and movement
- **Presence State Management**: Maintains a state machine tracking user presence (appeared/disappeared) with configurable proximity thresholds and patience timers
- **Detection Smoothing**: Applies configurable filters (EMA, Butterworth, Kalman, Adaptive) to reduce jitter in bounding box and keypoint outputs

## Prerequisites

```bash
uv sync --all-packages
```

Install NVIDIA container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Keypoints Convention

We use the [COCO keypoints convention](https://docs.ultralytics.com/datasets/pose/coco/#coco-pose-pretrained-models) which contains 17 keypoints.

The keypoints are shown in the following image:


<img src="https://user-images.githubusercontent.com/100993824/227864552-489d03de-e1b8-4ca2-8ac1-80dd99826cb7.png" width="200">

## Development Scripts

The following scripts are available for development and testing:

### Prerequisites

Install all dependencies:

```bash
uv sync --all-packages
```

### Trigger Tracker (`trigger_tracker.py`)

Enables or disables the Tracker service via Kafka. Useful for testing the system with or without person detection.

```bash
# Enable the tracker
uv run tracker-service/scripts/trigger_tracker.py --status "on"

# Disable the tracker
uv run tracker-service/scripts/trigger_tracker.py --status "off"
```

### Trigger User State (`trigger_user_state.py`)

Simulates user presence events via Kafka. Useful for testing how other services respond to user appearing or disappearing.

```bash
# Simulate a user appearing
uv run tracker-service/scripts/trigger_user_state.py --status "appeared"

# Simulate a user disappearing
uv run tracker-service/scripts/trigger_user_state.py --status "disappeared"

# Specify a robot ID
uv run tracker-service/scripts/trigger_user_state.py --status "appeared" --robot-id "researcher"
```
