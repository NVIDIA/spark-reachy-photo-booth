# Camera Service

The Camera Service captures images or video frames from either a physical camera device or a virtual camera (such as the simulator's Unity camera) and provides them to other components of the robot system, such as vision, interaction, or web ui modules.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run camera-service/src/main.py
```

## Overview

The Camera Service abstracts camera hardware and network video sources, ensuring a consistent interface for acquiring real-time images regardless of whether a physical camera or a simulated camera stream is used. It allows for easy switching between hardware camera input and simulation environments.

## Architecture

- **Input Sources**
  - *Physical Camera*: devices identified by their serial number.
  - *Simulator Virtual Camera*: Network-accessible video stream provided by the Unity simulator.

- **Streaming and API**
  - The service publishes images/frames on a predefined topic or endpoint, allowing consumers (such as the web ui or vision modules) to subscribe and receive the camera feed.

## Configuration

### Setting Up the Environment

Before running the Camera Service, you need to create a `.env` file containing your environment variables. The easiest way to do this is by copying the provided example:

```bash
cd camera-service
cp .env.example .env
```

### Setting Up the Camera

#### Using a Real Camera

To use a physical camera, you need to provide its serial number in the configuration.

1. **Find your camera's serial number:**

   Plug in your camera and run:

   ```bash
   sudo udevadm info --query=all /dev/video1 | grep 'USB_SERIAL_SHORT'
   ```

   Replace `/dev/video1` with the correct device if needed.

2. **Update the serial number in [`camera-service/compose.yaml`](compose.yaml):**

   Edit the [camera compose.yaml](compose.yaml) configuration as follows:

   ```yaml
   configs:
     config_camera:
       content: |
         camera_serial_or_url: "<your_serial_number>" # e.g., "E2C32290"
   ```

   Replace `<your_serial_number>` with the value you obtained in the previous step.

