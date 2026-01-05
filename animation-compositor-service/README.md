# Animation Compositor Service

## Overview

The Animation Compositor Service takes animation clips and audio tracks, mixes them together, and outputs the final animation frames and audio for playback.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |
| `robot-controller` | Robot hardware control service |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run animation-compositor-service/src/main.py
```

## Architecture

- **Inputs:** Receives animation and audio clips from the Animation Database service.
- **Composition:** Mixes audio tracks together and composes animation frames from the incoming clips.
- **Outputs:** Sends the composed animation frames to the robot and plays the audio.

## Configuration

The Animation Compositor requires configuration of its audio output device to ensure proper playback. This section will help you find available devices and set the correct configuration.

### 1. List Available Audio Devices

To list all audio input and output devices along with their names, indices, channels, and supported sample rates, run:

```bash
uv sync --all-packages
uv run animation-compositor-service/scripts/list_sound_devices.py
```

Example output:

```
Available input devices:
----------------------------------------
Index:  5 | Name: 'HDA Intel PCH: ALC1220 Analog (hw:2,0)'           | Channel Count:  2 | Sample Rate: 44100 Hz
Index: 11 | Name: 'C922 Pro Stream Webcam: USB Audio (hw:3,0)'       | Channel Count:  2 | Sample Rate: 32000 Hz
Index: 14 | Name: 'default'                                          | Channel Count: 64 | Sample Rate: 44100 Hz


Available output devices:
----------------------------------------
Index:  0 | Name: 'HDA NVidia: DELL U2715H (hw:0,3)'                 | Channel Count:  2 | Sample Rate: 44100 Hz
Index:  4 | Name: 'Zone Wired: USB Audio (hw:1,0)'                   | Channel Count:  2 | Sample Rate: 44100 Hz
Index: 12 | Name: 'hdmi'                                             | Channel Count:  2 | Sample Rate: 44100 Hz
```

### 2. Update the Service Configuration

Choose your preferred output device from the list above. Use its index (e.g., `4`) **or** its name (e.g., `"Zone Wired: USB Audio (hw:1,0)"`) as the `output_device_index_or_name`. Also, note the **Sample Rate** column for the`sample_rate` configuration.

Edit the [`compose.yaml`](compose.yaml) to include these under the `audio_config` section:

```yaml
configs:
  config_animation_compositor:
    content: |
      audio_config:
        output_device_index_or_name: <your_output_device_index_or_name>  # e.g., 4 or "Zone Wired: USB Audio (hw:1,0)"
        sample_rate: <your_device_sample_rate>  # e.g., 44100
```

- If you omit `output_device_index_or_name`, the system default device will be used.
- Providing the correct `sample_rate` matching your device prevents audio playback issues.

For more details on other configurations, you can check the [Animation Compositor configuration](src/configuration.py). You can add your desired values in the [compose.yaml](compose.yaml).

> **Important:** All output audio services (**Animation Compositor**, **Animation Database**, and **Text-to-Speech**) **must use the same sample rate** for proper operation.

## Development Scripts

The following scripts are available for development and debugging:

### Prerequisites

Install all dependencies:

```bash
uv sync --all-packages
```

### List Sound Devices (`list_sound_devices.py`)

Lists all available audio input and output devices with their indices, names, channel counts, and sample rates. Used to identify the correct device for audio configuration.

```bash
uv run animation-compositor-service/scripts/list_sound_devices.py
```

### Plot Debug Positions (`plot_debug_positions.py`)

Analyzes and visualizes position data from compositor debug logs. Generates plots showing robot joint positions and velocities over time to help debug movement issues.

```bash
# Plot positions from a debug CSV file
uv run animation-compositor-service/scripts/plot_debug_positions.py debug_positions.csv
```

The script generates:
- Position plots for all joints (antennas, body, head position/rotation)
- Velocity plots showing rate of change
- A text file listing the highest velocity events for debugging sudden movements
