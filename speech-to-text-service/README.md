# Speech-to-Text Service

## Overview

A speech-to-text service that captures audio from a microphone, transcribes it to text using an ASR (Automatic Speech Recognition) model, and publishes the recognized text to Kafka topics for downstream consumers.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |
| `riva` | NVIDIA Riva ASR service (Parakeet) |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run speech-to-text-service/src/main.py
```

## Architecture

- Captures audio input from a selected microphone device.
- Performs speech recognition using a supported ASR backend (currently using NVIDIA Riva ASR).
- Publishes transcribed user speech as `UserUtterance` messages to the `user_utterance_topic` in Kafka with real-time status updates:
    - `STARTED`: Indicates the beginning of a new utterance capture.
    - `PARTIAL`: Provides partial, incremental transcriptions as the user speaks.
    - `FINISHED`: Signals completion with the final, full transcription.

## Configuration

The Speech-to-Text service requires configuration of its audio input device (microphone) to ensure proper capture of audio for transcription. This section will guide you in finding available input devices and setting the configuration correctly.

### 1. List Available Input Devices

To list all audio input and output devices along with their names, indices, channels, and supported sample rates, run:

```bash
uv sync --all-packages
uv run speech-to-text-service/scripts/list_sound_devices.py
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

Choose your preferred input device from the list above. Use its index (e.g., `11`) **or** its name (e.g., `"C922 Pro Stream Webcam: USB Audio (hw:3,0)"`) as the `input_device_index_or_name`. Also, note the **Sample Rate** shown for your device for the `sample_rate` configuration.

Edit the [`compose.yaml`](compose.yaml) to include the correct values under the `audio_config` section:

```yaml
configs:
  config_speech_to_text:
    content: |
      input_device_index_or_name: <your_input_device_index_or_name>  # e.g., 11 or "C922 Pro Stream Webcam: USB Audio (hw:3,0)"
      audio_config:
        sample_rate: <your_device_sample_rate>  # e.g., 16000, 22050, 44100, or 48000
```

- If you omit `input_device_index_or_name`, the system default input device will be used.
- Setting the correct `sample_rate` to match your device helps prevent audio input errors.

For more details on other configurations, you can check the [Speech-to-Text configuration](src/configuration.py). You can add your desired values in the [compose.yaml](compose.yaml).

> **Tip:** You can specify part of an input device's name instead of typing its full name. If an exact match is not found, the system will search for and select the first device whose name contains your string (case-insensitive). If still nothing matches, the default input device will be used automatically. For example, specifying `reSpeaker` will automatically pick the first device containing that word, such as `reSpeaker XVF3800 4-Mic Array: USB Audio (hw:1,0)`.

## Development Scripts

The following scripts are available for development and testing:

### Prerequisites

Install all dependencies:

```bash
uv sync --all-packages
```

### List Sound Devices (`list_sound_devices.py`)

Lists all available audio input and output devices with their indices, names, channel counts, and sample rates. Used to identify the correct microphone device for configuration.

```bash
uv run speech-to-text-service/scripts/list_sound_devices.py
```

### Trigger STT (`trigger_stt.py`)

Enables or disables the Speech-to-Text service via Kafka. Useful for testing the system with or without speech input.

```bash
# Enable speech-to-text
uv run speech-to-text-service/scripts/trigger_stt.py --status "on"

# Disable speech-to-text
uv run speech-to-text-service/scripts/trigger_stt.py --status "off"

# Show all options
uv run speech-to-text-service/scripts/trigger_stt.py --help
```

**Running with minimal services:**

```bash
docker compose up --build --watch redpanda speech-to-text
```
