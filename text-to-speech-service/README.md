# Text-to-Speech Service

## Overview

A text-to-speech service that converts robot utterances to audio using the KittenTTS model and publishes audio clips via Kafka.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run text-to-speech-service/src/main.py
```

## Architecture

The core workflow of the Text-to-Speech service is as follows:

- Listens for `HumanSpeechRequest` messages on the `human_speech_request_topic`. These messages contain the text that the robot should speak.
- Synthesizes high-quality speech audio from the received text using the configured TTS model (e.g., KittenTTS, Kokoro).
- Processes the generated audio as needed.
- Publishes the resulting audio data as a `ClipData` message to the `clip_data_topic` Kafka topic, making it available for playback by downstream services like the Animation Compositor.

This architecture enables seamless, real-time text-to-speech conversion as part of the robot's interactive pipeline.

## Configuration

There is not much configuration needed for the Text-to-Speech service. The most important thing is that **all output audio services** (Animation Compositor, Animation Database, and Text-to-Speech) **must use the same audio configuration** for proper operation. You can adjust the TTS model, voice, or engine by editing the corresponding fields in the [compose.yaml](compose.yaml) configuration block.

For more details on other configurations, you can check the [Text-to-Speech configuration](src/configuration.py). You can add your desired values in the [compose.yaml](compose.yaml).

## Development Scripts

The following scripts are available for development and testing:

### Prerequisites

Install all dependencies:

```bash
uv sync --all-packages
```

### Trigger TTS (`trigger_tts.py`)

Triggers speech synthesis from the command line. Sends a text message via Kafka to generate audio using the TTS model.

```bash
# Generate speech for a text string
uv run text-to-speech-service/scripts/trigger_tts.py --script "Hello world"

# Show all options
uv run text-to-speech-service/scripts/trigger_tts.py --help
```

**Running with minimal services:**

```bash
docker compose up --build --watch redpanda text-to-speech
```

**Verifying output:**
- The service generates an audio clip and publishes it to the `clip_data` Kafka topic.
- Run the Animation Compositor to capture and play the audio.

> **Note:** Ensure your Kafka broker's `max.message.bytes` is set sufficiently high (see root [README.md](../README.md)) to allow for transmission of long audio messages.

### Cache Kokoro Model and Voices (`cache_kokoro_model_and_voices.py`)

Pre-caches all Kokoro TTS model voices. Useful for warming up the model cache before deployment to reduce first-request latency.

```bash
uv run text-to-speech-service/scripts/cache_kokoro_model_and_voices.py
```

The script iterates through all available voice presets and generates a test phrase to cache each voice.
