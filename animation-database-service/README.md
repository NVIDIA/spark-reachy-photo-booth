# Animation Database Service

A service that provides a library of robot animations and procedural audio generation for the Reachy robot platform.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run animation-database-service/src/main.py
```

## Architecture

The Animation Database service is responsible for providing animation data and generating procedural animations and audio for playback in the robot platform. It serves as a backend library, supporting multiple types of requests:

- **Animation Library:** Serves a library of pre-made animation clips (created in Blender) for various robot expressions, movements, and behaviors. These are stored in the [Animation Library](assets/animLibrary/) directory and are ready to be composed or played back by other services (like the Animation Compositor).
- **Procedural Animation Generation:** Synthesizes animations procedurally on the fly, such as tracking, or looking at a specific position.
- **Procedural Audio Generation:** Generates audio tracks on the fly, such as motor sounds or other effects, using a fully parametric audio generation system.
- **Interop with Other Services:** Designed to be used together with the Animation Compositor service, ensuring playback of animation and audio. It shares compatible audio configuration parameters to guarantee smooth integration across the stack.

The typical flow is:
1. A client (such as the Interaction Manager) requests an animation clip and/or procedural audio with specified parameters.
2. The Animation Database loads and returns the requested data, applying procedural generation if needed.
3. The Animation Compositor handles the timing/mixing and final output for the robot.


## Documentation

Explore more details about the Animation Database service and its components:

- **[Assets Documentation](assets/README.md):** Guide to the structure, creation, and usage of the Blender animations library included with the service.
- **[Audio API Documentation](src/audio/README.md):** Reference for the audio generation subsystem, including available endpoints and usage examples for generating procedural sounds.

## Configuration

The Animation Database service requires very minimal configuration for most users. By default, it will look for animation assets in the [Animation Library](assets/animLibrary/) and use the default audio parameters designed to match the Animation Compositor service.

You can see the available configuration options in the [Animation Database configuration](src/configuration.py). You can add your desired values in the [compose.yaml](compose.yaml).

> **Important:** All output audio services (**Animation Compositor**, **Animation Database**, and **Text-to-Speech**) **must use the same sample rate** for proper operation.

## Development Scripts

The following scripts are available for development and testing:

### Prerequisites

Install all dependencies:

```bash
uv sync --all-packages
```

### Test Clips (`test_clips.py`)

Interactive tool for triggering pre-made and procedural animations via Kafka. Useful for testing animation playback without running the full stack.

**Basic animation commands:**

```bash
# Play a single animation clip by name
uv run animation-database-service/scripts/test_clips.py play <clip_name>

# Play multiple animation clips sequentially
uv run animation-database-service/scripts/test_clips.py play_multiple <clip_name_1> <clip_name_2> ...

# Stop a currently playing animation
uv run animation-database-service/scripts/test_clips.py stop

# Change the audio volume for an animation
uv run animation-database-service/scripts/test_clips.py volume <volume_value>
```

**Procedural animation commands:**

```bash
# Make the robot look at a target position (x, y coordinates)
uv run animation-database-service/scripts/test_clips.py look_at <x> <y>

# Start procedural tracking animation
uv run animation-database-service/scripts/test_clips.py track --state start

# Stop procedural tracking animation
uv run animation-database-service/scripts/test_clips.py track --state stop
```

> **Tip:** All commands accept additional options (such as `--robot-id`, `--volume`, `--fade-in`, etc.). Run with `--help` for details:

```bash
uv run animation-database-service/scripts/test_clips.py play --help
```

### Test Procedural Audio (`test_procedural_audio.py`)

Generates and exports test audio files for each motor sound type. Useful for previewing and debugging the procedural audio generation system.

```bash
uv run animation-database-service/scripts/test_procedural_audio.py
```

The script generates WAV files for body angle, head yaw/pitch/roll, and antenna sounds in a `scripts/audio_tests/` directory, then plays the last generated file and creates a visualization.
