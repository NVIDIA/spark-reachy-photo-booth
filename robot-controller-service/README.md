# Robot Controller Service

## Overview

The Robot Controller Service controls the Reachy Mini robot's hardware, managing motors, sensors (including the ReSpeaker microphone array), and physical robot movements. It bridges the physical robot with the software stack.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |
| `agent` | LLM-powered agent service |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run robot-controller-service/src/main.py
```

## Configuration

### Using the Robot with Real Hardware

This section guides you through running the `robot-controller` service with an actual Reachy Mini robot, not just the simulator.

#### 1. Enable Device Access in Docker

To let the service communicate with the real robot and its connected sensors (like the ReSpeaker microphone), edit [compose.yaml](./compose.yaml):

- **Uncomment** the `devices` section for the `robot-controller` service:

  ```yaml
    devices:
      - /dev/ttyACM0:/dev/ttyACM0         # Serial: connection to Reachy Mini
      - /dev/bus/usb:/dev/bus/usb:rw      # USB hardware access (e.g., ReSpeaker mic)
  ```

> If you are **only using the simulator**, **comment out** these lines so Docker doesn’t try to map real hardware.


#### 2. Configure Service to Use Real Robot (Not Simulator)

By default, the service uses the real robot hardware. To confirm this, check or update the configuration in your [compose.yaml](./compose.yaml):

```yaml
configs:
  config_robot_controller:
    content: |
      reachy_config:
        use_sim: False  # ensure this line is present or leave it out (the default is False)
        # ...other reachy_config options...
```

- To use the robot **simulator** instead, set `use_sim: True`.


#### 3. Set the Robot’s Speaker Volume

For clear, audible speech output from your robot, maximize the volume on the ReSpeaker microphone array:

**a. Install audio utilities** (on your robot's computer):

```bash
sudo apt-get update
sudo apt-get install -y alsa-utils
```

**b. Run the provided setup script:**

```bash
./robot-controller-service/scripts/speaker_setup.sh
```

This [speaker_setup.sh](./scripts/speaker_setup.sh) script will:

- Automatically detect your ReSpeaker XVF3800 4-Mic Array (if present)
- Set PCM (speaker output) volume to 100% for optimal clarity


#### 4. USB Device Permissions for ReSpeaker Microphones

To allow Docker to access the ReSpeaker recording device (for Direction-of-Arrival and recording), you need to set up udev rules on your host system.

1. **Create the udev rules file:**

   ```bash
   sudo tee /etc/udev/rules.d/99-reachy-mini.rules << 'EOF'
   SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="55d3", MODE="0666", GROUP="dialout" #Reachy Mini
   SUBSYSTEM=="tty", ATTRS{idVendor}=="38fb", ATTRS{idProduct}=="1001", MODE="0666", GROUP="dialout" #Reachy Mini soundcard
   SUBSYSTEM=="usb", ATTR{idVendor}=="38fb", ATTR{idProduct}=="1001", GROUP="plugdev", MODE="0660"
   EOF
   ```

2. **Reload udev rules:**

   ```bash
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

3. **Add your user to the required groups:**

   ```bash
   sudo usermod -aG dialout,plugdev $USER
   ```

4. **Reconnect the ReSpeaker device:**

   You might need to unplug and re-plug the ReSpeaker USB (or reboot your machine) to ensure the new permissions take effect.


### Running the Robot Simulator

To use the robot simulator with graphical output, follow these steps:

1. **Allow Docker containers to access your X server**
   On your host machine, run:
   ```bash
   xhost +local:docker
   ```
   This lets Docker containers connect to your running graphical session.

2. **Enable X11 socket sharing in your Docker configuration**
   In your [compose.yaml](./compose.yaml), make sure the volume for `/tmp/.X11-unix` is enabled (uncommented), like so:
   ```yaml
   volumes:
     - /tmp/.X11-unix:/tmp/.X11-unix
   ```
   This allows graphical applications inside the simulator container to display on your host.

    > **Tip:**
    > If you encounter permissions issues or graphical apps don't show up, ensure your user has access to X and you ran the `xhost` command above in your current session.

3. **Running Without Audio (No Media Mode)**

   If you don't have the Reachy Mini robot connected to your machine, or you don't want to use the robot's audio capabilities, you can disable the media backend by updating this setting in your [compose.yaml](./compose.yaml):

   ```yaml
   configs:
     config_robot_controller:
       content: |
         reachy_config:
           media_backend: "no_media"
   ```

### Troubleshooting: Reducing Vibration in the Base and Antenna

**Problem:**
If you notice unwanted vibrations in the robot's base or antenna, you may need to tune the PID (Proportional-Integral-Derivative) controller values for the affected motors.

#### How to Adjust PID Values:

Motor PID values are configured in the [`data/hardware_config.yaml`](./data/hardware_config.yaml) file. Each motor entry in this configuration has a `pid` section that specifies its PID gains. For example:

```yaml
motors:
  - body_rotation:
      id: 10
      ...
      pid:
        - 300   # P (Proportional)
        - 0     # I (Integral)
        - 250   # D (Derivative)
  - left_antenna:
      id: 18
      ...
      pid:
        - 250   # P
        - 0     # I
        - 400   # D
```

To adjust a motor's PID values:

1. **Stop the robot-controller container** (if running).
2. **Edit `data/hardware_config.yaml`** and update the `pid:` values for the specific motor you wish to tune.
3. **Restart the robot-controller container** to apply your changes.

##### Tips for PID tuning

- **P (Proportional):** Increasing P can improve accuracy but may cause overshoot or vibration.
- **I (Integral):** Rarely used; can help with steady-state error but may cause instability.
- **D (Derivative):** Increasing D can make motion smoother and reduce vibration, but too much may cause sluggish response.
- Change only one value at a time and record your original settings so you can revert if needed.

Refer to [`data/hardware_config.yaml`](./data/hardware_config.yaml) for more examples and descriptions of the format.

## Development Scripts

The following scripts are available for development and debugging:

### Prerequisites

Install all dependencies:

```bash
uv sync --all-packages
```

### Speaker Setup (`speaker_setup.sh`)

Configures the ReSpeaker microphone array's speaker output for optimal volume. Automatically detects the device and sets PCM volume to 100%.

```bash
./robot-controller-service/scripts/speaker_setup.sh
```

### Extract Positions (`extract_positions.py`)

Extracts robot position data from controller log CSV files. Parses both "Message position" and "Final position" entries into separate CSV files for analysis.

```bash
# Extract positions from a log file
uv run robot-controller-service/scripts/extract_positions.py logs_data.csv

# Specify custom output file names
uv run robot-controller-service/scripts/extract_positions.py logs_data.csv message_pos.csv final_pos.csv
```

### Play Positions (`play_positions.py`)

Plays back recorded position data on the Reachy Mini robot. Reads position data from a CSV file (generated by `extract_positions.py`) and replays them at 30 FPS.

```bash
# Play positions from a CSV file
uv run robot-controller-service/scripts/play_positions.py positions.csv
```

> **Note:** Requires a connected Reachy Mini robot. The script will move to the starting position, play back all recorded positions, then return to the zero pose.