# Step 1. Clone the repo

To easily manage containers without `sudo`, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with `sudo`.

Open a new terminal and test Docker access. In the terminal, run:

```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with `sudo`.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

```bash
git clone https://github.com/NVIDIA/spark-reachy-photo-booth.git
cd spark-reachy-photo-booth
```

> [!WARNING]
> This playbook is expected to be run directly on your DGX Spark and with the included web browser.

# Step 2. Create your environment

```bash
cp .env.example .env
```

Edit `.env` and set:

- **`NVIDIA_API_KEY`**: your NVIDIA API key (must start with `nvapi-...`)
- **`HF_TOKEN`**: your Hugging Face token (must start with `hf_...`)
- **`EXTERNAL_MINIO_BASE_URL`**: leave unchanged, unless you want to (see the section "Enable QR-code sharing on your local network")

To access the FLUX.1-Kontext-dev model, sign in to your Hugging Face account, then review and accept the [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and [FLUX.1-Kontext-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx) License Agreements and Acceptable Use Policy.

The remaining values are configured with reasonable defaults for local development (MinIO). For production deployments or untrusted environments, these values should be changed and stored securely.

# Step 3. Set up Reachy

- Plug the power cable to the base of Reachy and to a power outlet.
- Plug a USB-C cable to the base of Reachy and to the DGX Spark.
- Engage the power switch at the base of Reachy. The LED next to the switch should turn red.

You can verify that the robot is detected by running:

```bash
lsusb | grep Reachy
```

You should see a device printed in the terminal similar to `Bus 003 Device 003: ID 38fb:1001 Pollen Robotics Reachy Mini Audio`.

Run the following command to make sure the Reachy speaker can reach the maximum volume.

```bash
./robot-controller-service/scripts/speaker_setup.sh
```

![Setup](images/setup.jpg)

# Step 4. Start the stack

Sign in to the nvcr.io registry:
```bash
docker login nvcr.io -u "\$oauthtoken"
```

When prompted for a password, enter your NGC personal API key.

```bash
docker compose up --build -d
```

This command pulls and builds container images, and downloads the required model artifacts. The first run can take between 30 minutes and 2 hours, depending on your internet speed. Subsequent runs usually complete in about 5 minutes.

# Step 5. Open the UI in your browser

On the DGX Spark, open Firefox (pre-installed) and browse to the **Web UI**: [http://127.0.0.1:3001](http://127.0.0.1:3001).

> [!TIP]
> The Web UI is accessible only when all containers are up and running.
> You can also check the status of all the containers with `docker compose ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"`.
> If one or more containers are failing, inspect the logs with `docker compose logs -f <container_name>`.

> [!TIP]
> You can remotely **spectate** the ongoing interaction by opening an ssh session with X11 forwarding enabled (`ssh -X <USER>@<SPARK_IP>`).
> You should be able to open Firefox from this session and connect to [http://127.0.0.1:3001](http://127.0.0.1:3001).

> [!NOTE]
> The UI has a small impact on the performance of image generation. In order to optimize the performance of the image generation step in the experience, you can install and use Chromium instead of Firefox, as well as reduce the display resolution.

# Step 6. Optional: Enable QR-code sharing on your local network

Reachy can take pictures of people and generate images based on them. The web UI displays the generated images along with a QR code for downloading them. This section explains how to set up the system so that the QR code is accessible from users' phones.

For QR codes to open on your phone, your DGX Spark and phone must be on the same local network. Ensure that your router permits device-to-device communication within the network.

### 1. Find your Spark’s local IP address

On the Spark, run the following command:

```bash
ip -f inet addr show enP7s7 | grep inet
```

Or this command if your Spark is connected through Wi-Fi

```bash
ip -f inet addr show wlP9s9 | grep inet
```

Find the IPv4 on your LAN (often something like `192.168.x.x` or `10.x.x.x`).

### 2. Ensure MinIO is reachable from your phone

- **Same network**: connect your phone to the same Wi‑Fi/LAN as the DGX Spark.
- **Firewall**: by default, DGX Spark does not block incoming requests. If you installed a firewall, allow inbound traffic to the DGX Spark on **`9010` (MinIO API)**.

### 3. Update `.env` and restart

Edit `.env` and replace:

- **`EXTERNAL_MINIO_BASE_URL=127.0.0.1:9010`** → **`EXTERNAL_MINIO_BASE_URL=<SPARK_LAN_IP>:9010`**

Then restart:

```bash
docker compose down
docker compose up --build -d
```

# Step 7. Optional: Going Further & Customizing the Application

## Guides

- [Getting Started](getting-started.md) – In-depth setup and configuration walkthrough
- [Writing Your First Service](writing-your-first-service.md) – How to create and integrate a new service

## Service Configuration

Each service has its own README with details on customization, environment variables, and service-specific configuration:

| Service | Description |
|---------|-------------|
| [agent-service](../agent-service/README.md) | LLM-powered agent workflow and decision logic |
| [animation-compositor-service](../animation-compositor-service/README.md) | Combines animation clips and audio mixing |
| [animation-database-service](../animation-database-service/README.md) | Animation library and procedural animation generation |
| [camera-service](../camera-service/README.md) | Camera capture and image acquisition |
| [interaction-manager-service](../interaction-manager-service/README.md) | Event orchestration and robot utterance management |
| [metrics-service](../metrics-service/README.md) | Metrics collection and monitoring |
| [remote-control-service](../remote-control-service/README.md) | Web-based remote control interface |
| [robot-controller-service](../robot-controller-service/README.md) | Direct robot hardware control |
| [speech-to-text-service](../speech-to-text-service/README.md) | Audio transcription (NVIDIA Riva/Parakeet) |
| [text-to-speech-service](../text-to-speech-service/README.md) | Speech synthesis |
| [tracker-service](../tracker-service/README.md) | Person detection and tracking |
| [ui-server-service](../ui-server-service/README.md) | Backend for the web UI |

For detailed guidance on customizing service configurations, extending the demo with new tools, or creating your own services, refer to the [Development](development) tab.
