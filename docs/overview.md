# Basic idea

![Teaser](images/teaser.jpg)

Spark & Reachy Photo Booth is an interactive and event-driven photo booth demo that combines the **DGX Spark™** with the **Reachy Mini** robot to create an engaging multimodal AI experience. The system showcases:

- **A multi-modal agent** built with the `NeMo Agent Toolkit`
- **A ReAct loop** driven by the `openai/gpt-oss-20b` LLM powered by `TensorRT-LLM`
- **Voice interaction** based on `nvidia/riva-parakeet-ctc-1.1B` and `hexgrad/Kokoro-82M`
- **Image generation** with `black-forest-labs/FLUX.1-Kontext-dev` for image-to-image restyling
- **User position tracking** built with `facebookresearch/detectron2` and `FoundationVision/ByteTrack`
- **MinIO** for storing captured/generated images as well as sharing them via QR-code

The demo is based on a several services that communicate through a message bus.

![Architecture diagram](images/architecture-diagram.png)

> [!NOTE]
> This playbook applies to both the Reachy Mini and Reachy Mini Lite robots. For simplicity, we’ll refer to the robot as Reachy throughout this playbook.

# What you'll accomplish

You'll deploy a complete photo booth system on DGX Spark running multiple inference models locally — LLM, image generation, speech recognition, speech generation, and computer vision — all without cloud dependencies. The Reachy robot interacts with users through natural conversation, captures photos, and generates custom images based on prompts, demonstrating real-time multimodal AI processing on edge hardware.

# What to know before starting

- Basic Docker and Docker Compose knowledge
- Basic network configuration skills

# Prerequisites

**Hardware Requirements:**
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- A monitor, a keyboard, and a mouse to run this playbook directly on the DGX Spark.
- [Reachy Mini or Reachy Mini Lite robot](https://pollen-robotics-reachy-mini.hf.space/)

> [!TIP]
> Make sure your Reachy robot firmware is up to date. You can find instructions to update it [here](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini).
**Software Requirements:**
- The official DGX Spark OS image including all required utilities such as Git, Docker, NVIDIA drivers, and the NVIDIA Container Toolkit
- An internet connection for the DGX Spark
- NVIDIA NGC Personal API Key (**`NVIDIA_API_KEY`**). [Create a key](https://org.ngc.nvidia.com/setup/api-keys) if necessary. Make sure to enable the `NGC Catalog` scope when creating the key.
- Hugging Face access token (**`HF_TOKEN`**). [Create a token](https://huggingface.co/settings/tokens) if necessary. Make sure to create a token with _Read access to contents of all public gated repos you can access_ permission.


# Ancillary files

All required assets can be found in this repository.

- The Docker Compose application
- Various configuration files
- Source code for all the services
- Detailed documentation

# Time & risk

* **Estimated time:** 2 hours including hardware setup, container building, and model downloads
* **Risk level:** Medium
* **Rollback:** Docker containers can be stopped and removed to free resources. Downloaded models can be deleted from cache directories. Robot and peripheral connections can be safely disconnected. Network configurations can be reverted by removing custom settings.
* **Last Updated:** 01/27/2026
  * 1.0.0 First Publication

# Governing terms
Your use of the Spark Playbook scripts is governed by [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) and enables use of separate open source and proprietary software governed by their respective licenses: [Flux.1-Kontext NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/black-forest-labs/containers/flux.1-kontext-dev?version=1.1), [Parakeet 1.1b CTC en-US ASR NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-1-1b-ctc-en-us?version=1.4), [TensorRT-LLM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc1), [minio/minio](https://hub.docker.com/r/minio/minio), [arizephoenix/phoenix](https://hub.docker.com/r/arizephoenix/phoenix), [grafana/otel-lgtm](https://hub.docker.com/r/grafana/otel-lgtm), [Python](https://hub.docker.com/_/python), [Node.js](https://hub.docker.com/_/node), [nginx](https://hub.docker.com/_/nginx), [busybox](https://hub.docker.com/_/busybox), [UV Python Packager](https://docs.astral.sh/uv/), [Redpanda](https://www.redpanda.com/), [Redpanda Console](https://www.redpanda.com/), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), [FLUX.1-Kontext-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx).

> [!NOTE]
> FLUX.1-Kontext-dev and FLUX.1-Kontext-dev-onnx are models released for non-commercial use. Contact sales@blackforestlabs.ai for commercial terms. You are responsible for accepting the applicable License Agreements and Acceptable Use Policies, and for ensuring your HF token has the correct permissions.
