# Spark & Reachy Mini Playbook

* [Overview](docs/overview.md)
* [Instructions](docs/instructions.md)
* [Development](docs/development.md)
* [Troubleshooting](docs/troubleshooting.md)

## Third-Party Dependencies

This application allows you to deploy various third-party Python, JavaScript, model, and container dependencies. You are responsible for your use and acceptance of any accompanying terms.

### Python

The licenses of the PyPI dependencies are available [here](./THIRDPARTYLICENSES).

### JavaScript

The licenses of the npm dependencies are available [here](./ui-frontend/THIRDPARTYLICENSES).

### NVIDIA Container Registry (nvcr.io)

| Container | Image | Website |
|-----------|-------|---------|
| Flux (Image Generation) | `nvcr.io/nim/black-forest-labs/flux.1-kontext-dev` | https://catalog.ngc.nvidia.com/orgs/nim/teams/black-forest-labs/containers/flux.1-kontext-dev |
| Parakeet (Speech-to-Text) | `nvcr.io/nim/nvidia/parakeet-1-1b-ctc-en-us` | https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-1-1b-ctc-en-us |
| TensorRT-LLM | `nvcr.io/nvidia/tensorrt-llm/release` | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release |

> [!NOTE]
> The Flux.1-Kontext NIM uses a model that is for non-commercial use. Contact sales@blackforestlabs.ai for commercial terms. You are responsible for accepting the applicable License Agreements and Acceptable Use Policies, and for ensuring your HF token has the correct permissions.

### Docker Hub (docker.io)

| Container | Image | Website |
|-----------|-------|---------|
| MinIO | `minio/minio` | https://hub.docker.com/r/minio/minio |
| Phoenix | `arizephoenix/phoenix` | https://hub.docker.com/r/arizephoenix/phoenix |
| Grafana LGTM | `docker.io/grafana/otel-lgtm` | https://hub.docker.com/r/grafana/otel-lgtm |
| Python | `python` | https://hub.docker.com/_/python |
| Node.js | `node` | https://hub.docker.com/_/node |
| Nginx | `nginx` | https://hub.docker.com/_/nginx |
| BusyBox | `busybox` | https://hub.docker.com/_/busybox |

### GitHub Container Registry (ghcr.io)

| Container | Image | Website |
|-----------|-------|---------|
| UV Python Packager | `ghcr.io/astral-sh/uv` | https://docs.astral.sh/uv/ |

### Redpanda Registry (docker.redpanda.com)

| Container | Image | Website |
|-----------|-------|---------|
| Redpanda | `docker.redpanda.com/redpandadata/redpanda` | https://redpanda.com/ |
| Redpanda Console | `docker.redpanda.com/redpandadata/console` | https://redpanda.com/ |

### Model

| Model | Website |
|-------|---------|
| gpt-oss-20b | https://huggingface.co/openai/gpt-oss-20b |
| FLUX.1-Kontext-dev | https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev |
| FLUX.1-Kontext-dev-onnx | https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx |

> [!NOTE]
> FLUX.1-Kontext-dev and FLUX.1-Kontext-dev-onnx are models released for non-commercial use. Contact sales@blackforestlabs.ai for commercial terms. You are responsible for accepting the applicable License Agreements and Acceptable Use Policies, and for ensuring your HF token has the correct permissions.

## Contributions

This project does currently not accept external contributions.

## Security Notice

This playbook is intended for local DGX Spark deployment only. It is provided as a reference implementation and is not production-ready. Do not expose any of its services beyond your trusted, local development environment.
