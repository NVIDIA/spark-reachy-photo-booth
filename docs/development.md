# Development

This section provides comprehensive instructions for customizing and developing upon the Reachy Photo Booth application. If you're looking to deploy and run the application as-is, refer to the [Instructions](instructions.md) tab instead — this Development guide is specifically for those who need to make modifications to the application.

# Step 1. System dependencies

In order to use the Python development setup of the repository install the following packages:

```bash
sudo apt install python3.12-dev portaudio19-dev
```

To create the Python **venv** install uv by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to generate the Python **venv**:

```bash
uv sync --all-packages
```

# Step 2. Get acquainted with the build and development process

Every folder suffixed by `-service` is a standalone Python program that runs in its own container. You must always start the services by interacting with the `docker-compose.yaml` at the root of the repository. You can enable code hot reloading for all the Python services by running:

```bash
docker compose up -d --build --watch
```

Whenever you change some Python code in the repository the associated container will be updated and automatically restarted.

The [Getting Started](getting-started.md) guide provides a comprehensive walkthrough of the build system, development workflow, debugging strategies, and monitoring infrastructure.

# Step 3. Make changes to the application

Now that your development environment is set up, here are the most common customizations developers typically explore.

## Customize configuration parameters

Each service has configurable parameters including system prompts, audio devices, model settings, and more. Check the individual service READMEs and the `src/configuration.py` files for detailed configuration options. Note that the default configuration in `src/configuration.py` might also be overridden in the `compose.yaml` file. Check out the following services to get started:

- [speech-to-text-service](../speech-to-text-service/README.md) - Configure audio devices and transcription settings
- [text-to-speech-service](../text-to-speech-service/README.md) - Adjust voice synthesis parameters
- [agent-service](../agent-service/README.md) - Customize LLM system prompts, agent behavior, and decision logic

See the [instructions](instructions.md) for a complete list of all services and their READMEs.

## Extend the demo with new tools

The agent-service and interaction-manager-service are the core services for extending the demo with new capabilities:

- [agent-service](../agent-service/README.md) - Add new agent tools and capabilities here
- [interaction-manager-service](../interaction-manager-service/README.md) - Manage event orchestration and robot utterances

## Create your own service

The [Writing Your First Service](writing-your-first-service.md) guide provides a step-by-step tutorial on scaffolding, implementing, and integrating a new microservice into the system. Follow this guide to create custom services that extend the photo booth functionality.
