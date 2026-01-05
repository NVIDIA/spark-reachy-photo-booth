# Getting Started

This guide will walk you through setting up and running Spark & Reachy Photo Booth on your development environment.

## Prerequisites

- Docker and Docker Compose
- UV package manager
- VS Code (optional, for debugging)
- Nix (optional)

## Quick Start

### 1. Start the Development Environment

To start the complete development environment with all services:

```shell
docker compose up --build --watch
```

This command will:


- Build all Docker images
- Start the Workmesh services
- Launch the observability stack (LGTM)
- Start the message broker (Redpanda)
- Enable hot-reload for development

### 2. Access the Management Interfaces

Once running, you can access:

- **Redpanda Console**: `http://127.0.0.1:8080` - Message broker administration and topic exploration
- **Grafana Dashboard**: `http://127.0.0.1:3000` - Full LGTM observability stack

### 3. Customize Configuration

Modify your service's `compose.yaml` file to:

- Add environment variables
- Bind additional ports
- Mount volumes
- Configure service dependencies

## Development Workflow

### Hot Reload Development

The development environment provides automatic code reloading:

- Changes to Python files under `src/` trigger container restarts
- Dependencies added via `uv add <package>` automatically rebuild and restart containers
- File modifications are detected and applied in real-time

### Managing Dependencies

Add Python packages to your service:

```shell
cd <your-service-directory>
uv add <package-name>
```

Docker will automatically detect the `pyproject.toml` changes and rebuild the container.

### Run Formatter

```shell
ruff format
```

### Run Linters

```shell
ruff check
```

### Run Type Checker

```shell
uv run pyright
```

### Synchronize Dependencies

Run the following command to synchronize the current environment with the dependencies of all workspaces.

```shell
uv sync --all-packages
```

Running this command and restarting your IDE or language server might be needed for your imports to be recognized and your auto-completion to work properly.

### Selective Service Startup

Start only specific services for focused development:

```shell
docker compose up --build --watch redpanda redpanda-console lgtm-otel service-1 service-2
```

### Filter Logs

To see only a subset of the logs of the service you can start the services with the following command

```shell
docker compose up --build --watch --no-attach redpanda --no-attach redpanda-console --no-attach lgtm-otel --no-attach minio
```

## Debugging

### Basic Debugging

Use print statements and logging for immediate feedback. The hot-reload system will apply changes instantly.

### Advanced Debugging with VS Code

For breakpoint debugging within containers:

1. Install VS Code extensions: "Dev Containers" and "Remote Explorer"
2. Start with debug configuration:

   ```shell
   docker compose -f docker-compose.yaml -f $NAME/service-debug.override.yaml up --build
   ```

3. In VS Code Remote Explorer, right-click your container and select "Attach in a new window"
4. Open the `/app` folder in the container
5. Install the Python Debugger extension
6. Set breakpoints and debug as normal

Changes made in the container are synchronized with your local `src/` directory.

## Configuration Management

### File Watching

The system automatically watches and responds to:

- `pyproject.toml` modifications (triggers rebuilds)
- Source files under `src/` (triggers hot-reload)

### Custom Watch Patterns

Add additional file watch patterns by modifying your service's `compose.yaml` configuration.

## Monitoring and Observability

### Message Broker Monitoring

Access the Redpanda Console at `http://127.0.0.1:8080` to:

- View active topics
- Inspect message content
- Monitor queue depths
- Debug message flow

### Application Metrics

The LGTM stack at `http://127.0.0.1:3000` provides:

- **Loki**: Centralized logging
- **Grafana**: Visualization and dashboards
- **Tempo**: Distributed tracing
- **Mimir**: Metrics collection

### OpenTelemetry Integration

Services can send telemetry data to:

- HTTP endpoint: `http://lgtm-otel:4318`
- gRPC endpoint: `lgtm-otel:4317`

## Next Steps

- Explore the ping-pong example to understand service communication patterns
- Review the Workmesh API documentation for advanced service development
- Configure custom observability dashboards for your specific use case
