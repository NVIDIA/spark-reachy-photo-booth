# Metrics Service

## Overview

The Metrics service derives **OpenTelemetry metrics** from the system's Kafka message flow. It subscribes to request/response topics and emits histograms for latency and throughput.

This service does **not** expose an HTTP API; metrics are exported to the stack's OpenTelemetry Collector (LGTM) over **OTLP gRPC**.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |
| `lgtm-otel` | OpenTelemetry Collector (Grafana LGTM stack) |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run metrics-service/src/main.py
```

## Emitted metrics

- **`tts.latency` (Histogram, ms)**: time from `HumanSpeechRequest` → `ClipData` (correlated by `action_uuid`).
- **`tts.rate` (Histogram, chars/s)**:  script_length_chars / latency_seconds (convert ms → s) based on the same correlation.
- **`e2e.latency` (Histogram, ms)**: time from `UserUtterance` with status `USER_UTTERANCE_FINISHED` → next `ClipData` within a 3-minute window.

## Kafka topics
The service listens to:

- `human_speech_request_topic`
- `user_utterance_topic`
- `clip_data_topic`

## Configuration
Configuration is loaded from `/config.yaml` if present (see `workmesh.config.load_config`). If `/config.yaml` is not provided, defaults are used.

Common fields (inherited from `workmesh.config.BaseConfig`):

- **`broker_url`**: Kafka URL (default is `kafka://redpanda:9092` in Docker, otherwise `kafka://localhost:19092`)
- **`consumer_group`**: Kafka consumer group (defaults to the class name)
- **`offset_type`**: `latest` or `earliest` (default `latest`)
- **`enable_auto_commit`**: default `true`
- **`otel_endpoint`**: OTLP gRPC endpoint (default `http://lgtm-otel:4317`)
- **`log_level`**: `DEBUG|INFO|WARNING|ERROR|CRITICAL` (default `INFO`)

Example `/config.yaml`:

```yaml
broker_url: kafka://redpanda:9092
consumer_group: metrics
offset_type: latest
otel_endpoint: http://lgtm-otel:4317
log_level: INFO
```

## Running

### With the main stack
Bring up the core stack (including Kafka/Redpanda + LGTM):

```bash
docker compose up --build
```

### Debug / live code sync
The debug override keeps the container running and bind-mounts `src/` for rapid iteration:

```bash
docker compose -f metrics-service/compose.yaml -f metrics-service/service-debug.override.yaml up --build
```

## Viewing metrics
The default `otel_endpoint` points at the repo’s LGTM container. Once running, open Grafana at `http://localhost:3000` and explore metrics (Prometheus data source in the LGTM image).


