# UI Server Service

## Overview

The UI Server provides a real-time web interface for monitoring and visualizing the robot system. It displays the camera feed with tracking overlays, conversation transcripts, tool execution status, and generated content via WebSocket connections.

## Running the Service

### Prerequisites

Start the required infrastructure services:

| Service | Description |
|---------|-------------|
| `redpanda` | Message broker for Kafka communication |

These can be started via Docker Compose from the repository root.

### Starting the Service

```bash
uv run ui-server-service/src/main.py
```