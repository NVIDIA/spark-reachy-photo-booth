# Spark & Reachy Photo Booth Integration Tests

Integration tests for Spark & Reachy Photo Booth services via Kafka messaging.

## Quick Start

```bash
# Install dependencies
cd tests && uv sync

# Start services
docker compose up minio redpanda agent text-to-speech

# Run tests (if in container, set KAFKA_BROKER_URL=kafka://redpanda:9092)
uv run pytest -v
```

## Running Tests

### Start Services

```bash
docker compose up redpanda text-to-speech  # For TTS tests
docker compose up minio redpanda agent           # For Agent tests
```

### Run Tests

```bash
cd tests
uv run pytest              # All tests
uv run pytest -m tts       # Only TTS tests
uv run pytest -m agent     # Only Agent tests
uv run pytest -m "not slow"  # Skip slow tests
```

### Configure Kafka URL

```bash
# CLI option
uv run pytest --kafka-url kafka://redpanda:9092

# Environment variable
export KAFKA_BROKER_URL=kafka://redpanda:9092
uv run pytest
```

**Default**: `kafka://localhost:19092`
**Inside containers**: Use `kafka://redpanda:9092`

## Writing Tests

### Basic Pattern

```python
import time
import uuid
import pytest
from test_utils import MessageListener
from workmesh import your_request_topic, your_response_topic
from workmesh.config import ConsumerConfig
from workmesh.service import Producer

@pytest.mark.integration
@pytest.mark.asyncio
async def test_your_service(producer: Producer, consumer_config: ConsumerConfig):
    action_uuid = str(uuid.uuid4())

    async with MessageListener(your_response_topic, consumer_config) as listener:
        # Send request
        request = YourRequestMessage(action_uuid=action_uuid, ...)
        await producer.publish(your_request_topic, request)
        await producer.flush()

        # Wait for response
        await listener.wait_for(min_count=1, timeout=10.0)

        # Validate
        response = next((m for m in listener.messages if m.action_uuid == action_uuid), None)
        assert response is not None
        assert response.some_field == expected_value
```

## Fixtures

- **`producer`**: Configured workmesh Producer for publishing messages
- **`consumer_config`**: ConsumerConfig with unique consumer group and latest offset

## Utilities

### `MessageListener`

Context manager for collecting messages from a topic:

```python
async with MessageListener(topic, consumer_config) as listener:
    # ... publish messages ...
    await listener.wait_for(min_count=1, timeout=10.0)
    # Access listener.messages
```

### Agent Test Helpers

```python
from test_agent import ToolCall, parse_tool_calls

tool_calls = parse_tool_calls(listener.messages)
ask_human_calls = [tc for tc in tool_calls if tc.tool_name == "ask_human"]
started = [tc for tc in ask_human_calls if tc.is_started()]
```

## Troubleshooting

**Tests timeout:**
- Verify services are running: `docker compose ps`
- Check service logs: `docker compose logs <service>`
- Increase timeout if needed

**Connection refused:**
- Ensure Kafka is running
- Check broker URL matches your environment (localhost:19092 vs redpanda:9092)

**No messages received:**
- Verify service subscribed to correct topic
- Check Redpanda Console: http://localhost:8080
- View topic messages: `docker exec -it redpanda rpk topic consume <topic_name>`

## Test Markers

- `integration`: Requires running services
- `tts`: Text-to-Speech tests
- `agent`: Agent tests
- `slow`: Long-running tests (30s+)
