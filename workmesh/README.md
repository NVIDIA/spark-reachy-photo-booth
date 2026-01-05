# Workmesh

## Overview

Workmesh is a shared Python library that provides the messaging framework for Kafka-based communication between services. It defines message types using Protocol Buffers, manages topic subscriptions and publications, and provides base classes and decorators for building event-driven services.

## Generating protos

```bash
cd src/workmesh/messages
python -m grpc_tools.protoc --python_out=. --pyi_out=. -I. messages.proto
```