# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
from asyncio import Task
from collections.abc import AsyncGenerator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Concatenate

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord
from google.protobuf.message import Message

from workmesh.config import BaseConfig, ConsumerConfig, ProducerConfig
from workmesh.service_telemetry import ServiceTelemetry


@dataclass
class Topic[V: Message]:
    name: str
    value_type: type[V]

    def __hash__(self):
        return hash(self.name)


def subscribe[V: Message, S: Service](topic: Topic[V]):
    """
    Decorator to subscribe a method to a Kafka topic.

    The decorated method can have one of these signatures:
        async def handler(self, message: V) -> None
        async def handler(self, message: V, kafka_record) -> None
    """

    def decorator(method: Callable[..., Coroutine[Any, Any, Any]]):
        setattr(method, Service.TOPIC_ATTR_NAME, topic)
        return method

    return decorator


def produces[V: Message, S: Service](topic: Topic[V]):
    def decorator[**P](
        method: Callable[Concatenate[S, P], AsyncGenerator[V, None]],
    ):
        async def wrapper(self: S, *args: P.args, **kwargs: P.kwargs) -> None:
            async for message in method(self, *args, **kwargs):
                await self.publish(topic, message)

        return wrapper

    return decorator


class Producer:
    def __init__(self, config: ProducerConfig | None = None):
        if config is None:
            config = ProducerConfig()
        assert config.broker_url.host is not None
        self._producer: AIOKafkaProducer = AIOKafkaProducer(
            bootstrap_servers=config.broker_url.host
            + ":"
            + str(config.broker_url.port),
            max_request_size=config.max_request_size,
        )
        self._started = False

    async def publish[V: Message](self, topic: Topic[V], message: V) -> None:
        if not self._started:
            await self._producer.start()
            self._started = True
        await self._producer.send(topic.name, value=message.SerializeToString())

    async def stop(self) -> None:
        await self._producer.flush()
        await self._producer.stop()
        if not self._started:
            return
        self._started = False

    async def flush(self) -> None:
        await self._producer.flush()

    async def __aenter__(self) -> "Producer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.stop()


class Consumer:
    TOPIC_ATTR_NAME = "__registered_topic__"

    def __init__[V: Message, S: Service](self, config: ConsumerConfig | None = None):
        if config is None:
            config = ConsumerConfig()
        assert config.broker_url.host is not None
        self._consumer: AIOKafkaConsumer = AIOKafkaConsumer(
            bootstrap_servers=config.broker_url.host
            + ":"
            + str(config.broker_url.port),
            group_id=(
                config.consumer_group
                if config.consumer_group
                else self.__class__.__name__
            ),
            auto_offset_reset=config.offset_type.value,
            enable_auto_commit=config.enable_auto_commit,
        )
        self._subscriptions: dict[
            Topic[Message], list[Callable[..., Coroutine[Any, Any, Any]]]
        ] = {}
        self._subscribed = False

    async def consume(self) -> tuple[Message, Topic[Message]] | None:
        if not self._subscribed:
            if len(self._subscriptions.items()) != 0:
                self._consumer.subscribe([topic.name for topic in self._subscriptions])
            await self._consumer.start()
            self._subscribed = True
        msg: ConsumerRecord[None, bytes] = await self._consumer.getone()
        for topic, callbacks in self._subscriptions.items():
            proto_msg = topic.value_type()
            if topic.name == msg.topic:
                assert msg.value is not None
                proto_msg.ParseFromString(msg.value)
                for callback in callbacks:
                    # If the handler accepts kafka_record in their parameters.
                    # provide Kafka metadata for correlation/metrics.
                    try:
                        sig = inspect.signature(callback)
                        accepts_kafka_record = "kafka_record" in sig.parameters
                    except (TypeError, ValueError):
                        # Builtins / C-extensions may not have signatures;
                        # fall back to legacy call.
                        accepts_kafka_record = False

                    if accepts_kafka_record:
                        asyncio.create_task(callback(proto_msg, kafka_record=msg))
                    else:
                        asyncio.create_task(callback(proto_msg))
                return (proto_msg, topic)
        return None

    def subscribe(
        self,
        topic: Topic[Message],
        callback: Callable[..., Coroutine[Any, Any, Any]] | None = None,
    ) -> None:
        if topic not in self._subscriptions:
            self._subscriptions[topic] = []
        if callback:
            self._subscriptions[topic].append(callback)

    async def close(self) -> None:
        await self._consumer.stop()
        self._subscribed = False


class Service(Producer, Consumer):
    def __init__[V: Message, S: Service](
        self, config: BaseConfig | None = None
    ) -> None:
        if config is None:
            config = BaseConfig()
        Consumer.__init__(self, config)
        Producer.__init__(self, config)

        self.service_telemetry = ServiceTelemetry(
            service_name=self.__class__.__name__,
            service_instance_id=config.consumer_group,
            otel_endpoint=config.otel_endpoint,
            log_level=config.log_level,
        )
        self.meter_provider = self.service_telemetry.get_meter_provider()
        self.logger = self.service_telemetry.getLogger()

        self._tasks: list[Task[Any]] = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if topic := getattr(attr, Service.TOPIC_ATTR_NAME, None):
                self.subscribe(topic, attr)

    def create_task(self, coroutine: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine)
        self._tasks.append(task)
        return task

    async def stop(self) -> None:
        self.logger.info("Stopping service %s", self.__class__.__name__)
        self._subscriptions = {}
        for task in self._tasks:
            task.cancel()
        await Producer.stop(self)
        await self._consumer.stop()
        self.service_telemetry.shutdown()

    async def _start_consuming(self) -> None:
        try:
            while True:
                await self.consume()
        finally:
            await Producer.stop(self)
            await self._consumer.stop()

    async def run(self) -> None:
        try:
            self.create_task(self._start_consuming())
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            self.logger.info("Shutting down service.")
