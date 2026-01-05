# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import uuid

import typer
from pydantic.networks import KafkaDsn
from workmesh.config import ConsumerConfig, ProducerConfig
from workmesh.messages import (
    Command,
    HumanSpeechRequest,
    Service,
    ServiceCommand,
    ToolStatus,
    UserUtterance,
    UserUtteranceStatus,
)
from workmesh.service import Consumer, Producer

from workmesh import (
    human_speech_request_topic,
    protobuf_map_to_dict,
    service_command_topic,
    tool_status_topic,
    user_utterance_topic,
)

app = typer.Typer()


class MockUserService:
    """A mock user service that listens to text events and can respond automatically."""

    def __init__(self, broker_url: str = "kafka://localhost:19092"):
        self.broker_url = broker_url
        self.producer = None
        self.consumer = None
        self.running = False
        self.waiting_for_input = True
        self.stt_service_enabled = True

    async def setup(self):
        """Setup producer and consumer."""
        self.producer = Producer(
            config=ProducerConfig(broker_url=KafkaDsn(self.broker_url))
        )
        self.consumer = Consumer(
            config=ConsumerConfig(
                broker_url=KafkaDsn(self.broker_url), consumer_group="mock_user_service"
            )
        )
        self.consumer.subscribe(service_command_topic, self.on_service_command_received)  # type: ignore
        self.consumer.subscribe(human_speech_request_topic, self.on_text_received)  # type: ignore
        self.consumer.subscribe(
            tool_status_topic,  # type: ignore
            self.on_tool_status_received,  # type: ignore
        )

    async def on_service_command_received(self, message: ServiceCommand) -> None:
        """Handle incoming STT service command messages."""

        if message.target_service != Service.STT:
            return

        if message.command == Command.ENABLE:
            self.stt_service_enabled = True
        elif message.command == Command.DISABLE:
            self.stt_service_enabled = False

    async def on_text_received(self, message: HumanSpeechRequest) -> None:
        """Handle incoming text messages."""

        print(f"\n🤖 AGENT: {message.script}")
        self.waiting_for_input = True

    async def on_tool_status_received(self, message: ToolStatus) -> None:
        """Handle incoming tool status messages."""

        if message.status == ToolStatus.Status.TOOL_CALL_STARTED:
            print(f"\n🤖 TOOL CALL STARTED: {message.name}")
            # Extract input using common utility
            if message.name == "ask_human":
                prompt_value = protobuf_map_to_dict(message.input).get("prompt")
                print(f"\n🤖 AGENT: {prompt_value}")
                self.waiting_for_input = True

        elif message.status == ToolStatus.Status.TOOL_CALL_COMPLETED:
            print(f"\n🤖 TOOL CALL COMPLETED: {message.name}")
        elif message.status == ToolStatus.Status.TOOL_CALL_FAILED:
            print(f"\n🤖 TOOL CALL FAILED: {message.name}")

    async def send_text(self, text: str):
        """Send a text message."""

        if not self.stt_service_enabled:
            print(f"\n🤖 STT SERVICE IS OFF, NOT SENDING TEXT: {text}")
            return

        action_uuid = str(uuid.uuid4())
        user_utterance_started = UserUtterance(
            action_uuid=action_uuid,
            status=UserUtteranceStatus.USER_UTTERANCE_STARTED,
            timestamp=int(time.time() * 1000),
            text=text,
        )
        if self.producer:
            await self.producer.publish(user_utterance_topic, user_utterance_started)
        user_utterance_finished = UserUtterance(
            action_uuid=action_uuid,
            status=UserUtteranceStatus.USER_UTTERANCE_FINISHED,
            timestamp=int(time.time() * 1000),
            text=text,
        )

        if self.producer:
            await self.producer.publish(user_utterance_topic, user_utterance_finished)
            await self.producer.flush()
        self.waiting_for_input = False

    async def run_consumer(self):
        """Run the consumer loop."""
        self.running = True
        try:
            while self.running and self.consumer:
                await self.consumer.consume()
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping mock user service...")
            self.running = False

    async def read_terminal_input(self):
        """Read input from terminal only when waiting for input."""
        loop = asyncio.get_event_loop()

        def get_input():
            try:
                return input("🙋 USER: ")
            except EOFError:
                return None

        print("\n💬 Type your message and press Enter to send:")
        print("   Type 'quit', 'exit', or 'q' to stop\n")

        while self.running:
            if self.waiting_for_input:
                try:
                    user_input = await loop.run_in_executor(None, get_input)

                    if user_input is None:  # EOFError
                        break

                    user_input = user_input.strip()

                    if user_input.lower() in ["quit", "exit", "q"]:
                        print("👋 Goodbye!")
                        self.running = False
                        break
                    elif user_input:
                        await self.send_text(user_input)

                except Exception as e:
                    print(f"Error reading input: {e}")
                    break
            else:
                await asyncio.sleep(0.1)

    async def stop(self):
        """Stop the service and cleanup resources."""
        self.running = False
        if self.producer:
            await self.producer._producer.stop()
        if self.consumer:
            await self.consumer._consumer.stop()


@app.command(name="interactive")
def run_mock_service() -> None:
    """Run the mock user service that listens to text events and responds."""

    async def main():
        service = MockUserService()
        await service.setup()

        print("🚀 Starting Mock User Service...")
        print("   - Listening for AGENT messages (will be logged)")
        print("   - Reading USER input from terminal")
        print("   - Press Ctrl+C to stop")

        try:
            consumer_task = asyncio.create_task(service.run_consumer())
            input_task = asyncio.create_task(service.read_terminal_input())

            done, pending = await asyncio.wait(
                [consumer_task, input_task], return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()

        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
        finally:
            await service.stop()

    asyncio.run(main())


@app.command(name="listen")
def listen_only() -> None:
    """Listen to text events without responding."""

    async def main():
        consumer = Consumer(
            config=ConsumerConfig(
                broker_url=KafkaDsn("kafka://localhost:19092"),
                consumer_group="text_listener",
            )
        )

        async def on_text(message: HumanSpeechRequest) -> None:
            print(f"📥 [{message.robot_id}] {message.script}")

        consumer.subscribe(human_speech_request_topic, on_text)  # type: ignore

        print("👂 Listening to text events... Press Ctrl+C to stop")

        try:
            while True:
                await consumer.consume()
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping listener...")
        finally:
            await consumer._consumer.stop()

    asyncio.run(main())


if __name__ == "__main__":
    app()
