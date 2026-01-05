# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for the Text-to-Speech service."""

import uuid

import pytest
from test_utils import MessageListener
from workmesh.config import ConsumerConfig
from workmesh.messages import Robot
from workmesh.service import Producer

from workmesh import (
    HumanSpeechRequest,
    clip_data_topic,
    human_speech_request_topic,
)


@pytest.mark.integration
@pytest.mark.tts
@pytest.mark.asyncio
async def test_tts_generates_audio_from_text(
    producer: Producer,
    consumer_config: ConsumerConfig,
):
    """Test that TTS service generates audio from a text script."""
    action_uuid = str(uuid.uuid4())
    robot_id = Robot.RESEARCHER
    test_script = "Hello, this is a test of the text to speech system."

    async with MessageListener(clip_data_topic, consumer_config) as listener:
        # Send speech request
        request = HumanSpeechRequest(
            action_uuid=action_uuid,
            robot_id=robot_id,
            script=test_script,
        )
        await producer.publish(human_speech_request_topic, request)
        await producer.flush()

        # Wait for TTS to generate audio
        await listener.wait_for(min_count=1, timeout=10.0)

        # Find response with matching UUID
        response = None
        for msg in listener.messages:
            if msg.action_uuid == action_uuid:
                response = msg
                break

        assert response is not None, "Should receive response with matching UUID"
        assert response.action_uuid == action_uuid, "Action UUID should match"
        assert response.robot_id == robot_id, "Robot ID should match"
        assert response.HasField("audio"), "Response should contain audio"
        assert len(response.audio.audio_buffer) > 0, "Audio buffer should not be empty"
        assert response.audio.sample_rate > 0, "Sample rate should be positive"
        assert response.audio.bits_per_sample > 0, "Bits per sample should be positive"
        assert response.audio.channel_count > 0, "Channel count should be positive"


@pytest.mark.integration
@pytest.mark.tts
@pytest.mark.asyncio
async def test_tts_handles_multiple_requests(
    producer: Producer,
    consumer_config: ConsumerConfig,
):
    """Test that TTS service can handle multiple sequential requests."""
    robot_id = Robot.RESEARCHER
    requests_data = [
        (str(uuid.uuid4()), "First test message."),
        (str(uuid.uuid4()), "Second test message."),
        (str(uuid.uuid4()), "Third test message."),
    ]

    async with MessageListener(clip_data_topic, consumer_config) as listener:
        # Send all requests
        for action_uuid, script in requests_data:
            request = HumanSpeechRequest(
                action_uuid=action_uuid,
                robot_id=robot_id,
                script=script,
            )
            await producer.publish(human_speech_request_topic, request)

        await producer.flush()

        # Wait for all responses
        await listener.wait_for(min_count=3, timeout=15.0)

        # Verify we got all expected responses
        received_uuids = {msg.action_uuid for msg in listener.messages}
        expected_uuids = {uuid for uuid, _ in requests_data}

        assert received_uuids == expected_uuids, (
            "Should receive responses for all requests"
        )

        # Verify all responses have audio
        for msg in listener.messages:
            assert msg.HasField("audio"), "Response should contain audio"
            assert len(msg.audio.audio_buffer) > 0, "Audio buffer should not be empty"
