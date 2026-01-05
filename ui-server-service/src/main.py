# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import re
import signal
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from math import sqrt
from urllib.parse import urlparse

import uvicorn
from configuration import UiServerConfig
from fastapi import FastAPI, WebSocket
from messages import (
    AppState,
    AskHuman,
    BoundingCircle,
    FillCircleAnimation,
    GenerateImage,
    LookAtHuman,
    StaticAnimation,
    Transcript,
)
from pydantic import HttpUrl
from session import Session
from workmesh.config import load_config
from workmesh.messages import (
    Color,
    Command,
    Frame,
    Keypoint,
    LightCommand,
    PresenceStatus,
    RemoteControlCommand,
    UserDetection,
    UserUtteranceStatus,
)
from workmesh.messages import (
    Service as ServiceEnum,
)
from workmesh.service import Service
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    ClipStatus,
    HumanSpeechRequest,
    ServiceCommand,
    ToolStatus,
    UserState,
    UserUtterance,
    camera_frame_topic,
    clip_status_topic,
    human_speech_request_topic,
    light_command_topic,
    remote_control_command_topic,
    routed_user_utterance_topic,
    service_command_topic,
    subscribe,
    tool_status_topic,
    user_detection_topic,
    user_state_topic,
)


class State:
    def __init__(self, minio_host: str):
        self.state = AppState(tracking_data=[])
        self.user_state: PresenceStatus | None = None
        self.tracker_enabled: bool = True
        self.minio_host = minio_host
        self.qr_code_task: asyncio.Task[None] | None = None
        self.transcript_task: asyncio.Task[None] | None = None

    async def update(
        self,
        utterance: UserUtterance,
        update_callback: Callable[[AppState], Awaitable[None]],
    ):
        # Cancel any pending clear task since we have new speech
        if self.transcript_task:
            self.transcript_task.cancel()
            self.transcript_task = None

        match utterance.status:
            case UserUtteranceStatus.USER_UTTERANCE_UPDATED:
                self.state.transcript = Transcript(text=utterance.text, author="User")
                return self.state
            case UserUtteranceStatus.USER_UTTERANCE_FINISHED:
                self.state.transcript = Transcript(text=utterance.text, author="User")

                # Wait 3 seconds before clearing the transcript
                duration = 3.0

                async def clear_task():
                    await asyncio.sleep(duration)
                    self.state.transcript = None
                    await update_callback(self.state)

                self.transcript_task = asyncio.create_task(clear_task())
                return self.state
        return None

    async def update_bot_speech(self, request: HumanSpeechRequest):
        # Cancel any pending clear task
        if self.transcript_task:
            self.transcript_task.cancel()
            self.transcript_task = None

        self.state.transcript = Transcript(
            text=request.script, author="Bot", id=request.action_uuid
        )
        return self.state

    async def update_clip_status(
        self,
        clip_status: ClipStatus,
        update_callback: Callable[[AppState], Awaitable[None]],
    ):
        if (
            clip_status.status == ClipStatus.Status.FINISHED
            and self.state.transcript
            and self.state.transcript.id == clip_status.action_uuid
        ):
            # Wait 3 seconds before clearing the transcript
            duration = 3.0

            async def clear_task():
                await asyncio.sleep(duration)
                self.state.transcript = None
                await update_callback(self.state)

            self.transcript_task = asyncio.create_task(clear_task())
            return self.state
        return None

    def distance(self, a: Keypoint, b: Keypoint):
        return sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    async def update_service(self, command: ServiceCommand):
        match command.target_service:
            case ServiceEnum.AGENT:
                if command.command == Command.RESTART:
                    self.state = AppState(tracking_data=[])
                    print("Restarting, going to sleep...")
                    return self.state
            case ServiceEnum.TRACKER:
                self.state.tracking_data = []
                self.tracker_enabled = command.command == Command.ENABLE
                return self.state

    async def update_detection(self, detection: UserDetection):
        if not self.tracker_enabled:
            return None
        tracking_data = []
        for i, skel in enumerate(detection.skeletons):
            nose = skel.keypoints[0]
            left_ear = skel.keypoints[3]
            right_ear = skel.keypoints[4]
            gravity_center_x = (left_ear.x + right_ear.x + nose.x) / 3
            gravity_center_y = (left_ear.y + right_ear.y + nose.y) / 3
            gravity_center = Keypoint(
                x=gravity_center_x, y=gravity_center_y, confidence=1
            )
            radius = max(
                self.distance(nose, gravity_center),
                self.distance(left_ear, gravity_center),
                self.distance(right_ear, gravity_center),
            )

            is_primary = i == detection.marker_id
            if self.state == PresenceStatus.USER_DISAPPEARED:
                is_primary = False
            tracking_data.append(
                BoundingCircle(
                    center_x=gravity_center_x,
                    center_y=gravity_center_y,
                    radius=radius,
                    is_primary=is_primary,
                )
            )
        self.state.tracking_data = tracking_data
        return self.state

    async def update_tool(
        self, tool: ToolStatus, update_callback: Callable[[AppState], Awaitable[None]]
    ):
        match tool.status:
            case ToolStatus.Status.TOOL_CALL_STARTED:
                match tool.name:
                    case "ask_human":
                        if isinstance(self.state.tool, GenerateImage):
                            return None
                        self.state.tool = AskHuman()
                        return self.state
                    case "look_at_human":
                        self.state.tool = LookAtHuman()
                        return self.state
                    case "generate_image":
                        value = tool.input.get("image_url_or_path")
                        assert value is not None
                        url = urlparse(value.string_value)
                        new_url = url._replace(netloc=self.minio_host).geturl()
                        self.state.tool = GenerateImage(captured_image=HttpUrl(new_url))
                        return self.state
            case ToolStatus.Status.TOOL_CALL_PROCESSED:
                match tool.name:
                    case "ask_human":
                        if isinstance(self.state.tool, GenerateImage):
                            self.state.tool = AskHuman()
                            return self.state
            case ToolStatus.Status.TOOL_CALL_COMPLETED:
                match tool.name:
                    case "generate_image":
                        print(tool.response)
                        # Expected response format:
                        # GenerateImageOutput(image_url='...', qrcode_url='...')
                        # We use regex to extract the URLs
                        image_url_match = re.search(r"image_url='(.*?)'", tool.response)
                        qrcode_url_match = re.search(
                            r"qrcode_url='(.*?)'", tool.response
                        )

                        if image_url_match:
                            url = urlparse(image_url_match.group(1))
                            new_url = url._replace(netloc=self.minio_host).geturl()
                            assert isinstance(self.state.tool, GenerateImage)
                            captured_image = self.state.tool.captured_image
                            self.state.tool = GenerateImage(
                                captured_image=captured_image,
                                generated_image=HttpUrl(new_url),
                            )

                        if qrcode_url_match:
                            url = urlparse(qrcode_url_match.group(1))
                            new_url = url._replace(netloc=self.minio_host).geturl()
                            self.state.qr_code = HttpUrl(new_url)
                            await update_callback(self.state)
                        return self.state
                    case "look_at_human":
                        url = urlparse(tool.response.split("=")[1][1:-1])
                        new_url = url._replace(netloc=self.minio_host).geturl()
                        self.state.tool = LookAtHuman(captured_image=HttpUrl(new_url))
                        return self.state
        return None

    async def update_user_state(self, user_state: UserState):
        self.user_state = user_state.status
        if user_state.status == PresenceStatus.USER_DISAPPEARED:
            for circle in self.state.tracking_data:
                circle.is_primary = False
            return self.state
        return None

    async def update_light(self, command: LightCommand):
        if command.HasField("static_animation"):
            static = command.static_animation
            self.state.animation = StaticAnimation(
                color=Color.Name(static.color),  # type: ignore[arg-type]
                in_transition=static.in_transition_duration,
            )
            if static.color == Color.GRAY:
                self.state.countdown_started_at = time.time()
                self.state.countdown_duration = static.in_transition_duration
            else:
                self.state.countdown_started_at = None
                self.state.countdown_duration = None
        elif command.HasField("fill_circle_animation"):
            self.state.countdown_started_at = None
            self.state.countdown_duration = None
            fill = command.fill_circle_animation
            self.state.animation = FillCircleAnimation(
                primary_color=Color.Name(fill.primary_color),  # type: ignore[arg-type]
                secondary_color=Color.Name(fill.secondary_color),  # type: ignore[arg-type]
                in_transition=fill.in_transition_duration,
                duration=fill.fill_duration,
            )
        return self.state

    def handle_abort(self):
        self.state = AppState(tracking_data=[])
        return self.state


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(UiServerConfig)
    app.state.app_state = State(config.minio_public_host)
    app.state.sessions = []

    service = UiServerService(config)
    app.state.service = service
    executor = ServiceExecutor([service])

    async def run():
        try:
            await executor.run(handle_sigint=False)
        except Exception:
            os.kill(os.getpid(), signal.SIGINT)

    task = asyncio.create_task(run())
    yield
    await executor.cleanup_async()
    await task


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def connect(websocket: WebSocket):
    async with Session(websocket, on_abort=app.state.service.send_abort) as session:
        app.state.sessions.append(session)
        await session.update(app.state.app_state.state)
        await session.run()
        app.state.sessions.remove(session)


class UiServerService(Service):
    def __init__(self, config: UiServerConfig | None = None) -> None:
        super().__init__(config)

    async def send_abort(self):
        await self.publish(
            remote_control_command_topic, RemoteControlCommand(command="abort")
        )

    @subscribe(remote_control_command_topic)
    async def remote_control_command(self, command: RemoteControlCommand):
        if (
            command.command.lower() == "abort"
            and (state := app.state.app_state.handle_abort()) is not None
        ):
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(routed_user_utterance_topic)
    async def user_utterance(self, utterance: UserUtterance):
        async def update_callback(state):
            for s in app.state.sessions:
                await s.update(state)

        if (
            state := await app.state.app_state.update(utterance, update_callback)
        ) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(human_speech_request_topic)
    async def human_speech_request(self, request: HumanSpeechRequest):
        if (state := await app.state.app_state.update_bot_speech(request)) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(clip_status_topic)
    async def clip_status(self, clip_status: ClipStatus):
        async def update_callback(state):
            for s in app.state.sessions:
                await s.update(state)

        if (
            state := await app.state.app_state.update_clip_status(
                clip_status, update_callback
            )
        ) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(camera_frame_topic)
    async def camera_frame(self, frame: Frame):
        for s in app.state.sessions:
            await s.update_frame(frame)

    @subscribe(tool_status_topic)
    async def tool_status(self, tool: ToolStatus):
        async def update_callback(state):
            for s in app.state.sessions:
                await s.update(state)

        if (
            state := await app.state.app_state.update_tool(tool, update_callback)
        ) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(user_detection_topic)
    async def tracker_data(self, user_detection: UserDetection):
        if (
            state := await app.state.app_state.update_detection(user_detection)
        ) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(service_command_topic)
    async def service_command(self, service_command: ServiceCommand):
        if (
            state := await app.state.app_state.update_service(service_command)
        ) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(user_state_topic)
    async def user_state(self, user_state: UserState):
        if (
            state := await app.state.app_state.update_user_state(user_state)
        ) is not None:
            for s in app.state.sessions:
                await s.update(state)

    @subscribe(light_command_topic)
    async def light_command(self, command: LightCommand):
        if (state := await app.state.app_state.update_light(command)) is not None:
            for s in app.state.sessions:
                await s.update(state)


if __name__ == "__main__":
    config = load_config(UiServerConfig)
    uvicorn.run(
        "main:app",
        port=config.port,
        host=str(config.host),
        log_level=config.log_level.lower(),
        workers=1,
        reload=False,
    )
