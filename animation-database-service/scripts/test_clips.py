# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, Literal

import typer
from workmesh.messages import (
    AnimationProperties,
    AudioProperties,
    ChangeVolume,
    ClipProperties,
    LookAtParameters,
    PlayClip,
    Position2D,
    ProceduralClip,
    ProceduralClipPause,
    ProceduralClipStart,
    ProceduralClipStop,
    ProceduralState,
    ProceduralType,
    StopClip,
    TrackParameters,
    Transition,
)
from workmesh.messages import Robot as RobotProto
from workmesh.service import Producer, Topic

from workmesh import (
    change_volume_topic,
    play_clip_topic,
    procedural_clip_topic,
    stop_clip_topic,
)

app = typer.Typer()


class Robot(Enum):
    RESEARCHER = "researcher"

    def to_proto(self) -> RobotProto:
        if self == Robot.RESEARCHER:
            return RobotProto.RESEARCHER
        raise ValueError(f"Invalid robot: {self}")


async def producer_send(topic: Topic, message: Any) -> None:
    async with Producer() as producer:
        await producer.publish(topic, message)


@app.command(name="play")
def play_clip(
    clip_name: str,
    action_uuid: str = "123",
    robot_id: Robot = Robot.RESEARCHER,
    loop: bool = False,
    loop_overlap: float = 1.0,
    priority: int = 0,
    opacity: float = 1,
    blend_in: float = 0.5,
    blend_out: float = 0.5,
    volume: float = 1,
    fade_in: float = 0.0,
    fade_out: float = 0.0,
):
    message = PlayClip(
        action_uuid=action_uuid,
        robot_id=robot_id.to_proto(),
        clip_name=clip_name,
        clip_properties=ClipProperties(loop=loop, loop_overlap=loop_overlap),
        animation_properties=AnimationProperties(
            priority=priority,
            opacity=opacity,
            blending=Transition(transition_in=blend_in, transition_out=blend_out),
        ),
        audio_properties=AudioProperties(
            volume=volume,
            fading=Transition(transition_in=fade_in, transition_out=fade_out),
        ),
    )

    asyncio.run(producer_send(play_clip_topic, message))


@app.command(name="play_multiple")
def play_multiple(clip_names: list[str], time_between_clips: float = 2.0):
    async def play_clip_async():
        async with Producer() as producer:
            for clip_name in clip_names:
                action_uuid = str(uuid.uuid4())
                await producer.publish(
                    play_clip_topic,
                    PlayClip(clip_name=clip_name, action_uuid=action_uuid),
                )
                print(f"Clip {clip_name} Action UUID: {action_uuid}")
                await asyncio.sleep(time_between_clips)

    asyncio.run(play_clip_async())


@app.command(name="volume")
def change_volume(volume: float, action_uuid: str = "123"):
    asyncio.run(
        producer_send(
            change_volume_topic, ChangeVolume(action_uuid=action_uuid, volume=volume)
        )
    )


@app.command(name="stop")
def stop_clip(action_uuid: str = "123", fade_out: float = 0.0):
    asyncio.run(
        producer_send(
            stop_clip_topic, StopClip(action_uuid=action_uuid, fade_out=fade_out)
        )
    )


@app.command(name="look_at", context_settings={"ignore_unknown_options": True})
def look_at(
    x: float = 1.0,
    y: float = 1.0,
    duration: float = 3.0,
    volume: float = 1.0,
    robot_id: Robot = Robot.RESEARCHER,
    state: Literal["start", "pause", "stop"] = "start",
    enable_pause: bool = True,
):
    if state == "start":
        message = ProceduralClip(
            robot_id=robot_id.to_proto(),
            action_uuid=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            type=ProceduralType.LOOK_AT,
            state=ProceduralState.START,
            start=ProceduralClipStart(
                volume=volume,
                look_at_parameters=LookAtParameters(
                    target_position=Position2D(x=x, y=y),
                    duration=duration,
                ),
            ),
        )
    elif state == "pause":
        message = ProceduralClip(
            robot_id=robot_id.to_proto(),
            action_uuid=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            type=ProceduralType.LOOK_AT,
            state=ProceduralState.PAUSE,
            pause=ProceduralClipPause(enable=enable_pause),
        )

    elif state == "stop":
        message = ProceduralClip(
            robot_id=robot_id.to_proto(),
            action_uuid=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            type=ProceduralType.LOOK_AT,
            state=ProceduralState.STOP,
            stop=ProceduralClipStop(),
        )

    asyncio.run(producer_send(procedural_clip_topic, message))


@app.command(name="track")
def track(
    robot_id: Robot = Robot.RESEARCHER,
    slow_mode_distance_threshold: float = 0.3,
    fast_mode_distance_threshold: float = 0.5,
    state: Literal["start", "stop", "pause"] = "start",
    blend_in: float = 0.75,
    blend_out: float = 0.75,
    enable_pause: bool = True,
    slow_speed: float = 1.0,
    fast_speed: float = 2.0,
):
    if state == "start":
        message = ProceduralClip(
            robot_id=robot_id.to_proto(),
            action_uuid=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            type=ProceduralType.TRACK,
            state=ProceduralState.START,
            start=ProceduralClipStart(
                blend_in_duration=blend_in,
                track_parameters=TrackParameters(
                    slow_mode_distance_threshold=slow_mode_distance_threshold,
                    fast_mode_distance_threshold=fast_mode_distance_threshold,
                    slow_speed=slow_speed,
                    fast_speed=fast_speed,
                ),
            ),
        )

    elif state == "stop":
        message = ProceduralClip(
            robot_id=robot_id.to_proto(),
            action_uuid=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            type=ProceduralType.TRACK,
            state=ProceduralState.STOP,
            stop=ProceduralClipStop(blend_out_duration=blend_out),
        )

    elif state == "pause":
        message = ProceduralClip(
            robot_id=robot_id.to_proto(),
            action_uuid=str(uuid.uuid4()),
            timestamp=int(time.time() * 1000),
            type=ProceduralType.TRACK,
            state=ProceduralState.PAUSE,
            pause=ProceduralClipPause(enable=enable_pause),
        )

    asyncio.run(producer_send(procedural_clip_topic, message))


if __name__ == "__main__":
    app()
