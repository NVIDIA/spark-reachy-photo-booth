# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import uuid

import statesman
from configuration import VoiceConfig
from event_manager import EventManager
from robot_utterance_manager import RobotUtteranceManager
from utils import Position, Robot
from workmesh.messages import (
    AnimationProperties,
    AudioProperties,
    ClipProperties,
    HumanSpeechRequest,
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

from workmesh import (
    Service,
    human_speech_request_topic,
    play_clip_topic,
    procedural_clip_topic,
    stop_clip_topic,
)


class RobotStateMachine(statesman.StateMachine):
    class Config:
        arbitrary_types_allowed = True

    robot_id: Robot
    voice_config: VoiceConfig
    _service: Service
    _logger: logging.Logger
    _event_manager: EventManager
    _robot_utterance_manager: RobotUtteranceManager

    # Global clip volume
    global_clip_volume: float = 1.0

    def __init__(self, service: Service, robot_id: Robot, voice_config: VoiceConfig):
        super().__init__(robot_id=robot_id, voice_config=voice_config)
        self._service = service
        self._logger = service.logger
        self._event_manager = getattr(service, "event_manager")  # noqa: B009
        self._robot_utterance_manager = getattr(service, "robot_utterance_manager")  # noqa: B009

    async def safe_trigger_event(self, event_name: str, *args, **kwargs) -> None:
        """Wrapper for trigger_event with exception handling"""
        try:
            await self.trigger_event(event_name, *args, **kwargs)
        except RuntimeError as e:
            if "event trigger failed" in str(e):
                self._logger.warning(f"Invalid transition: {e}")
            else:
                raise

    async def trigger(self, command: str) -> None:
        """Remote Control trigger"""
        raise NotImplementedError("Subclass must implement trigger method!")

    def set_global_clip_volume(self, volume: float) -> None:
        self.global_clip_volume = volume

    #########################################################
    # Play normal clips
    #########################################################

    async def play_clip(
        self,
        clip_name: str,
        priority: int,
        opacity: float,
        loop: bool = False,
        loop_overlap: float = 1.0,
        blend_in: float = 0.75,
        blend_out: float = 0.75,
        fade_in: float = 0,
        fade_out: float = 0,
        volume: float | None = None,
    ) -> str:
        """Play a clip on the robot"""
        action_uuid = str(uuid.uuid4())
        volume = volume if volume is not None else self.global_clip_volume
        message = PlayClip(
            action_uuid=action_uuid,
            robot_id=self.robot_id.to_proto(),
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
        await self._service.publish(play_clip_topic, message)
        await self.wait_for_clip_started(action_uuid)
        return action_uuid

    async def play_and_wait(
        self,
        clip_name: str,
        priority: int,
        opacity: float,
        blend_in: float = 0.75,
        blend_out: float = 0.75,
        fade_in: float = 0,
        fade_out: float = 0,
        volume: float | None = None,
    ) -> None:
        """Play a clip on the robot and wait for it to finish playing"""
        action_uuid = await self.play_clip(
            clip_name=clip_name,
            priority=priority,
            opacity=opacity,
            blend_in=blend_in,
            blend_out=blend_out,
            fade_in=fade_in,
            fade_out=fade_out,
            volume=volume,
        )
        await self.wait_for_clip(action_uuid)

    async def wait_for_clip(self, action_uuid: str) -> None:
        """Wait for a clip to finish playing"""
        await self._event_manager.wait_for_clip_finished(action_uuid)

    async def wait_for_clip_started(self, action_uuid: str) -> bool:
        """Wait for a clip to start playing"""
        return await self._event_manager.wait_for_clip_started(action_uuid)

    async def stop_clip(self, action_uuid: str, fade_out: float = 0.0) -> None:
        """Stop a clip on the robot"""

        message = StopClip(action_uuid=action_uuid, fade_out=fade_out)
        await self._service.publish(stop_clip_topic, message)

    #########################################################
    # Play procedural clips
    #########################################################

    async def track_start(
        self,
        slow_mode_distance_threshold: float,
        fast_mode_distance_threshold: float,
        slow_speed: float,
        fast_speed: float,
        blend_in_duration: float = 1.0,
    ) -> str:
        """Track an object"""
        message = ProceduralClip(
            action_uuid=str(uuid.uuid4()),
            robot_id=self.robot_id.to_proto(),
            type=ProceduralType.TRACK,
            state=ProceduralState.START,
            start=ProceduralClipStart(
                blend_in_duration=blend_in_duration,
                track_parameters=TrackParameters(
                    slow_mode_distance_threshold=slow_mode_distance_threshold,
                    fast_mode_distance_threshold=fast_mode_distance_threshold,
                    slow_speed=slow_speed,
                    fast_speed=fast_speed,
                ),
            ),
        )
        await self._service.publish(procedural_clip_topic, message)
        return message.action_uuid

    async def track_pause(self, action_uuid: str, enable: bool = True) -> None:
        """Pause tracking an object"""
        message = ProceduralClip(
            action_uuid=action_uuid,
            robot_id=self.robot_id.to_proto(),
            type=ProceduralType.TRACK,
            state=ProceduralState.PAUSE,
            pause=ProceduralClipPause(enable=enable),
        )
        await self._service.publish(procedural_clip_topic, message)

    async def track_stop(
        self, action_uuid: str, blend_out_duration: float = 1.0
    ) -> None:
        """Stop tracking an object"""
        message = ProceduralClip(
            action_uuid=action_uuid,
            robot_id=self.robot_id.to_proto(),
            type=ProceduralType.TRACK,
            state=ProceduralState.STOP,
            stop=ProceduralClipStop(
                blend_out_duration=blend_out_duration,
            ),
        )
        await self._service.publish(procedural_clip_topic, message)

    async def look_at_position(
        self,
        start_body_angle: float,
        target: Position | float,
        action_uuid: str | None = None,
        duration: float = 1.0,
        volume: float | None = None,
    ) -> None:
        """Look at a position"""
        if action_uuid is None:
            action_uuid = str(uuid.uuid4())

        if isinstance(target, Position):
            look_at_parameters = LookAtParameters(
                start_body_angle=start_body_angle,
                target_position=Position2D(x=target.x, y=target.y),
                duration=duration,
            )
        else:
            look_at_parameters = LookAtParameters(
                start_body_angle=start_body_angle,
                target_body_angle=target,
                duration=duration,
            )

        volume = volume if volume is not None else self.global_clip_volume
        message = ProceduralClip(
            action_uuid=action_uuid,
            robot_id=self.robot_id.to_proto(),
            type=ProceduralType.LOOK_AT,
            start=ProceduralClipStart(
                volume=volume,
                look_at_parameters=look_at_parameters,
            ),
        )
        await self._service.publish(procedural_clip_topic, message)
        await self.wait_for_clip(action_uuid)

    #########################################################
    # Request speech
    #########################################################

    async def request_human_speech(
        self, text: str, action_uuid: str | None = None
    ) -> str:
        """Request human speech generation for a specific robot with specific voice config"""  # noqa: E501

        if not action_uuid:
            action_uuid = str(uuid.uuid4())

        await self._service.publish(
            human_speech_request_topic,
            HumanSpeechRequest(
                action_uuid=action_uuid,
                robot_id=self.robot_id.to_proto(),
                script=text,
            ),
        )
        return action_uuid
