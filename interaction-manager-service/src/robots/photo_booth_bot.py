# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import contextlib
from typing import Any

import statesman
from configuration import PhotoBoothBotConfig
from light_manager import LightEffect, LightManager
from pydantic import Field
from utils import Degrees, Position, Robot
from workmesh.messages import (
    Color,
    Command,
    FillCircleAnimation,
    ServiceCommand,
    StaticAnimation,
)
from workmesh.messages import (
    Service as ServiceName,
)

from robots.robot_state_machine import RobotStateMachine
from workmesh import (
    Service,
    UserUtterance,
    routed_user_utterance_topic,
    service_command_topic,
)


class PhotoBoothBotStateMachine(RobotStateMachine):
    class States(statesman.StateEnum):
        sleep = "Sleep"
        wake_up = "Wake Up"
        find = "Find User"
        track = "Track User"
        focus = "Focus"
        think = "Think"
        look_at = "Look at Position"
        take_picture = "Take Picture"
        abort = "Abort"

    robot_name: str = "PHOTO BOOTH BOT"

    # Tasks
    _finding_user_task: asyncio.Task | None = None

    # Events
    tracking_event: asyncio.Event = Field(default_factory=asyncio.Event)
    user_found_event: asyncio.Event = Field(default_factory=asyncio.Event)
    user_centered_event: asyncio.Event = Field(default_factory=asyncio.Event)

    # Animation UUIDs
    _tracking_animation_uuid: str | None = None
    _thinking_animation_uuid: str | None = None
    _listening_animation_uuid: str | None = None
    _attentive_animation_uuid: str | None = None
    _focus_animation_uuid: str | None = None
    _idle_animation_uuid: str | None = None

    # Keep track if the user was found before
    _was_user_found: bool = False

    # Robot talking state
    is_talking_lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    _is_talking: bool = False

    # Keep track of the body angle
    current_body_angle: Degrees = 0.0
    _user_last_body_angle: Degrees | None = None

    # Configuration
    config: PhotoBoothBotConfig = Field(default_factory=PhotoBoothBotConfig)

    # Light Control
    light_manager: LightManager | None = None

    #########################################################
    # State Machine Management
    #########################################################

    def __init__(self, service: Service, robot_config: PhotoBoothBotConfig):
        super().__init__(service, Robot.RESEARCHER, robot_config.voice_config)

        # Configuration
        self.config = robot_config

        # Control Lights
        self.light_manager = LightManager(service)

    async def _restart_state_machine(self) -> None:
        self._logger.info(f"[{self.robot_name}] Restarting services...")

        # Start with the tracker and STT services off
        await self.service_off(ServiceName.TRACKER)
        await self.service_off(ServiceName.STT)

        # Restart the agent service
        await self.service_restart(ServiceName.AGENT)
        self._service.reset_interaction()  # type: ignore

        # Reset the finding user task
        if self._finding_user_task:
            self._finding_user_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._finding_user_task

            self._finding_user_task = None

        # Restart the compositor service to reset all animations
        await self.service_restart(ServiceName.COMPOSITOR)
        await asyncio.sleep(1)

    #########################################################
    # State events and transitions
    #########################################################

    @statesman.event(source=States.__any__, target=States.abort)  # type: ignore
    async def abort(self) -> None:
        """Abort the bot"""
        self._logger.info(f"[{self.robot_name}] Transition to ABORT")

    @statesman.event(
        source=[None, States.find, States.track, States.abort],  # type: ignore
        target=States.sleep,
        before=_restart_state_machine,
    )
    async def start(self, user_not_found: bool = False) -> None:
        """Start the bot"""
        self._logger.info(f"[{self.robot_name}] Transition to SLEEP")

        if user_not_found:
            await self._user_not_found()

        # Reset state variables
        self.user_found_event.clear()
        self.user_centered_event.clear()
        self._was_user_found = False
        self._user_last_body_angle = None

    async def _user_not_found(self) -> None:
        """User not found"""
        # Look at front
        await self.look_at_position(
            self.current_body_angle, Position(x=0, y=0), duration=2.0
        )

        # Only say sorry if the user was found before
        if self._was_user_found:
            await self.handle_talking("Sorry, I couldn't find you.")

    @statesman.event(source=States.sleep, target=States.wake_up)  # type: ignore
    async def wake_up(self) -> None:
        """Wake up the robot"""
        self._logger.info(f"[{self.robot_name}] Transition to WAKE UP")

    @statesman.event(
        source=[States.wake_up, States.track, States.look_at, States.think],
        target=States.find,
    )  # type: ignore
    async def find_user(self) -> None:
        """Find User"""
        self._logger.info(f"[{self.robot_name}] Transition to FIND")
        self.user_found_event.clear()

    @statesman.event(
        source=States.find,
        target=States.track,
        after=lambda self: self.tracking_event.set(),
    )  # type: ignore
    async def track_user(self) -> None:
        """User Found"""
        self._logger.info(f"[{self.robot_name}] Transition to TRACK")

    async def user_found(self) -> None:
        """User Found"""
        self.user_found_event.set()

    async def user_disappeared(self) -> None:
        """User disappears"""
        self.user_found_event.clear()

        if self.state == self.States.track:
            await self.find_user()

    @statesman.event(
        source=[States.track, States.focus, States.take_picture], target=States.think
    )  # type: ignore
    async def think(self, **kwargs: Any) -> None:
        """Think"""
        self._logger.info(f"[{self.robot_name}] Transition to THINK")

        # Stop the tracker service
        await self.service_off(ServiceName.TRACKER)

    @statesman.event(source=States.track, target=States.take_picture)  # type: ignore
    async def take_picture(self) -> None:
        """Take Picture"""
        self._logger.info(f"[{self.robot_name}] Transition to TAKE_PICTURE")

    @statesman.event(source=[States.think, States.look_at], target=States.focus)  # type: ignore
    async def focus(self) -> None:
        """Focus"""
        self._logger.info(f"[{self.robot_name}] Transition to FOCUS")

    @statesman.event(source=States.think, target=States.look_at)  # type: ignore
    async def look_at(self) -> None:
        """Look at position"""
        self._logger.info(f"[{self.robot_name}] Transition to LOOK_AT")

    async def user_started_speaking(self, message: UserUtterance) -> None:
        """User started speaking"""
        if self.state == self.States.sleep:
            return

        await self.handle_listening()
        await self.start_user_utterance(message)

    async def user_partial_utterance(self, message: UserUtterance) -> None:
        """User partial utterance"""
        if self.state == self.States.sleep:
            return

        # Send partial utterances if robot is not talking
        async with self.is_talking_lock:
            if not self._is_talking:
                await self._service.publish(routed_user_utterance_topic, message)

    async def user_stopped_speaking(self, message: UserUtterance):
        """User stopped speaking"""
        if self._listening_animation_uuid:
            await self.stop_clip(self._listening_animation_uuid)
            self._listening_animation_uuid = None

        async with self.is_talking_lock:
            assert self.light_manager is not None
            await self.light_manager.light_off("listening")
            if not self._is_talking:
                await self._service.publish(routed_user_utterance_topic, message)
                if self.state != self.States.sleep:
                    self._service.create_task(self.think(nod=True))
            else:
                # TODO: Explore hint to user that utterance was ignored; blink blue?
                # See issue 475
                pass

    #########################################################
    # State entry handlers
    #########################################################

    ######### Sleep #########

    @statesman.enter_state(States.sleep)  # type: ignore
    async def on_enter_sleep(self) -> None:
        """Handle entering sleep state"""
        self._logger.info(f"[{self.robot_name}] Entering SLEEP")

        # Move to look at front first if not already
        if abs(self.current_body_angle) > 1e-2:
            self._logger.info(f"[{self.robot_name}] Moving to look at front")
            await self.look_at_position(
                self.current_body_angle, Position(x=0, y=0), duration=2.0
            )

        # Set default blue light
        assert self.light_manager is not None
        await self.light_manager.light_on(
            "default",
            LightEffect(
                priority=3,
                animation=StaticAnimation,
                primary_color=Color.BLUE,
                in_transition_duration=self.config.light_config.quick_transition_duration,
            ),
        )

        # Set light when sleeping
        assert self.light_manager is not None
        await self.light_manager.light_on(
            "sleep",
            LightEffect(
                priority=2,
                animation=StaticAnimation,
                primary_color=Color.GREEN,
                in_transition_duration=self.config.light_config.slow_transition_duration,
            ),
        )

        # Bot is SLEEPING
        self._sleep_animation_uuid = await self.play_clip(
            clip_name="sleep3",
            priority=0,
            opacity=1,
            loop=True,
            blend_in=1,
            blend_out=1,
        )

    @statesman.exit_state(States.sleep)  # type: ignore
    async def on_exit_sleep(self) -> None:
        """Handle exiting sleep state"""

        await self.service_off(ServiceName.STT)

        assert self.light_manager is not None
        await self.light_manager.light_off("sleep")

        self._logger.info(f"[{self.robot_name}] Exiting SLEEP")

    ######### Wake Up #########

    @statesman.enter_state(States.wake_up)  # type: ignore
    async def on_enter_wake_up(self) -> None:
        """Handle entering scan state"""
        self._logger.info(f"[{self.robot_name}] Entering WAKE UP")

        # Bot WAKES UP
        await self.play_and_wait(clip_name="wakeUp1", priority=0, opacity=1)

        # Wake up sentence + look around
        speech_uuid: str | None = None
        wake_up_utterance = self._robot_utterance_manager.get_robot_utterance("wake_up")
        looking_around_animation_uuid = await self.play_clip(
            clip_name="lookAroundShort", priority=0, opacity=1, blend_out=1
        )
        if wake_up_utterance:
            speech_uuid = await self.request_human_speech(wake_up_utterance)
            await self.wait_for_clip(speech_uuid)
        await self.wait_for_clip(looking_around_animation_uuid)

    @statesman.exit_state(States.wake_up)  # type: ignore
    async def on_exit_wake_up(self) -> None:
        """Handle exiting wake up state"""
        self._logger.info(f"[{self.robot_name}] Exiting WAKE UP")

    ######### Find #########

    @statesman.enter_state(States.find)  # type: ignore
    async def on_enter_find(self) -> None:
        """Handle entering find state"""
        self._logger.info(f"[{self.robot_name}] Entering FIND")

        async def finding_user_sequence() -> None:
            # Check the user's last position
            if (
                self._user_last_body_angle is not None
                and abs(self.current_body_angle - self._user_last_body_angle) > 1e-2
            ):
                await self.look_at_position(self.current_body_angle, self._user_last_body_angle)  # noqa: E501 # fmt: skip

            # Move head a bit while finding the user
            await self._idle()

            # Start tracker service
            await self.service_on(ServiceName.TRACKER)
            await asyncio.sleep(1)

            spin_count = 0
            sign = -1 if self.current_body_angle < 0 else 1
            find_range = self.config.animation_config.find_range / 2

            found_user = self.user_found_event.is_set()
            while not found_user and spin_count < 2:
                current_angle = self.current_body_angle

                if (
                    abs(
                        sign * self.config.animation_config.find_angle_step
                        + current_angle
                    )
                    < find_range
                ):
                    angle = (
                        sign * self.config.animation_config.find_angle_step
                        + current_angle
                    )
                else:
                    angle = sign * (find_range - abs(current_angle)) + current_angle
                    spin_count += 1
                    sign = -sign  # Update direction for next step

                await self.look_at_position(current_angle, angle, duration=1.5)
                self._logger.debug(f"[{self.robot_name}] Moved body to {angle}°")
                await asyncio.sleep(self.config.animation_config.find_angle_pause)
                found_user = self.user_found_event.is_set()

            # If the user is not found, go to sleep
            if not found_user:
                self._logger.info(f"[{self.robot_name}] User not found, going to sleep")
                self._service.create_task(self.start(user_not_found=True))
                return

            self._logger.debug(f"[{self.robot_name}] Finding animation completed")
            self._service.create_task(self.track_user())

        self._finding_user_task = self._service.create_task(finding_user_sequence())

    @statesman.exit_state(States.find)  # type: ignore
    async def on_exit_find(self) -> None:
        """Handle exiting find state"""
        self._logger.info(f"[{self.robot_name}] Exiting FIND")

        if self._finding_user_task:
            await self._finding_user_task
            self._finding_user_task = None

        if self._idle_animation_uuid:
            await self.stop_clip(self._idle_animation_uuid)
            self._idle_animation_uuid = None

    ######### Track #########

    @statesman.enter_state(States.track)  # type: ignore
    async def on_enter_track(self) -> None:
        """Handle entering track state"""
        self._logger.info(f"[{self.robot_name}] Entering TRACK")

        # User lost in during the transition to track state, go to find state
        if not self.user_found_event.is_set():
            self._logger.info(
                f"[{self.robot_name}] User not found when entering track state, "
                "going to find state"
            )
            self._service.create_task(self.find_user())
            return

        # Reset center status
        self.user_centered_event.clear()

        # Start TRACKING procedural animation
        if not self.tracking_event.is_set():
            self._tracking_animation_uuid = await self.track_start(
                slow_mode_distance_threshold=self.config.animation_config.tracking.slow_mode_distance_threshold,
                fast_mode_distance_threshold=self.config.animation_config.tracking.fast_mode_distance_threshold,
                slow_speed=self.config.animation_config.tracking.slow_speed,
                fast_speed=self.config.animation_config.tracking.fast_speed,
            )
            await asyncio.sleep(0.5)

            # Turn on STT service if robot is not talking
            async with self.is_talking_lock:
                if not self._is_talking:
                    await self.service_on(ServiceName.STT)

        # Bot is ATTENTIVE
        self._attentive_animation_uuid = await self.play_clip(
            clip_name="attentive",
            priority=1,
            opacity=1,
            loop=True,
            blend_in=0.1,
            blend_out=0.1,
        )

        # Set light when starting to track user
        assert self.light_manager is not None
        await self.light_manager.light_on(
            "attentive",
            LightEffect(
                priority=2,
                animation=StaticAnimation,
                primary_color=Color.GREEN,
                in_transition_duration=self.config.light_config.quick_transition_duration,
            ),
        )

    @statesman.exit_state(States.track)  # type: ignore
    async def on_exit_track(self, stop_tracking: bool = True) -> None:
        """Handle exiting track state"""
        self._logger.info(f"[{self.robot_name}] Exiting TRACK")

        # Stop STT services
        await self.service_off(ServiceName.STT)

        # Make sure that the enter track state is finished
        await self.tracking_event.wait()

        # Stop TRACKING procedural animation
        if self._tracking_animation_uuid and stop_tracking:
            await self.track_stop(self._tracking_animation_uuid)
            self._tracking_animation_uuid = None

        # Save current body angle
        self._user_last_body_angle = self.current_body_angle

        # Stop ATTENTIVE and LISTENING animations
        if self._attentive_animation_uuid:
            await self.stop_clip(self._attentive_animation_uuid)
            self._attentive_animation_uuid = None
        if self._listening_animation_uuid:
            await self.stop_clip(self._listening_animation_uuid)
            self._listening_animation_uuid = None

        # Turn off attentive and listening lights
        assert self.light_manager is not None
        await self.light_manager.light_off("attentive")
        await self.light_manager.light_off("listening")

        # Clear the tracking event
        self.tracking_event.clear()

    ##### Take Picture ######

    @statesman.enter_state(States.take_picture)  # type: ignore
    async def on_enter_take_picture(self) -> None:
        """Handle entering take picture state"""
        self._logger.info(f"[{self.robot_name}] Entering TAKE_PICTURE")

        # Wait for the user to be centered
        self._logger.debug(f"[{self.robot_name}] Waiting for user to be centered")
        try:
            await asyncio.wait_for(
                self.user_centered_event.wait(),
                timeout=self.config.center_user_timeout,
            )
            self._logger.debug(f"[{self.robot_name}] User is centered")
        except TimeoutError:
            self._logger.warning(
                f"[{self.robot_name}] User timed out while centering. "
                "Taking picture anyway."
            )

        # Pause the tracking animation
        if self._tracking_animation_uuid:
            await self.track_pause(self._tracking_animation_uuid)

        # Stop the tracker service
        await self.service_off(ServiceName.TRACKER)

    @statesman.exit_state(States.take_picture)  # type: ignore
    async def on_exit_take_picture(self) -> None:
        """Handle exiting take picture state"""
        self._logger.info(f"[{self.robot_name}] Exiting TAKE_PICTURE")

        # Stop the tracking animation
        if self._tracking_animation_uuid:
            await self.track_stop(self._tracking_animation_uuid, blend_out_duration=0.5)
            self._tracking_animation_uuid = None

    ######### Think #########

    @statesman.enter_state(States.think)  # type: ignore
    async def on_enter_think(self, nod: bool = False) -> None:
        """Handle entering think state"""
        self._logger.info(f"[{self.robot_name}] Entering THINK")

        # Bot acknowledges what the user said by NODDING
        if nod:
            await self.play_and_wait(
                clip_name="nod", priority=0, opacity=1, blend_in=0.25, blend_out=0.25
            )

        # Bot is THINKING
        self._thinking_animation_uuid = await self.play_clip(
            clip_name="intrigued5", priority=0, opacity=1, loop=True, loop_overlap=0.25
        )

    @statesman.exit_state(States.think)  # type: ignore
    async def on_exit_think(self) -> None:
        """Handle exiting think state"""
        self._logger.info(f"[{self.robot_name}] Exiting THINK")

        # Bot stops THINKING
        if self._thinking_animation_uuid:
            await self.stop_clip(self._thinking_animation_uuid)
            self._thinking_animation_uuid = None

    ######### Focus #########

    @statesman.enter_state(States.focus)  # type: ignore
    async def on_enter_focus(self) -> None:
        """Handle entering focus state"""
        self._logger.info(f"[{self.robot_name}] Entering FOCUS")

        # Set light when focusing (doing something)
        assert self.light_manager is not None
        await self.light_manager.light_on(
            "focus",
            LightEffect(
                priority=2,
                animation=FillCircleAnimation,
                primary_color=Color.INTENSE_BLUE,
                secondary_color=Color.BLUE,
                fill_duration=self.config.light_config.focus_duration,
                in_transition_duration=self.config.light_config.quick_transition_duration,
            ),
        )

        # Bot is FOCUSING
        self._focus_animation_uuid = await self.play_clip(
            clip_name="focus", priority=1, opacity=1, loop=True, blend_in=1, blend_out=1
        )

    @statesman.exit_state(States.focus)  # type: ignore
    async def on_exit_focus(self) -> None:
        """Handle exiting focus state"""
        self._logger.info(f"[{self.robot_name}] Exiting FOCUS")

        # Bot stops FOCUSING
        if self._focus_animation_uuid:
            await self.stop_clip(self._focus_animation_uuid)
            self._focus_animation_uuid = None

        # Turn off focus light
        assert self.light_manager is not None
        await self.light_manager.light_off("focus")

    ######### Look at #########

    @statesman.enter_state(States.look_at)  # type: ignore
    async def on_enter_look_at(
        self, position: Position, action_uuid: str | None = None
    ) -> None:
        """Handle entering look at state"""
        self._logger.info(f"[{self.robot_name}] Entering LOOK_AT")

        # Robot LOOKS AT the position
        await self.look_at_position(self.current_body_angle, position, action_uuid)

        # Robot Idles
        await self._idle()

    @statesman.exit_state(States.look_at)  # type: ignore
    async def on_exit_look_at(self) -> None:
        """Handle exiting look at state"""
        self._logger.info(f"[{self.robot_name}] Exiting LOOK_AT")

        if self._idle_animation_uuid:
            await self.stop_clip(self._idle_animation_uuid)
            self._idle_animation_uuid = None

    ######### Abort #########

    @statesman.enter_state(States.abort)  # type: ignore
    async def on_enter_abort(self) -> None:
        """Handle entering abort state"""
        self._logger.info(f"[{self.robot_name}] Entering ABORT")
        self._service.create_task(self.start())  # Go to sleep state

    @statesman.exit_state(States.abort)  # type: ignore
    async def on_exit_abort(self) -> None:
        """Handle exiting abort state"""
        self._logger.info(f"[{self.robot_name}] Exiting ABORT")

    #########################################################
    # Global event handlers (not tied to specific states)
    #########################################################

    async def handle_talking(
        self,
        script: str,
        action_uuid: str | None = None,
        light_on: bool = True,
        look_direction: str | None = None,
    ) -> None:
        """Handle talking event from any state"""
        self._logger.info(f"[{self.robot_name}] Talking...")
        self._logger.info(f"[{self.robot_name}] Script: '{script}'")

        # Set light when talking
        if light_on:
            assert self.light_manager is not None
            await self.light_manager.light_on(
                "talking",
                LightEffect(
                    priority=0,
                    animation=StaticAnimation,
                    primary_color=Color.INTENSE_BLUE,
                    in_transition_duration=self.config.light_config.quick_transition_duration,
                ),
            )

        async with self.is_talking_lock:
            self._is_talking = True
            if not self.config.enable_listening_while_speaking:
                # Turn off STT service
                await self.service_off(ServiceName.STT)

        try:
            # Send TTS request message
            speech_uuid = await self.request_human_speech(script, action_uuid)

            # Wait for the speech to start to sync with the talking animation
            speech_success = await self.wait_for_clip_started(speech_uuid)

            # Bot is TALKING
            # Loop if the speech was successful,
            # otherwise just run it once to fake talking
            if look_direction in ["left", "right"]:
                if look_direction == "left":
                    talking_uuid = await self.play_clip(
                        clip_name="talkingLeftShoulder",
                        priority=0,
                        opacity=1,
                        loop=speech_success,
                    )
                else:
                    talking_uuid = await self.play_clip(
                        clip_name="talkingRightShoulder",
                        priority=0,
                        opacity=1,
                        loop=speech_success,
                    )
            else:
                talking_uuid = await self.play_clip(
                    clip_name="talking", priority=0, opacity=1, loop=speech_success
                )

            # Wait until the speech is done playing and stop the talking animation
            await self.wait_for_clip(speech_uuid)
            await self.stop_clip(talking_uuid)
        finally:
            # Turn off the talking light
            assert self.light_manager is not None
            await self.light_manager.light_off("talking")

            async with self.is_talking_lock:
                self._is_talking = False
                # Turn on the STT service if still tracking the user
                if self.state == self.States.track:
                    await self.service_on(ServiceName.STT)

    async def handle_listening(self) -> None:
        """Handle listening event from any state"""
        self._logger.info(f"[{self.robot_name}] Listening...")

        # Set light when actually listening (user speaking)
        assert self.light_manager is not None
        await self.light_manager.light_on(
            "listening",
            LightEffect(
                priority=1,
                animation=StaticAnimation,
                primary_color=Color.INTENSE_GREEN,
                in_transition_duration=self.config.light_config.quick_transition_duration,
            ),
        )

        if self.config.enable_listen_animation:
            self._listening_animation_uuid = await self.play_clip(
                clip_name="listen1",
                priority=1,
                opacity=1,
                loop=True,
                blend_in=0.25,
                blend_out=0.25,
            )

    async def handle_look_at(
        self, position: Position, action_uuid: str | None = None
    ) -> None:
        """Handle self-transition within look_at state"""
        self._logger.info(f"[{self.robot_name}] Looking at action")

        # Allows self-transition within look_at state
        if self.state == self.States.look_at:
            await self.on_enter_look_at(position, action_uuid)
        else:
            self._service.create_task(self.safe_trigger_event("look_at", position, action_uuid))  # noqa: E501 # fmt: skip

    async def handle_prepare_for_picture(self) -> None:
        """Handle preparing for picture event from any state"""
        self._logger.info(f"[{self.robot_name}] Preparing for picture...")

        # Set light when preparing for taking a picture
        assert self.light_manager is not None
        await self.light_manager.light_on(
            "picture_preparation",
            LightEffect(
                priority=2,
                animation=StaticAnimation,
                primary_color=Color.GRAY,
                in_transition_duration=self.config.light_config.picture_preparation_duration,
            ),
        )

        # Play the picture preparation animation
        await self.play_and_wait(
            clip_name="picturePreparation", priority=0, opacity=1, blend_out=0.0
        )

    async def handle_take_picture(self) -> None:
        """Handle taking picture event from any state"""
        self._logger.info(f"[{self.robot_name}] Taking picture...")

        # Set light when taking a picture
        hold_duration = 0.1
        assert self.light_manager is not None
        await self.light_manager.light_blink(
            LightEffect(
                animation=StaticAnimation,
                primary_color=Color.WHITE,
                in_transition_duration=0,
            ),
            duration=hold_duration,
        )

        # Play the picture taking animation
        picture_uuid = await self.play_clip(
            clip_name="takePicture", priority=0, opacity=1, blend_in=0.0
        )

        # Wait for the picture taking animation to finish
        await self.wait_for_clip(picture_uuid)

        # Turn off picture light
        assert self.light_manager is not None
        await self.light_manager.light_off("picture_preparation")

        self._logger.info(f"[{self.robot_name}] Picture taken!")

    #########################################################
    # Aux animations
    #########################################################

    async def _idle(self) -> None:
        self._idle_animation_uuid = await self.play_clip(
            clip_name="idle3old", priority=1, opacity=1, loop=True
        )

    #########################################################
    # Auxiliary functions
    #########################################################

    async def service_on(self, service: ServiceName) -> None:
        message = ServiceCommand(command=Command.ENABLE, target_service=service)
        await self._service.publish(service_command_topic, message)
        self._logger.debug(f"{ServiceName.Name(service)} service started")

    async def service_off(self, service: ServiceName) -> None:
        message = ServiceCommand(command=Command.DISABLE, target_service=service)
        await self._service.publish(service_command_topic, message)
        self._logger.debug(f"{ServiceName.Name(service)} service stopped")

    async def service_restart(self, service: ServiceName) -> None:
        message = ServiceCommand(command=Command.RESTART, target_service=service)
        await self._service.publish(service_command_topic, message)
        self._logger.debug(f"{ServiceName.Name(service)} service restarted")

    async def start_user_utterance(self, message: UserUtterance) -> None:
        """Start a user utterance"""

        # Only send a started message if the robot is not talking
        async with self.is_talking_lock:
            if not self._is_talking:
                await self._service.publish(routed_user_utterance_topic, message)

    #########################################################
    # Remote control event handlers
    #########################################################

    async def trigger(self, command: str) -> None:
        """Handle incoming events"""

        if command == "Start":
            await self.safe_trigger_event("wake_up")
        elif command == "UserAppeared":
            await self.safe_trigger_event("track_user")
        elif command == "UserDisappeared":
            await self.safe_trigger_event("find_user")
        elif command == "Listen":
            await self.handle_listening()
        elif command == "Think":
            await self.safe_trigger_event("think")
        elif command == "Focus":
            await self.safe_trigger_event("focus")
        elif command == "HumanInTheLoop":
            await self.safe_trigger_event("find_user")
            await self.tracking_event.wait()
            await self.handle_talking("Hey! I need your help!")
        elif command == "TakePicture":
            await self.safe_trigger_event("find_user")
            await self.tracking_event.wait()
            await self.safe_trigger_event("take_picture", stop_tracking=False)
            await self.handle_prepare_for_picture()
            await self.handle_take_picture()
            await self.safe_trigger_event("think")
        elif command.lower() == "abort":
            # Handled in the interaction manager
            pass
        else:
            self._logger.warning(f"Unknown command: {command}")
