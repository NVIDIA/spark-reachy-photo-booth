# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import random
import time
from enum import Enum

from configuration import InteractionManagerConfig, PhotoBoothBotConfig
from event_manager import EventManager
from robot_utterance_manager import RobotUtteranceManager
from robots.photo_booth_bot import PhotoBoothBotStateMachine
from utils import KeyPosition, Position
from workmesh.config import load_config
from workmesh.messages import (
    ClipStatus,
    PresenceStatus,
    RemoteControlCommand,
    RobotFrame,
    ToolStatus,
    TrackingStatus,
    UserState,
    UserTrackingStatus,
    UserUtterance,
    UserUtteranceStatus,
)
from workmesh.messages import (
    Service as ServiceName,
)
from workmesh.service import Service, subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    clip_status_topic,
    protobuf_map_to_dict,
    remote_control_command_topic,
    robot_frame_topic,
    tool_status_topic,
    user_state_topic,
    user_tracking_status_topic,
    user_utterance_topic,
)


class ControlMessage(Enum):
    STOP_SERVICE = "stop_service"
    ABORT_INTERACTION = "abort_interaction"


class InteractionManagerService(Service):
    def __init__(self, config: InteractionManagerConfig) -> None:
        super().__init__(config)
        self.config = config

        # Create event manager
        self.event_manager: EventManager = EventManager(self.logger)

        # Create script manager
        self.robot_utterance_manager: RobotUtteranceManager = RobotUtteranceManager(
            utterances_path=self.config.robot_utterances_path, logger=self.logger
        )

        # Create and start robots state machines
        robot_config = self.config.robot_config.get("photo_booth_bot", None)
        if robot_config is None:
            raise ValueError("Photo booth bot config not found")
        elif not isinstance(robot_config, PhotoBoothBotConfig):
            raise ValueError("Photo booth bot config is not a PhotoBoothBotConfig")

        self.photo_bot = PhotoBoothBotStateMachine(
            service=self, robot_config=robot_config
        )
        self.photo_bot.set_global_clip_volume(self.config.global_clip_volume)

        self.create_task(self.start_robots())

        # Queue tool status messages for sequential processing
        self.tool_status_queue: asyncio.Queue[ToolStatus | ControlMessage] = (
            asyncio.Queue()
        )
        self.running_tools: dict[str, ToolStatus.Status] = {}
        self.stop_event: asyncio.Event = asyncio.Event()
        self.create_task(self.process_tool_status())

        # Keep track of what the robot is looking at
        self.last_look_at: KeyPosition = KeyPosition.USER

        # Manage utterance uuids
        self.last_utterance_uuid: str | None = None
        self.ask_human_uuid: str | None = None

    async def start_robots(self) -> None:
        self.logger.info("Starting robots...")
        await self.photo_bot.start()

    # Remote control updates
    @subscribe(remote_control_command_topic)
    async def on_remote_control(self, message: RemoteControlCommand) -> None:
        self.logger.info(f'Received remote control command: "{message.command}"')
        await self.photo_bot.trigger(message.command)

        if message.command.lower() == "abort":
            self.reset_interaction()
            self.tool_status_queue.put_nowait(ControlMessage.ABORT_INTERACTION)

    # Clip status updates
    @subscribe(clip_status_topic)
    async def on_clip_status(self, message: ClipStatus) -> None:
        await self.event_manager.add_status_event(message)

    # USER DETECTION
    @subscribe(user_state_topic)
    async def on_user_state(self, message: UserState) -> None:
        if message.status == PresenceStatus.USER_APPEARED:
            self.logger.info("User appeared")
            self.last_look_at = KeyPosition.USER
            await self.photo_bot.user_found()
        elif message.status == PresenceStatus.USER_DISAPPEARED:
            self.logger.info("User disappeared")
            await self.photo_bot.user_disappeared()

    # USER TRACKING STATUS
    @subscribe(user_tracking_status_topic)
    async def on_user_tracking_status(self, message: UserTrackingStatus) -> None:
        if message.status == TrackingStatus.USER_CENTERED:
            self.logger.debug("User centered")
            self.photo_bot.user_centered_event.set()
        elif message.status == TrackingStatus.USER_NOT_CENTERED:
            self.logger.debug("User not centered")
            self.photo_bot.user_centered_event.clear()

    # VOICE DETECTION
    @subscribe(user_utterance_topic)
    async def on_user_utterance(self, message: UserUtterance) -> None:
        if message.status == UserUtteranceStatus.USER_UTTERANCE_STARTED:
            self.logger.info("User input detected")
            self.last_utterance_uuid = message.action_uuid

            if self.ask_human_uuid:
                message.action_uuid = self.ask_human_uuid
            await self.photo_bot.user_started_speaking(message)
        elif (
            self.last_utterance_uuid == message.action_uuid
            and message.status == UserUtteranceStatus.USER_UTTERANCE_UPDATED
        ):
            self.logger.debug(f"User input updated: '{message.text}'")
            await self.photo_bot.user_partial_utterance(message)
        elif (
            self.last_utterance_uuid == message.action_uuid
            and message.status == UserUtteranceStatus.USER_UTTERANCE_FINISHED
        ):
            self.logger.info(f"User input finished: '{message.text}'")
            self.last_utterance_uuid = None

            if self.ask_human_uuid:
                message.action_uuid = self.ask_human_uuid
            await self.photo_bot.user_stopped_speaking(message)

    # ROBOT FRAME
    @subscribe(robot_frame_topic)
    async def on_robot_frame(self, message: RobotFrame) -> None:
        self.photo_bot.current_body_angle = message.frame.body_angle

    # TOOL CALLS
    @subscribe(tool_status_topic)
    async def on_tool_status(self, message: ToolStatus) -> None:
        self.logger.info(
            "Tool status received. "
            + f"Action UUID: {message.action_uuid} "
            + f"Status: {message.Status.Name(message.status)}"
        )

        # Ignore tool call processed messages as they're triggered
        # by the interaction manager itself
        if message.status == ToolStatus.Status.TOOL_CALL_PROCESSED:
            return

        # Add message to queue
        await self.tool_status_queue.put(message)

    async def process_tool_status(self) -> None:
        self.logger.info("Tool status processing started")

        comment_task: asyncio.Task | None = None
        comment_stop_event: asyncio.Event = asyncio.Event()

        # Cancel the comment task
        async def cleanup_comment_task() -> None:
            nonlocal comment_task, comment_stop_event
            if comment_task is not None:
                comment_stop_event.set()
                await comment_task
                comment_task = None
                comment_stop_event.clear()

        while True:
            message = await self.tool_status_queue.get()

            if isinstance(message, ControlMessage):
                await cleanup_comment_task()

                if message == ControlMessage.ABORT_INTERACTION:
                    self.logger.info("Safety button pressed. Stopping interaction")
                    await self.photo_bot.abort()
                    continue
                elif message == ControlMessage.STOP_SERVICE:
                    self.stop_event.set()
                    break

            self.logger.debug(f"Processing tool status: {message}")

            uuid = message.action_uuid
            tool_name = message.name
            tool_input = protobuf_map_to_dict(message.input)
            self.logger.debug(f"Tool data: {tool_input}")
            if not tool_name:
                self.logger.warning("Tool name not found in tool message")
                continue

            # Tool names
            screen_tools = self.config.tool_names.get("screen", [])
            human_tools = self.config.tool_names.get("human", [])
            image_generation_tools = self.config.tool_names.get("image_generation", [])
            take_picture_tools = self.config.tool_names.get("taking_picture", [])
            end_tools = self.config.tool_names.get("end", [])
            start_tools = self.config.tool_names.get("start", [])
            # Room mapping
            screen_position = self.config.room_mapping.get(
                "screen", Position(x=-1, y=-1)
            )

            if message.status == ToolStatus.Status.TOOL_CALL_STARTED:
                if tool_name in self.running_tools and tool_name not in human_tools:
                    self.logger.warning(
                        f"Tool '{tool_name}' already started. "
                        f"Don't execute actions again."
                    )
                    await self._send_tool_processed_message(message)
                    continue

                self.running_tools[tool_name] = message.status
                self.logger.info(f"Tool '{tool_name}' started")

                # Start interaction
                if tool_name in start_tools:
                    self.logger.info("Starting interaction")
                    await self.photo_bot.service_on(ServiceName.STT)

                elif tool_name in screen_tools:
                    # Move to look at the screen
                    await self._move_to_look_at(
                        screen_position, KeyPosition.SCREEN, uuid
                    )

                    # Robot talks to the user
                    await self._speak(tool_name, message.status)

                # Focus during image generation
                elif tool_name in image_generation_tools:
                    self.logger.info("Focusing during image generation")

                    # Move to look at the screen
                    await self._move_to_look_at(
                        screen_position, KeyPosition.SCREEN, uuid
                    )

                    await self.photo_bot.safe_trigger_event("focus")

                    # Get image comments + other utterances
                    image_comments = tool_input.get("comments_to_steps") or []
                    robot_response = self._create_comments(image_comments)

                    if robot_response:

                        async def make_comments(robot_response: list[str]) -> None:
                            self.logger.info("Making image generation comments")

                            # Agent comments are a list of strings
                            for comment in robot_response:
                                if comment_stop_event.is_set():
                                    break

                                # Allow some time to go to the focus animation
                                await asyncio.sleep(
                                    random.uniform(
                                        self.config.time_between_comments.min,
                                        self.config.time_between_comments.max,
                                    )
                                )

                                # Make comments
                                await self.photo_bot.handle_talking(
                                    comment,
                                    light_on=False,
                                    look_direction=self.config.comment_look_direction,
                                )

                            self.logger.info("Comments made")

                        comment_task = asyncio.create_task(make_comments(robot_response))  # noqa: E501 # fmt: skip

                # Look at user and ask for help
                elif tool_name in human_tools:
                    self.ask_human_uuid = uuid

                    # Ignore if there is no robot message
                    robot_response = tool_input.get("prompt")
                    if not robot_response:
                        self.logger.warning(
                            "Robot response not found in tool data for "
                            f"'{tool_name}' tool"
                        )

                    # Robot looks for user before asking for help
                    self.logger.info("Robot looks for the user")
                    self.last_look_at = KeyPosition.USER

                    # If the user is found, the robot is tracking
                    # and there is a robot response, ask for help
                    if await self._find_user() and robot_response:
                        self.logger.info("Robot talks to the user")
                        await self.photo_bot.handle_talking(robot_response, uuid)  # noqa: E501 # fmt: skip

                    # After the robot finishes talking, send a started message
                    # if the user started speaking while the robot was still talking
                    if self.last_utterance_uuid:
                        started_message = UserUtterance(
                            action_uuid=self.ask_human_uuid,
                            status=UserUtteranceStatus.USER_UTTERANCE_STARTED,
                            timestamp=int(time.time() * 1000),
                        )
                        await self.photo_bot.start_user_utterance(started_message)

                # Taking picture
                elif tool_name in take_picture_tools:
                    # Move to look at the user
                    if self.photo_bot.state in [
                        PhotoBoothBotStateMachine.States.think,
                        PhotoBoothBotStateMachine.States.look_at,
                    ]:
                        # Robot looks for user
                        self.logger.info("Robot looks for the user")
                        self.last_look_at = KeyPosition.USER

                        # If the user is found, the robot is tracking
                        if await self._find_user():
                            # Transition to take the picture
                            await self.photo_bot.safe_trigger_event(
                                "take_picture", stop_tracking=False
                            )

                            # Robot talks to the user
                            await self._speak(tool_name, message.status)

                            # Prepare for picture
                            await self.photo_bot.handle_prepare_for_picture()
                            await self._send_tool_processed_message(message)

                            # Take picture animation
                            await self.photo_bot.handle_take_picture()

                            # Move to think state after the picture is taken
                            await self.photo_bot.safe_trigger_event("think")

                            # We already sent the tool processed message,
                            # so we continue
                            continue

                # End the interaction
                elif tool_name in end_tools:
                    # Robot looks for user
                    self.logger.info("Robot looks for the user")
                    self.last_look_at = KeyPosition.USER

                    # If the user is found, the robot is tracking
                    if await self._find_user():
                        robot_response = tool_input.get("message") or None

                        if not robot_response:
                            self.logger.warning(
                                "Robot response not found in tool data for "
                                f"'{tool_name}' tool"
                            )
                        else:
                            await self.photo_bot.handle_talking(robot_response)

                    # End interaction animation
                    await self.photo_bot.safe_trigger_event("start")

                # Send tool processed message
                await self._send_tool_processed_message(message)

            elif message.status == ToolStatus.Status.TOOL_CALL_COMPLETED:
                # Remove tool from running tools
                if tool_name not in self.running_tools:
                    self.logger.warning(
                        f"Tool '{tool_name}' has not started so it cannot be completed"
                    )
                    continue

                self.logger.info(f"Tool '{tool_name}' completed")
                self.running_tools.pop(tool_name)

                if tool_name in start_tools:
                    self.logger.info("Interaction started")

                    # Wake up the robot
                    await self.photo_bot.safe_trigger_event("wake_up")

                    # Find the user
                    await self.photo_bot.safe_trigger_event("find_user")

                    # Wait for tracking to be ready or the robot goes to sleep
                    # before continuing to the next tool
                    await self._wait_for_tracking_or_sleep()

                # Image generation completed
                elif tool_name in image_generation_tools:
                    self.logger.info("Image generation completed")

                    # Cancel the comment task
                    await cleanup_comment_task()

                    # Robot talks to the user
                    await self._speak(tool_name, message.status)

                    # Go back to thinking
                    await self.photo_bot.safe_trigger_event("think")

                # Taking picture completed
                elif tool_name in take_picture_tools:
                    self.logger.info("Taking picture completed")

                    # Robot talks to the user
                    await self._speak(tool_name, message.status)

            elif message.status == ToolStatus.Status.TOOL_CALL_FAILED:
                # Ignore if the tool has not started
                if tool_name not in self.running_tools:
                    self.logger.warning(
                        f"Tool '{tool_name}' has not started so it cannot be failed"
                    )
                    continue

                if self.running_tools[tool_name] == ToolStatus.Status.TOOL_CALL_STARTED:  # noqa: E501 # fmt: skip
                    self.running_tools[tool_name] = message.status
                    self.logger.info(f"Tool '{tool_name}' failed")
                else:
                    self.logger.warning(
                        f"Tool '{tool_name}' has already failed. "
                        f"Stop processing this tool."
                    )
                    self.running_tools.pop(tool_name)

                    # Move the robot back up if focusing
                    if tool_name in image_generation_tools:
                        self.logger.info("Image generation failed. Moving robot back up.")  # noqa: E501 # fmt: skip

                        # Stop image generation comments
                        await cleanup_comment_task()

                        # Go back to thinking
                        await self.photo_bot.safe_trigger_event("think")

        self.logger.info("Tool status processing stopped")

    async def _send_tool_processed_message(self, original_message: ToolStatus) -> None:
        message = ToolStatus(
            action_uuid=original_message.action_uuid,
            status=ToolStatus.Status.TOOL_CALL_PROCESSED,
            robot_id=original_message.robot_id,
            timestamp=int(time.time() * 1000),
            name=original_message.name,
            input=original_message.input,
            response=original_message.response,
        )
        await self.publish(tool_status_topic, message)

    async def _wait_for_tracking_or_sleep(self) -> str:
        """Wait for tracking event to be set or robot to go to sleep.

        This prevents getting stuck waiting for tracking if the robot goes to sleep.
        """
        self.logger.debug("Waiting for tracking event to be set or sleep")
        tracking_task = asyncio.create_task(
            self.photo_bot.tracking_event.wait(), name="robot_is_tracking"
        )
        sleep_task = asyncio.create_task(
            self._wait_for_robot_to_go_to_sleep(), name="robot_went_to_sleep"
        )

        done, pending = await asyncio.wait(
            {tracking_task, sleep_task}, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel any pending tasks
        for task in pending:
            task.cancel()

        task_name = done.pop().get_name()
        self.logger.debug(f"Task Finished: {task_name}")
        return task_name

    async def _wait_for_robot_to_go_to_sleep(self) -> None:
        """Wait until the robot transitions to the 'sleep' state."""
        while self.photo_bot.state != self.photo_bot.States.sleep:
            await asyncio.sleep(0.1)

    async def _move_to_look_at(
        self, position: Position, key_position: KeyPosition, uuid: str
    ) -> None:
        if self.last_look_at != key_position:
            self.logger.info(f"Moving to look at {key_position.name}")
            await self.photo_bot.handle_look_at(position, uuid)
            self.last_look_at = key_position
            await self.event_manager.wait_for_clip_finished(uuid)

    def _create_comments(self, image_comments: list[str]) -> list[str]:
        # Retrieve the specific comments
        extra_comments: list[str | None] = [
            self.robot_utterance_manager.get_robot_utterance(
                "demo_information", "performance"
            ),
            self.robot_utterance_manager.get_robot_utterance(
                "demo_information", "facts"
            ),
        ]

        # Insert extra comments in between image comments
        result: list[str | None] = []

        for comment in image_comments:
            result.append(comment)
            if len(extra_comments) > 0:
                result.append(extra_comments.pop(0))

        if len(extra_comments) > 0:
            result.extend(extra_comments)

        # Add QR code comment
        result.append(
            self.robot_utterance_manager.get_robot_utterance("qr_code_preparation")
        )

        # Filter None values
        return [v for v in result if v is not None]

    async def _speak(self, tool_name: str, status: ToolStatus.Status) -> None:
        status_name = ToolStatus.Status.Name(status).split("_")[-1].lower()
        utterance = self.robot_utterance_manager.get_robot_utterance(
            tool_name, status_name
        )
        if utterance:
            await self.photo_bot.handle_talking(utterance)

    async def _find_user(self) -> bool:
        """Find the user and return True if the user is found, False otherwise."""

        if self.photo_bot.state != PhotoBoothBotStateMachine.States.track:
            # Wait until the robot starts tracking them
            await self.photo_bot.safe_trigger_event("find_user")

        # Wait for tracking to be ready, but cancel if user is lost
        task_name = await self._wait_for_tracking_or_sleep()
        return task_name == "robot_is_tracking"

    def reset_interaction(self) -> None:
        self.running_tools.clear()
        self.last_look_at = KeyPosition.USER
        self.last_utterance_uuid = None
        self.ask_human_uuid = None

        while not self.tool_status_queue.empty():
            self.tool_status_queue.get_nowait()

    async def stop(self) -> None:
        # Stop tool status processing
        await self.tool_status_queue.put(ControlMessage.STOP_SERVICE)
        await self.stop_event.wait()

        await super().stop()


async def main() -> None:
    config = load_config(InteractionManagerConfig)
    await ServiceExecutor([InteractionManagerService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
