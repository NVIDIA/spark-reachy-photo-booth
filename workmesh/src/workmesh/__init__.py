# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from .config import BaseConfig, ConsumerConfig, ProducerConfig, load_config
from .messages import (
    ChangeVolume,
    ClipData,
    ClipStatus,
    FilePublish,
    Frame,
    HumanSpeechRequest,
    ImageGeneration,
    LightCommand,
    Ping,
    PlayClip,
    Pong,
    ProceduralClip,
    ProcessedDoA,
    RealtimeDoA,
    RemoteControlCommand,
    RobotFrame,
    RobotSpeechRequest,
    ServiceCommand,
    StopClip,
    ToolStatus,
    UserDetection,
    UserState,
    UserTrackingStatus,
    UserUtterance,
)
from .protobuf_utils import (
    dict_to_protobuf_map,
    protobuf_map_to_dict,
    protobuf_value_to_python,
    python_to_protobuf_value,
)
from .service import Service, Topic, produces, subscribe
from .service_executor import ServiceExecutor

# Note: Please add the topics also in the `redpanda-console` config in
#      `docker-compose.yaml` under `protobuf` so that the console can
#       deserialize the messages.
# Note: Define your topics in alphabetical order.
animation_frame_out_topic = Topic("animation_frame_out", RobotFrame)
camera_frame_topic = Topic("camera_frame", Frame)
change_volume_topic = Topic("change_volume", ChangeVolume)
clip_data_topic = Topic("clip_data", ClipData)
clip_status_topic = Topic("clip_status", ClipStatus)
realtime_doa_topic = Topic("realtime_doa", RealtimeDoA)
processed_doa_topic = Topic("processed_doa", ProcessedDoA)
file_publish_topic = Topic("file_publish", FilePublish)
human_speech_request_topic = Topic("human_speech_request", HumanSpeechRequest)
image_generation_topic = Topic("image_generation", ImageGeneration)
ping_topic = Topic("ping", Ping)
play_clip_topic = Topic("play_clip", PlayClip)
pong_topic = Topic("pong", Pong)
procedural_clip_topic = Topic("procedural_clip", ProceduralClip)
remote_control_command_topic = Topic("remote_control_command", RemoteControlCommand)
robot_frame_topic = Topic("robot_frame", RobotFrame)
robot_speech_request_topic = Topic("robot_speech_request", RobotSpeechRequest)
routed_user_utterance_topic = Topic("routed_user_utterance", UserUtterance)
stop_clip_topic = Topic("stop_clip", StopClip)
user_state_topic = Topic("user_state", UserState)
user_detection_topic = Topic("user_detection", UserDetection)
user_tracking_status_topic = Topic("user_tracking_status", UserTrackingStatus)
user_utterance_topic = Topic("user_utterance", UserUtterance)
service_command_topic = Topic("service_command", ServiceCommand)
tool_status_topic = Topic("tool_status", ToolStatus)
light_command_topic = Topic("light_command", LightCommand)

__all__: list[str] = [
    # Topics
    "Topic",
    "animation_frame_out_topic",
    "camera_frame_topic",
    "change_volume_topic",
    "clip_data_topic",
    "clip_status_topic",
    "human_speech_request_topic",
    "image_generation_topic",
    "ping_topic",
    "play_clip_topic",
    "pong_topic",
    "procedural_clip_topic",
    "processed_doa_topic",
    "realtime_doa_topic",
    "remote_control_command_topic",
    "robot_frame_topic",
    "robot_speech_request_topic",
    "routed_user_utterance_topic",
    "service_command_topic",
    "stop_clip_topic",
    "tool_status_topic",
    "user_detection_topic",
    "user_state_topic",
    "user_tracking_status_topic",
    "user_utterance_topic",
    "light_command_topic",
    # Functions
    "dict_to_protobuf_map",
    "load_config",
    "produces",
    "protobuf_map_to_dict",
    "protobuf_value_to_python",
    "python_to_protobuf_value",
    "subscribe",
    # Service/Config
    "BaseConfig",
    "ConsumerConfig",
    "ProducerConfig",
    "ServiceExecutor",
    "Service",
]
