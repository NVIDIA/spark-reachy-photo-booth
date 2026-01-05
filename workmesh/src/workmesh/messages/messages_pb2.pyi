from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ImageEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    IMAGE_ENCODING_UNSPECIFIED: _ClassVar[ImageEncoding]
    JPEG: _ClassVar[ImageEncoding]
    PNG: _ClassVar[ImageEncoding]
    BMP: _ClassVar[ImageEncoding]
    RAW: _ClassVar[ImageEncoding]

class GenerationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TXT2IMG: _ClassVar[GenerationType]
    IMG2IMG: _ClassVar[GenerationType]
    FAKE2IMG: _ClassVar[GenerationType]

class Color(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSPARENT: _ClassVar[Color]
    GRAY: _ClassVar[Color]
    WHITE: _ClassVar[Color]
    BLUE: _ClassVar[Color]
    INTENSE_BLUE: _ClassVar[Color]
    GREEN: _ClassVar[Color]
    INTENSE_GREEN: _ClassVar[Color]
    RED: _ClassVar[Color]
    INTENSE_RED: _ClassVar[Color]

class ProceduralType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LOOK_AT: _ClassVar[ProceduralType]
    TRACK: _ClassVar[ProceduralType]

class ProceduralState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    START: _ClassVar[ProceduralState]
    UPDATE: _ClassVar[ProceduralState]
    PAUSE: _ClassVar[ProceduralState]
    STOP: _ClassVar[ProceduralState]

class Robot(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    RESEARCHER: _ClassVar[Robot]

class Service(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRACKER: _ClassVar[Service]
    AGENT: _ClassVar[Service]
    STT: _ClassVar[Service]
    COMPOSITOR: _ClassVar[Service]

class Command(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    ENABLE: _ClassVar[Command]
    DISABLE: _ClassVar[Command]
    RESTART: _ClassVar[Command]

class PresenceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    USER_APPEARED: _ClassVar[PresenceStatus]
    USER_DISAPPEARED: _ClassVar[PresenceStatus]

class TrackingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    USER_CENTERED: _ClassVar[TrackingStatus]
    USER_NOT_CENTERED: _ClassVar[TrackingStatus]

class UserUtteranceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    USER_UTTERANCE_STARTED: _ClassVar[UserUtteranceStatus]
    USER_UTTERANCE_UPDATED: _ClassVar[UserUtteranceStatus]
    USER_UTTERANCE_FINISHED: _ClassVar[UserUtteranceStatus]
IMAGE_ENCODING_UNSPECIFIED: ImageEncoding
JPEG: ImageEncoding
PNG: ImageEncoding
BMP: ImageEncoding
RAW: ImageEncoding
TXT2IMG: GenerationType
IMG2IMG: GenerationType
FAKE2IMG: GenerationType
TRANSPARENT: Color
GRAY: Color
WHITE: Color
BLUE: Color
INTENSE_BLUE: Color
GREEN: Color
INTENSE_GREEN: Color
RED: Color
INTENSE_RED: Color
LOOK_AT: ProceduralType
TRACK: ProceduralType
START: ProceduralState
UPDATE: ProceduralState
PAUSE: ProceduralState
STOP: ProceduralState
RESEARCHER: Robot
TRACKER: Service
AGENT: Service
STT: Service
COMPOSITOR: Service
ENABLE: Command
DISABLE: Command
RESTART: Command
USER_APPEARED: PresenceStatus
USER_DISAPPEARED: PresenceStatus
USER_CENTERED: TrackingStatus
USER_NOT_CENTERED: TrackingStatus
USER_UTTERANCE_STARTED: UserUtteranceStatus
USER_UTTERANCE_UPDATED: UserUtteranceStatus
USER_UTTERANCE_FINISHED: UserUtteranceStatus

class Animation(_message.Message):
    __slots__ = ("frame_rate", "data")
    FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    frame_rate: int
    data: AnimationData
    def __init__(self, frame_rate: _Optional[int] = ..., data: _Optional[_Union[AnimationData, _Mapping]] = ...) -> None: ...

class AnimationData(_message.Message):
    __slots__ = ("frames",)
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    frames: _containers.RepeatedCompositeFieldContainer[AnimationFrame]
    def __init__(self, frames: _Optional[_Iterable[_Union[AnimationFrame, _Mapping]]] = ...) -> None: ...

class AnimationFrame(_message.Message):
    __slots__ = ("body_angle", "r_antenna_angle", "l_antenna_angle", "head_position_x", "head_position_y", "head_position_z", "head_rotation")
    BODY_ANGLE_FIELD_NUMBER: _ClassVar[int]
    R_ANTENNA_ANGLE_FIELD_NUMBER: _ClassVar[int]
    L_ANTENNA_ANGLE_FIELD_NUMBER: _ClassVar[int]
    HEAD_POSITION_X_FIELD_NUMBER: _ClassVar[int]
    HEAD_POSITION_Y_FIELD_NUMBER: _ClassVar[int]
    HEAD_POSITION_Z_FIELD_NUMBER: _ClassVar[int]
    HEAD_ROTATION_FIELD_NUMBER: _ClassVar[int]
    body_angle: float
    r_antenna_angle: float
    l_antenna_angle: float
    head_position_x: float
    head_position_y: float
    head_position_z: float
    head_rotation: EulerAngle
    def __init__(self, body_angle: _Optional[float] = ..., r_antenna_angle: _Optional[float] = ..., l_antenna_angle: _Optional[float] = ..., head_position_x: _Optional[float] = ..., head_position_y: _Optional[float] = ..., head_position_z: _Optional[float] = ..., head_rotation: _Optional[_Union[EulerAngle, _Mapping]] = ...) -> None: ...

class EulerAngle(_message.Message):
    __slots__ = ("roll", "pitch", "yaw")
    ROLL_FIELD_NUMBER: _ClassVar[int]
    PITCH_FIELD_NUMBER: _ClassVar[int]
    YAW_FIELD_NUMBER: _ClassVar[int]
    roll: float
    pitch: float
    yaw: float
    def __init__(self, roll: _Optional[float] = ..., pitch: _Optional[float] = ..., yaw: _Optional[float] = ...) -> None: ...

class Audio(_message.Message):
    __slots__ = ("sample_rate", "bits_per_sample", "channel_count", "audio_buffer")
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    BITS_PER_SAMPLE_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_COUNT_FIELD_NUMBER: _ClassVar[int]
    AUDIO_BUFFER_FIELD_NUMBER: _ClassVar[int]
    sample_rate: int
    bits_per_sample: int
    channel_count: int
    audio_buffer: bytes
    def __init__(self, sample_rate: _Optional[int] = ..., bits_per_sample: _Optional[int] = ..., channel_count: _Optional[int] = ..., audio_buffer: _Optional[bytes] = ...) -> None: ...

class PlayClip(_message.Message):
    __slots__ = ("action_uuid", "robot_id", "clip_name", "clip_properties", "animation_properties", "audio_properties")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    CLIP_NAME_FIELD_NUMBER: _ClassVar[int]
    CLIP_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    ANIMATION_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    robot_id: Robot
    clip_name: str
    clip_properties: ClipProperties
    animation_properties: AnimationProperties
    audio_properties: AudioProperties
    def __init__(self, action_uuid: _Optional[str] = ..., robot_id: _Optional[_Union[Robot, str]] = ..., clip_name: _Optional[str] = ..., clip_properties: _Optional[_Union[ClipProperties, _Mapping]] = ..., animation_properties: _Optional[_Union[AnimationProperties, _Mapping]] = ..., audio_properties: _Optional[_Union[AudioProperties, _Mapping]] = ...) -> None: ...

class AnimationProperties(_message.Message):
    __slots__ = ("priority", "opacity", "blending")
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    OPACITY_FIELD_NUMBER: _ClassVar[int]
    BLENDING_FIELD_NUMBER: _ClassVar[int]
    priority: int
    opacity: float
    blending: Transition
    def __init__(self, priority: _Optional[int] = ..., opacity: _Optional[float] = ..., blending: _Optional[_Union[Transition, _Mapping]] = ...) -> None: ...

class AudioProperties(_message.Message):
    __slots__ = ("volume", "fading")
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    FADING_FIELD_NUMBER: _ClassVar[int]
    volume: float
    fading: Transition
    def __init__(self, volume: _Optional[float] = ..., fading: _Optional[_Union[Transition, _Mapping]] = ...) -> None: ...

class ClipProperties(_message.Message):
    __slots__ = ("loop", "loop_overlap")
    LOOP_FIELD_NUMBER: _ClassVar[int]
    LOOP_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    loop: bool
    loop_overlap: float
    def __init__(self, loop: bool = ..., loop_overlap: _Optional[float] = ...) -> None: ...

class ClipData(_message.Message):
    __slots__ = ("action_uuid", "robot_id", "animation", "audio", "clip_properties", "animation_properties", "audio_properties")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ANIMATION_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    CLIP_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    ANIMATION_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    AUDIO_PROPERTIES_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    robot_id: Robot
    animation: Animation
    audio: Audio
    clip_properties: ClipProperties
    animation_properties: AnimationProperties
    audio_properties: AudioProperties
    def __init__(self, action_uuid: _Optional[str] = ..., robot_id: _Optional[_Union[Robot, str]] = ..., animation: _Optional[_Union[Animation, _Mapping]] = ..., audio: _Optional[_Union[Audio, _Mapping]] = ..., clip_properties: _Optional[_Union[ClipProperties, _Mapping]] = ..., animation_properties: _Optional[_Union[AnimationProperties, _Mapping]] = ..., audio_properties: _Optional[_Union[AudioProperties, _Mapping]] = ...) -> None: ...

class StopClip(_message.Message):
    __slots__ = ("action_uuid", "fade_out")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    FADE_OUT_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    fade_out: float
    def __init__(self, action_uuid: _Optional[str] = ..., fade_out: _Optional[float] = ...) -> None: ...

class ChangeVolume(_message.Message):
    __slots__ = ("action_uuid", "volume")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    volume: float
    def __init__(self, action_uuid: _Optional[str] = ..., volume: _Optional[float] = ...) -> None: ...

class Transition(_message.Message):
    __slots__ = ("transition_in", "transition_out")
    TRANSITION_IN_FIELD_NUMBER: _ClassVar[int]
    TRANSITION_OUT_FIELD_NUMBER: _ClassVar[int]
    transition_in: float
    transition_out: float
    def __init__(self, transition_in: _Optional[float] = ..., transition_out: _Optional[float] = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("robot_id", "frame_index", "timestamp", "top_left_x", "top_left_y", "width", "height", "score")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_INDEX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TOP_LEFT_X_FIELD_NUMBER: _ClassVar[int]
    TOP_LEFT_Y_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    frame_index: int
    timestamp: int
    top_left_x: float
    top_left_y: float
    width: float
    height: float
    score: float
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., frame_index: _Optional[int] = ..., timestamp: _Optional[int] = ..., top_left_x: _Optional[float] = ..., top_left_y: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ..., score: _Optional[float] = ...) -> None: ...

class ClipStatus(_message.Message):
    __slots__ = ("action_uuid", "robot_id", "status", "timestamp")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        STARTED: _ClassVar[ClipStatus.Status]
        UPDATED: _ClassVar[ClipStatus.Status]
        FINISHED: _ClassVar[ClipStatus.Status]
        ERROR: _ClassVar[ClipStatus.Status]
    STARTED: ClipStatus.Status
    UPDATED: ClipStatus.Status
    FINISHED: ClipStatus.Status
    ERROR: ClipStatus.Status
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    robot_id: Robot
    status: ClipStatus.Status
    timestamp: int
    def __init__(self, action_uuid: _Optional[str] = ..., robot_id: _Optional[_Union[Robot, str]] = ..., status: _Optional[_Union[ClipStatus.Status, str]] = ..., timestamp: _Optional[int] = ...) -> None: ...

class RealtimeDoA(_message.Message):
    __slots__ = ("angle", "speech_detected")
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    SPEECH_DETECTED_FIELD_NUMBER: _ClassVar[int]
    angle: float
    speech_detected: bool
    def __init__(self, angle: _Optional[float] = ..., speech_detected: bool = ...) -> None: ...

class ProcessedDoA(_message.Message):
    __slots__ = ("angle", "is_final")
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    angle: float
    is_final: bool
    def __init__(self, angle: _Optional[float] = ..., is_final: bool = ...) -> None: ...

class FilePublish(_message.Message):
    __slots__ = ("action_uuid", "timestamp", "metadata", "file_url")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    FILE_URL_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    timestamp: int
    metadata: str
    file_url: str
    def __init__(self, action_uuid: _Optional[str] = ..., timestamp: _Optional[int] = ..., metadata: _Optional[str] = ..., file_url: _Optional[str] = ...) -> None: ...

class Frame(_message.Message):
    __slots__ = ("robot_id", "index", "timestamp", "data", "encoding")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    ENCODING_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    index: int
    timestamp: int
    data: bytes
    encoding: ImageEncoding
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., index: _Optional[int] = ..., timestamp: _Optional[int] = ..., data: _Optional[bytes] = ..., encoding: _Optional[_Union[ImageEncoding, str]] = ...) -> None: ...

class HumanSpeechRequest(_message.Message):
    __slots__ = ("action_uuid", "robot_id", "script")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    SCRIPT_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    robot_id: Robot
    script: str
    def __init__(self, action_uuid: _Optional[str] = ..., robot_id: _Optional[_Union[Robot, str]] = ..., script: _Optional[str] = ...) -> None: ...

class ImageGeneration(_message.Message):
    __slots__ = ("action_uuid", "timestamp", "prompt", "generation_type", "context_image")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TYPE_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_IMAGE_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    timestamp: int
    prompt: str
    generation_type: GenerationType
    context_image: bytes
    def __init__(self, action_uuid: _Optional[str] = ..., timestamp: _Optional[int] = ..., prompt: _Optional[str] = ..., generation_type: _Optional[_Union[GenerationType, str]] = ..., context_image: _Optional[bytes] = ...) -> None: ...

class StaticAnimation(_message.Message):
    __slots__ = ("color", "in_transition_duration")
    COLOR_FIELD_NUMBER: _ClassVar[int]
    IN_TRANSITION_DURATION_FIELD_NUMBER: _ClassVar[int]
    color: Color
    in_transition_duration: float
    def __init__(self, color: _Optional[_Union[Color, str]] = ..., in_transition_duration: _Optional[float] = ...) -> None: ...

class FillCircleAnimation(_message.Message):
    __slots__ = ("primary_color", "secondary_color", "in_transition_duration", "fill_duration")
    PRIMARY_COLOR_FIELD_NUMBER: _ClassVar[int]
    SECONDARY_COLOR_FIELD_NUMBER: _ClassVar[int]
    IN_TRANSITION_DURATION_FIELD_NUMBER: _ClassVar[int]
    FILL_DURATION_FIELD_NUMBER: _ClassVar[int]
    primary_color: Color
    secondary_color: Color
    in_transition_duration: float
    fill_duration: float
    def __init__(self, primary_color: _Optional[_Union[Color, str]] = ..., secondary_color: _Optional[_Union[Color, str]] = ..., in_transition_duration: _Optional[float] = ..., fill_duration: _Optional[float] = ...) -> None: ...

class LightCommand(_message.Message):
    __slots__ = ("static_animation", "fill_circle_animation")
    STATIC_ANIMATION_FIELD_NUMBER: _ClassVar[int]
    FILL_CIRCLE_ANIMATION_FIELD_NUMBER: _ClassVar[int]
    static_animation: StaticAnimation
    fill_circle_animation: FillCircleAnimation
    def __init__(self, static_animation: _Optional[_Union[StaticAnimation, _Mapping]] = ..., fill_circle_animation: _Optional[_Union[FillCircleAnimation, _Mapping]] = ...) -> None: ...

class Ping(_message.Message):
    __slots__ = ("ping_id", "content")
    PING_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    ping_id: str
    content: str
    def __init__(self, ping_id: _Optional[str] = ..., content: _Optional[str] = ...) -> None: ...

class Pong(_message.Message):
    __slots__ = ("ping_id",)
    PING_ID_FIELD_NUMBER: _ClassVar[int]
    ping_id: str
    def __init__(self, ping_id: _Optional[str] = ...) -> None: ...

class Position2D(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class ProceduralClip(_message.Message):
    __slots__ = ("robot_id", "action_uuid", "timestamp", "type", "state", "start", "stop", "update", "pause")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    START_FIELD_NUMBER: _ClassVar[int]
    STOP_FIELD_NUMBER: _ClassVar[int]
    UPDATE_FIELD_NUMBER: _ClassVar[int]
    PAUSE_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    action_uuid: str
    timestamp: int
    type: ProceduralType
    state: ProceduralState
    start: ProceduralClipStart
    stop: ProceduralClipStop
    update: ProceduralClipUpdate
    pause: ProceduralClipPause
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., action_uuid: _Optional[str] = ..., timestamp: _Optional[int] = ..., type: _Optional[_Union[ProceduralType, str]] = ..., state: _Optional[_Union[ProceduralState, str]] = ..., start: _Optional[_Union[ProceduralClipStart, _Mapping]] = ..., stop: _Optional[_Union[ProceduralClipStop, _Mapping]] = ..., update: _Optional[_Union[ProceduralClipUpdate, _Mapping]] = ..., pause: _Optional[_Union[ProceduralClipPause, _Mapping]] = ...) -> None: ...

class ProceduralClipStart(_message.Message):
    __slots__ = ("blend_in_duration", "volume", "look_at_parameters", "track_parameters")
    BLEND_IN_DURATION_FIELD_NUMBER: _ClassVar[int]
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    LOOK_AT_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    TRACK_PARAMETERS_FIELD_NUMBER: _ClassVar[int]
    blend_in_duration: float
    volume: float
    look_at_parameters: LookAtParameters
    track_parameters: TrackParameters
    def __init__(self, blend_in_duration: _Optional[float] = ..., volume: _Optional[float] = ..., look_at_parameters: _Optional[_Union[LookAtParameters, _Mapping]] = ..., track_parameters: _Optional[_Union[TrackParameters, _Mapping]] = ...) -> None: ...

class ProceduralClipStop(_message.Message):
    __slots__ = ("blend_out_duration",)
    BLEND_OUT_DURATION_FIELD_NUMBER: _ClassVar[int]
    blend_out_duration: float
    def __init__(self, blend_out_duration: _Optional[float] = ...) -> None: ...

class ProceduralClipPause(_message.Message):
    __slots__ = ("enable",)
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    def __init__(self, enable: bool = ...) -> None: ...

class ProceduralClipUpdate(_message.Message):
    __slots__ = ("clip",)
    CLIP_FIELD_NUMBER: _ClassVar[int]
    clip: ClipData
    def __init__(self, clip: _Optional[_Union[ClipData, _Mapping]] = ...) -> None: ...

class LookAtParameters(_message.Message):
    __slots__ = ("start_body_angle", "target_position", "target_body_angle", "duration")
    START_BODY_ANGLE_FIELD_NUMBER: _ClassVar[int]
    TARGET_POSITION_FIELD_NUMBER: _ClassVar[int]
    TARGET_BODY_ANGLE_FIELD_NUMBER: _ClassVar[int]
    DURATION_FIELD_NUMBER: _ClassVar[int]
    start_body_angle: float
    target_position: Position2D
    target_body_angle: float
    duration: float
    def __init__(self, start_body_angle: _Optional[float] = ..., target_position: _Optional[_Union[Position2D, _Mapping]] = ..., target_body_angle: _Optional[float] = ..., duration: _Optional[float] = ...) -> None: ...

class TrackParameters(_message.Message):
    __slots__ = ("slow_mode_distance_threshold", "fast_mode_distance_threshold", "slow_speed", "fast_speed")
    SLOW_MODE_DISTANCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    FAST_MODE_DISTANCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SLOW_SPEED_FIELD_NUMBER: _ClassVar[int]
    FAST_SPEED_FIELD_NUMBER: _ClassVar[int]
    slow_mode_distance_threshold: float
    fast_mode_distance_threshold: float
    slow_speed: float
    fast_speed: float
    def __init__(self, slow_mode_distance_threshold: _Optional[float] = ..., fast_mode_distance_threshold: _Optional[float] = ..., slow_speed: _Optional[float] = ..., fast_speed: _Optional[float] = ...) -> None: ...

class RemoteControlCommand(_message.Message):
    __slots__ = ("command",)
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    command: str
    def __init__(self, command: _Optional[str] = ...) -> None: ...

class RobotFrame(_message.Message):
    __slots__ = ("robot_id", "frame", "timestamp")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    frame: AnimationFrame
    timestamp: int
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., frame: _Optional[_Union[AnimationFrame, _Mapping]] = ..., timestamp: _Optional[int] = ...) -> None: ...

class RobotSpeechRequest(_message.Message):
    __slots__ = ("script", "robot_id", "word_speed_range", "pitch_shift", "skip_chance")
    SCRIPT_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    WORD_SPEED_RANGE_FIELD_NUMBER: _ClassVar[int]
    PITCH_SHIFT_FIELD_NUMBER: _ClassVar[int]
    SKIP_CHANCE_FIELD_NUMBER: _ClassVar[int]
    script: str
    robot_id: Robot
    word_speed_range: float
    pitch_shift: float
    skip_chance: float
    def __init__(self, script: _Optional[str] = ..., robot_id: _Optional[_Union[Robot, str]] = ..., word_speed_range: _Optional[float] = ..., pitch_shift: _Optional[float] = ..., skip_chance: _Optional[float] = ...) -> None: ...

class ServiceCommand(_message.Message):
    __slots__ = ("command", "target_service")
    COMMAND_FIELD_NUMBER: _ClassVar[int]
    TARGET_SERVICE_FIELD_NUMBER: _ClassVar[int]
    command: Command
    target_service: Service
    def __init__(self, command: _Optional[_Union[Command, str]] = ..., target_service: _Optional[_Union[Service, str]] = ...) -> None: ...

class ToolStatus(_message.Message):
    __slots__ = ("robot_id", "action_uuid", "timestamp", "status", "name", "input", "response")
    class Status(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        TOOL_CALL_STARTED: _ClassVar[ToolStatus.Status]
        TOOL_CALL_COMPLETED: _ClassVar[ToolStatus.Status]
        TOOL_CALL_FAILED: _ClassVar[ToolStatus.Status]
        TOOL_CALL_PROCESSED: _ClassVar[ToolStatus.Status]
    TOOL_CALL_STARTED: ToolStatus.Status
    TOOL_CALL_COMPLETED: ToolStatus.Status
    TOOL_CALL_FAILED: ToolStatus.Status
    TOOL_CALL_PROCESSED: ToolStatus.Status
    class InputEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _struct_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_struct_pb2.Value, _Mapping]] = ...) -> None: ...
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    action_uuid: str
    timestamp: int
    status: ToolStatus.Status
    name: str
    input: _containers.MessageMap[str, _struct_pb2.Value]
    response: str
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., action_uuid: _Optional[str] = ..., timestamp: _Optional[int] = ..., status: _Optional[_Union[ToolStatus.Status, str]] = ..., name: _Optional[str] = ..., input: _Optional[_Mapping[str, _struct_pb2.Value]] = ..., response: _Optional[str] = ...) -> None: ...

class Keypoint(_message.Message):
    __slots__ = ("x", "y", "confidence")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    confidence: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., confidence: _Optional[float] = ...) -> None: ...

class Skeleton(_message.Message):
    __slots__ = ("keypoints",)
    KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
    keypoints: _containers.RepeatedCompositeFieldContainer[Keypoint]
    def __init__(self, keypoints: _Optional[_Iterable[_Union[Keypoint, _Mapping]]] = ...) -> None: ...

class UserDetection(_message.Message):
    __slots__ = ("robot_id", "bounding_boxes", "skeletons", "marker_id", "user_id")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    BOUNDING_BOXES_FIELD_NUMBER: _ClassVar[int]
    SKELETONS_FIELD_NUMBER: _ClassVar[int]
    MARKER_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    bounding_boxes: _containers.RepeatedCompositeFieldContainer[BoundingBox]
    skeletons: _containers.RepeatedCompositeFieldContainer[Skeleton]
    marker_id: int
    user_id: int
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., bounding_boxes: _Optional[_Iterable[_Union[BoundingBox, _Mapping]]] = ..., skeletons: _Optional[_Iterable[_Union[Skeleton, _Mapping]]] = ..., marker_id: _Optional[int] = ..., user_id: _Optional[int] = ...) -> None: ...

class UserState(_message.Message):
    __slots__ = ("robot_id", "user_id", "timestamp", "status")
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    robot_id: Robot
    user_id: int
    timestamp: int
    status: PresenceStatus
    def __init__(self, robot_id: _Optional[_Union[Robot, str]] = ..., user_id: _Optional[int] = ..., timestamp: _Optional[int] = ..., status: _Optional[_Union[PresenceStatus, str]] = ...) -> None: ...

class UserTrackingStatus(_message.Message):
    __slots__ = ("action_uuid", "robot_id", "timestamp", "status")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    ROBOT_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    robot_id: Robot
    timestamp: int
    status: TrackingStatus
    def __init__(self, action_uuid: _Optional[str] = ..., robot_id: _Optional[_Union[Robot, str]] = ..., timestamp: _Optional[int] = ..., status: _Optional[_Union[TrackingStatus, str]] = ...) -> None: ...

class UserUtterance(_message.Message):
    __slots__ = ("action_uuid", "status", "timestamp", "text")
    ACTION_UUID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    action_uuid: str
    status: UserUtteranceStatus
    timestamp: int
    text: str
    def __init__(self, action_uuid: _Optional[str] = ..., status: _Optional[_Union[UserUtteranceStatus, str]] = ..., timestamp: _Optional[int] = ..., text: _Optional[str] = ...) -> None: ...
