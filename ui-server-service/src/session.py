# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Awaitable, Callable
from contextlib import suppress

from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from camera_track import StreamTrack
from fastapi import WebSocket, WebSocketDisconnect
from messages import (
    AbortMessage,
    AppState,
    BaseMessage,
    Signalling,
)
from workmesh.messages import Frame


class Session:
    def __init__(
        self,
        websocket: WebSocket,
        on_abort: Callable[[], Awaitable[None]] | None = None,
    ):
        self.video_track = StreamTrack(1920, 1080)
        self.peer_connection: RTCPeerConnection = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=[])
        )
        self.peer_connection.addTrack(self.video_track)
        self.websocket = websocket
        self.on_abort = on_abort

    async def send(self, msg: Signalling | AppState):
        data = msg.model_dump_json()
        await self.websocket.send_text(data)

    async def receive(self) -> Signalling | AppState | AbortMessage:
        data = await self.websocket.receive_text()
        return BaseMessage.model_validate_json(data).root

    async def create_offer(self) -> RTCSessionDescription:
        offer = await self.peer_connection.createOffer()
        await self.peer_connection.setLocalDescription(offer)
        return self.peer_connection.localDescription

    async def set_remote_description(self, description: RTCSessionDescription):
        await self.peer_connection.setRemoteDescription(description)

    async def update(self, state: AppState):
        await self.send(state)

    async def update_frame(self, frame: Frame):
        self.video_track.update_frame(frame)

    async def __aenter__(self):
        await self.websocket.accept()
        offer = await self.create_offer()
        await self.send(Signalling(type="offer", sdp=offer.sdp))
        match await self.receive():
            case Signalling(sdp=sdp, type=type):
                await self.set_remote_description(
                    RTCSessionDescription(sdp=sdp, type=type)
                )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        with suppress(RuntimeError):
            await self.websocket.close()
        await self.peer_connection.close()

    async def read_from_websocket(self):
        with suppress(WebSocketDisconnect):
            while data := await self.websocket.receive_text():
                message = BaseMessage.model_validate_json(data).root
                if isinstance(message, AbortMessage):
                    print("abort message received")
                    if self.on_abort:
                        await self.on_abort()
                else:
                    print(data)

    async def run(self):
        await self.read_from_websocket()
