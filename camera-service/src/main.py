# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any
from urllib.parse import urlencode, urlparse

import cv2
import pyudev  # type: ignore
from aiohttp import web
from configuration import CameraConfig
from workmesh.config import load_config
from workmesh.messages import Frame, ImageEncoding, Robot
from workmesh.service import produces
from workmesh.service_executor import ServiceExecutor

from workmesh import Service, camera_frame_topic


def is_valid_url(url_string: str) -> bool:
    try:
        result = urlparse(url_string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def _get_camera_records() -> list[dict[str, str]]:
    ctx = pyudev.Context()  # type: ignore[attr-defined]
    records: list[dict[str, str]] = []
    for dev in ctx.list_devices(subsystem="video4linux"):
        if not getattr(dev, "device_node", None):
            continue
        parent = dev.find_parent("usb", "usb_device")
        vendor = model = serial = ""
        if parent is not None:
            vendor = (parent.attributes.get("manufacturer") or b"").decode(
                errors="ignore"
            )
            model = (parent.attributes.get("product") or b"").decode(errors="ignore")
            serial = (parent.attributes.get("serial") or b"").decode(errors="ignore")
        name = (
            dev.properties.get("ID_V4L_PRODUCT")
            or f"{vendor} {model}".strip()
            or dev.device_node
        )
        usb_id = (
            f"{dev.properties.get('ID_VENDOR_ID')}:{dev.properties.get('ID_MODEL_ID')}"
        )
        records.append(
            {
                "devnode": str(dev.device_node),
                "name": str(name),
                "usb_id": str(usb_id),
                "serial": str(serial),
            }
        )
    return records


def _device_index_for_serial(serial: str) -> tuple[bool, int | None]:
    """
    Get the device index for a given serial number.

    Args:
        serial: The serial number of the camera.

    Returns:
        (bool, int | None): Whether the camera was found and the device index.
    """

    record = next(
        (rec for rec in _get_camera_records() if serial in rec.get("serial", "")), None
    )
    if record is not None:
        return True, int(record.get("devnode", "").split("/dev/video")[1])
    default_record = next(
        (
            rec
            for rec in _get_camera_records()
            if rec.get("devnode", "").startswith("/dev/video")
        ),
        None,
    )
    if default_record is not None:
        return False, int(default_record.get("devnode", "").split("/dev/video")[1])
    return False, None


class CameraService(Service):
    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._config = config
        self._frame_id = 0
        self._me = getattr(Robot, self._config.robot_id.name)

        max_width = max(
            self._config.streaming_resolution.width,
            self._config.capture_resolution.width,
        )
        max_height = max(
            self._config.streaming_resolution.height,
            self._config.capture_resolution.height,
        )

        if is_valid_url(self._config.camera_serial_or_url):
            # Add query parameters for width, height, and fps to the camera URL
            base_url = self._config.camera_serial_or_url
            params = {
                "width": max_width,
                "height": max_height,
                "fps": self._config.fps,
            }
            camera_url = f"{base_url}?{urlencode(params)}"

            self.logger.info(f"Attempting to connect to remote webcam: {camera_url}")
            self._camera_index = camera_url

        else:
            found, self._camera_index = _device_index_for_serial(
                self._config.camera_serial_or_url
            )
            if not found and self._camera_index is None:
                raise ValueError(
                    "No cameras available. Please check the camera connection."
                )
            if not found:
                self.logger.warning(
                    f"No camera found with serial {self._config.camera_serial_or_url}. "
                    f"Using default camera index {self._camera_index}"
                )

        self._camera = cv2.VideoCapture(self._camera_index)  # pyright: ignore[reportArgumentType, reportCallIssue]
        self._camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),  # pyright: ignore[reportAttributeAccessIssue]
        )
        self._camera.set(cv2.CAP_PROP_FPS, self._config.fps)
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
        self.logger.info(
            f"Serial ID: {self._config.camera_serial_or_url} "
            f"--> Camera Index: {self._camera_index}"
        )
        self.logger.info(
            f"Serial ID: {self._config.camera_serial_or_url} "
            f"--> Robot ID: {self._config.robot_id}"
        )

        self.logger.info(f"Log level: {self._config.log_level}")

        # Initialize HTTP server
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        if self._config.enable_http_server:
            self.app = web.Application()
            self.app.router.add_get("/capture", self.handle_capture_request)

        self.logger.info(f"Mirror: {self._config.mirror_horizontal}")
        self._is_eof = False
        self.create_task(self.produce_camera_frame())  # pyright: ignore[reportAttributeAccessIssue]

    @produces(camera_frame_topic)
    async def produce_camera_frame(self) -> AsyncGenerator[Frame, Any]:
        """Publish camera frames to the camera topic."""
        await asyncio.sleep(2)  # Wait for camera to be ready

        while True:
            ret, frame = self._camera.read()
            if not ret:
                self._is_eof = True
                self.logger.info(f"EOF received from camera {self._camera_index}")
                break
            self._frame_id += 1
            # Scale frame to self._config.streaming_resolution
            target_width = self._config.streaming_resolution.width
            target_height = self._config.streaming_resolution.height
            if target_width != frame.shape[1] or target_height != frame.shape[0]:
                self.logger.debug(
                    f"Scale frame {self._frame_id} from "
                    f"{frame.shape[1]}x{frame.shape[0]} "
                    f"to {target_width}x{target_height}"
                )
                frame = cv2.resize(
                    frame, (target_width, target_height), interpolation=cv2.INTER_AREA
                )
            self.logger.debug(
                f"Mirror frame {self._frame_id}: {self._config.mirror_horizontal}"
            )
            if self._config.mirror_horizontal:
                frame = cv2.flip(frame, 1)
            self.logger.debug(f"Encode frame {self._frame_id}")
            _, encoded_frame = cv2.imencode(self._config.encoding.value, frame)
            encoded_frame = encoded_frame.tobytes()
            self.logger.debug(f"Send frame {self._frame_id}")
            yield Frame(
                index=self._frame_id,
                data=encoded_frame,
                encoding=getattr(ImageEncoding, self._config.encoding.name),
                robot_id=self._me,
                timestamp=int(time.time() * 1000),
            )
            self.logger.debug(f"Sent frame {self._frame_id}")

    async def handle_capture_request(self, _0: web.Request) -> web.Response:
        self.logger.info("Capturing image...")
        ret, frame = self._camera.read()
        if not ret:
            return web.Response(status=500, text="Failed to capture image")
        target_width = self._config.capture_resolution.width
        target_height = self._config.capture_resolution.height
        if target_width != frame.shape[1] or target_height != frame.shape[0]:
            self.logger.debug(
                f"Scale frame from {frame.shape[1]}x{frame.shape[0]} "
                f"to {target_width}x{target_height}"
            )
            frame = cv2.resize(
                frame, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
        self.logger.debug(f"Mirror single shot frame: {self._config.mirror_horizontal}")
        if self._config.mirror_horizontal:
            frame = cv2.flip(frame, 1)
        self.logger.debug("Encode single shot frame")
        encoding = self._config.encoding
        _1, encoded_frame = cv2.imencode(encoding.value, frame)
        encoded_frame = encoded_frame.tobytes()
        return web.Response(
            status=200,
            body=encoded_frame,
            headers={
                "Content-Type": f"image/{encoding.name.lower()}",
                "Content-Disposition": f'inline; filename="capture{encoding.value}"',
            },
        )

    async def start_http_server(self, host: str = "0.0.0.0", port: int = 7071) -> None:
        """Start the HTTP server"""
        if self.app is None:
            raise ValueError("App is not initialized")
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, host, port)
        await self.site.start()
        self.logger.info(f"HTTP server started on http://{host}:{port}")

    async def stop_http_server(self) -> None:
        """Stop the HTTP server"""
        if self.runner and self.site:
            await self.site.stop()
            await self.runner.cleanup()
            self.logger.info("HTTP server stopped")

    async def run(self) -> None:
        # super().run() # This will "block"

        # Wait for camera's EOF.
        await asyncio.sleep(2)  # Wait for camera to be ready
        while not self._is_eof:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the camera service."""
        await self.stop_http_server()
        await super().stop()
        self._camera.release()


async def main():
    config = load_config(CameraConfig)
    service = CameraService(config)
    if config.enable_http_server:
        await service.start_http_server(
            str(config.http_bind_address), config.http_bind_port
        )
    await ServiceExecutor([service]).run()


if __name__ == "__main__":
    asyncio.run(main())
