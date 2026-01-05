# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aiokafka import ConsumerRecord
from configuration import MetricsServiceConfig
from opentelemetry.metrics import Histogram
from workmesh.config import load_config
from workmesh.messages import (
    ClipData,
    HumanSpeechRequest,
    UserUtterance,
    UserUtteranceStatus,
)
from workmesh.service import Service, subscribe
from workmesh.service_executor import ServiceExecutor

from workmesh import (
    clip_data_topic,
    human_speech_request_topic,
    user_utterance_topic,
)


class BaseMetric(ABC):
    """Base class for metrics that correlate request/response pairs."""

    def __init__(self, service: Service, max_pending_age_seconds: int = 600):
        self.service = service
        self.logger = service.logger
        self.pending: dict[str, tuple[Any, int]] = {}
        self.max_pending_age_seconds = max_pending_age_seconds
        self.total_tracked = 0
        self.total_matched = 0
        self.total_orphaned = 0

    @abstractmethod
    def get_correlation_key(self, msg: Any) -> str:
        """Extract the correlation key from a message."""
        pass

    @abstractmethod
    def on_request(self, msg: Any, kafka_record: ConsumerRecord) -> None:
        """Handle incoming request message."""
        pass

    @abstractmethod
    def on_response(self, msg: Any, kafka_record: ConsumerRecord) -> None:
        """Handle incoming response message."""
        pass

    def cleanup_stale(self) -> int:
        """Remove stale pending requests. Returns number of items removed."""
        current_time_ms = int(time.time() * 1000)
        max_age_ms = self.max_pending_age_seconds * 1000

        stale_keys = [
            key
            for key, (_, timestamp) in self.pending.items()
            if current_time_ms - timestamp > max_age_ms
        ]

        for key in stale_keys:
            del self.pending[key]

        return len(stale_keys)


@dataclass
class PendingTTSRequest:
    """Data stored for a pending TTS request."""

    request: HumanSpeechRequest
    script_length: int


class TTSLatencyMetric(BaseMetric):
    """Tracks TTS latency from HumanSpeechRequest to ClipData."""

    def __init__(self, service: Service):
        super().__init__(service)
        meter = service.meter_provider.get_meter("tts_metrics")
        self.latency_histogram: Histogram = meter.create_histogram(
            name="tts.latency",
            unit="ms",
            description="Time from HumanSpeechRequest to ClipData response",
            explicit_bucket_boundaries_advisory=[
                5,
                10,
                15,
                20,
                25,
                30,
                40,
                50,
                75,
                100,
                150,
                200,
                300,
                500,
                1000,
            ],
        )

    def get_correlation_key(self, msg: HumanSpeechRequest | ClipData) -> str:
        return msg.action_uuid

    def on_request(self, msg: HumanSpeechRequest, kafka_record: ConsumerRecord) -> None:
        key = self.get_correlation_key(msg)
        self.pending[key] = (
            PendingTTSRequest(request=msg, script_length=len(msg.script)),
            kafka_record.timestamp,
        )
        self.total_tracked += 1
        self.logger.debug(
            f"[TTS Latency] Tracked request: {key}, script_length={len(msg.script)}"
        )

    def on_response(self, msg: ClipData, kafka_record: ConsumerRecord) -> None:
        key = self.get_correlation_key(msg)
        pending_data = self.pending.pop(key, None)

        if pending_data is None:
            self.total_orphaned += 1
            self.logger.warning(f"[TTS Latency] Orphaned response: {key}")
            return

        pending, request_timestamp = pending_data
        latency_ms = kafka_record.timestamp - request_timestamp

        self.latency_histogram.record(latency_ms)
        self.total_matched += 1

        self.logger.info(
            f"[TTS Latency] Recorded: {key}, latency={latency_ms}ms, "
            f"script_length={pending.script_length}"
        )


class TTSRateMetric(BaseMetric):
    """Tracks TTS processing rate in characters per second."""

    def __init__(self, service: Service):
        super().__init__(service)
        meter = service.meter_provider.get_meter("tts_metrics")
        self.rate_histogram: Histogram = meter.create_histogram(
            name="tts.rate",
            unit="chars/s",
            description="TTS processing rate in characters per second",
            explicit_bucket_boundaries_advisory=[
                100,
                200,
                300,
                400,
                500,
                550,
                600,
                650,
                700,
                750,
                800,
                900,
                1000,
                1200,
                1500,
                2000,
            ],
        )

    def get_correlation_key(self, msg: HumanSpeechRequest | ClipData) -> str:
        return msg.action_uuid

    def on_request(self, msg: HumanSpeechRequest, kafka_record: ConsumerRecord) -> None:
        key = self.get_correlation_key(msg)
        self.pending[key] = (
            PendingTTSRequest(request=msg, script_length=len(msg.script)),
            kafka_record.timestamp,
        )
        self.total_tracked += 1

    def on_response(self, msg: ClipData, kafka_record: ConsumerRecord) -> None:
        key = self.get_correlation_key(msg)
        pending_data = self.pending.pop(key, None)

        if pending_data is None:
            self.total_orphaned += 1
            return

        pending, request_timestamp = pending_data
        latency_ms = kafka_record.timestamp - request_timestamp

        if latency_ms > 0:
            rate = (pending.script_length / latency_ms) * 1000
            self.rate_histogram.record(rate)
            self.total_matched += 1
            self.logger.info(f"[TTS Rate] Recorded: {key}, rate={rate:.2f} chars/s")


class E2ELatencyMetric:
    """
    Tracks end-to-end latency from user speech finish to robot audio generation.

    Uses timestamp-based correlation (not action_uuid) since UserUtterance and ClipData
    have different UUIDs. Tracks the most recent UserUtterance.FINISHED and measures
    time to the next ClipData within a configurable time window.
    """

    def __init__(self, service: Service, max_window_seconds: int = 180):
        self.service = service
        self.logger = service.logger
        self.max_window_ms = max_window_seconds * 1000
        self.latest_user_utterance: tuple[int, str] | None = (
            None  # (timestamp_ms, action_uuid)
        )
        self.total_tracked = 0
        self.total_matched = 0
        self.total_discarded = 0

        meter = service.meter_provider.get_meter("e2e_metrics")
        self.latency_histogram: Histogram = meter.create_histogram(
            name="e2e.latency",
            unit="ms",
            description="End-to-end latency from user speech finish to robot"
            " audio generation",
            explicit_bucket_boundaries_advisory=[
                1000,
                2000,
                3000,
                5000,
                7500,
                10000,
                15000,
                20000,
                30000,
                45000,
                60000,
                90000,
                120000,
            ],
        )

    def on_user_utterance_finished(
        self, msg: UserUtterance, kafka_record: ConsumerRecord
    ) -> None:
        """Track when a user finishes speaking."""
        if msg.status == UserUtteranceStatus.USER_UTTERANCE_FINISHED:
            self.latest_user_utterance = (kafka_record.timestamp, msg.action_uuid)
            self.total_tracked += 1
            self.logger.debug(
                f"[E2E Latency] Tracked user utterance finish: {msg.action_uuid}"
            )

    def on_clip_data(self, msg: ClipData, kafka_record: ConsumerRecord) -> None:
        """Measure latency to the first ClipData after user finished speaking."""
        if self.latest_user_utterance is None:
            return

        utterance_timestamp, utterance_uuid = self.latest_user_utterance
        latency_ms = kafka_record.timestamp - utterance_timestamp

        # Check if within acceptable time window
        if latency_ms > self.max_window_ms:
            self.total_discarded += 1
            self.logger.debug(
                f"[E2E Latency] Discarded: latency={latency_ms}ms exceeds max"
                f" window={self.max_window_ms}ms"
            )
            # Clear stale utterance
            self.latest_user_utterance = None
            return

        # Only measure positive latencies (ClipData after UserUtterance)
        if latency_ms > 0:
            self.latency_histogram.record(latency_ms)
            self.total_matched += 1
            self.logger.info(
                f"[E2E Latency] Recorded: latency={latency_ms}ms "
                f"(from utterance {utterance_uuid} to clip {msg.action_uuid})"
            )
            # Clear to avoid double-counting with next ClipData
            self.latest_user_utterance = None


class MetricsService(Service):
    """
    Metrics service for tracking OpenTelemetry metrics from Kafka message flows.

    Uses a modular metric system where each metric is a separate class instance.
    """

    CLEANUP_INTERVAL_SECONDS = 300  # 5 minutes
    E2E_MAX_WINDOW_SECONDS = 180  # 3 minutes

    def __init__(self, config: MetricsServiceConfig | None = None) -> None:
        if config is None:
            config = MetricsServiceConfig()
        super().__init__(config)

        # Initialize metrics
        self.tts_latency_metric = TTSLatencyMetric(self)
        self.tts_rate_metric = TTSRateMetric(self)
        self.e2e_latency_metric = E2ELatencyMetric(
            self, max_window_seconds=self.E2E_MAX_WINDOW_SECONDS
        )

        # List of all metrics for cleanup and stats (E2E doesn't use BaseMetric)
        self.metrics: list[BaseMetric] = [
            self.tts_latency_metric,
            self.tts_rate_metric,
        ]

        # Start cleanup task
        self.create_task(self._cleanup_task())

        self.logger.info(
            "MetricsService initialized with metrics: TTS Latency, TTS Rate, "
            "E2E Latency"
        )

    @subscribe(human_speech_request_topic)
    async def on_human_speech_request(
        self, msg: HumanSpeechRequest, kafka_record
    ) -> None:
        """Delegate to all metrics that track HumanSpeechRequest."""
        if kafka_record is None:
            self.logger.error("kafka_record is required for metrics tracking")
            return

        # Notify all relevant metrics
        self.tts_latency_metric.on_request(msg, kafka_record)
        self.tts_rate_metric.on_request(msg, kafka_record)

    @subscribe(user_utterance_topic)
    async def on_user_utterance(self, msg: UserUtterance, kafka_record) -> None:
        """Delegate to all metrics that track UserUtterance."""
        if kafka_record is None:
            self.logger.error("kafka_record is required for metrics tracking")
            return

        # Notify E2E metric
        self.e2e_latency_metric.on_user_utterance_finished(msg, kafka_record)

    @subscribe(clip_data_topic)
    async def on_clip_data(self, msg: ClipData, kafka_record) -> None:
        """Delegate to all metrics that track ClipData."""
        if kafka_record is None:
            self.logger.error("kafka_record is required for metrics tracking")
            return

        # Notify all relevant metrics
        self.tts_latency_metric.on_response(msg, kafka_record)
        self.tts_rate_metric.on_response(msg, kafka_record)
        self.e2e_latency_metric.on_clip_data(msg, kafka_record)

    async def _cleanup_task(self) -> None:
        """Periodically clean up stale pending requests across all metrics."""
        while True:
            await asyncio.sleep(self.CLEANUP_INTERVAL_SECONDS)

            total_cleaned = 0
            for metric in self.metrics:
                cleaned = metric.cleanup_stale()
                total_cleaned += cleaned

            if total_cleaned > 0:
                self.logger.warning(
                    f"Cleaned up {total_cleaned} stale requests across all metrics"
                )

            # Log stats for each metric
            for metric in self.metrics:
                metric_name = metric.__class__.__name__
                self.logger.info(
                    f"[{metric_name}] Pending: {len(metric.pending)}, "
                    f"Tracked: {metric.total_tracked}, "
                    f"Matched: {metric.total_matched}, "
                    f"Orphaned: {metric.total_orphaned}"
                )

            # Log E2E metric stats separately (doesn't extend BaseMetric)
            self.logger.info(
                f"[E2ELatencyMetric] "
                f"Tracked: {self.e2e_latency_metric.total_tracked}, "
                f"Matched: {self.e2e_latency_metric.total_matched}, "
                f"Discarded: {self.e2e_latency_metric.total_discarded}"
            )


async def main() -> None:
    config = load_config(MetricsServiceConfig)
    await ServiceExecutor([MetricsService(config)]).run()


if __name__ == "__main__":
    asyncio.run(main())
