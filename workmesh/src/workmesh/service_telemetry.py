# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
from os import linesep

from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler, ReadableLogRecord
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from pydantic.networks import HttpUrl


class ServiceTelemetry:
    def __init__(
        self,
        service_name: str,
        service_instance_id: str | None = None,
        otel_endpoint: HttpUrl | None = None,
        log_level: str = "INFO",
    ):
        self.service_name = service_name

        self.service_instance_id = service_instance_id

        resource = Resource.create(
            {
                "service.name": service_name,
                "service.instance.id": (
                    service_instance_id if service_instance_id else service_name
                ),
            }
        )

        self._logger_provider = LoggerProvider(
            resource=resource,
        )

        self._otel_endpoint = otel_endpoint or HttpUrl("http://lgtm-otel:4317")
        endpoint = f"{self._otel_endpoint.scheme}://{self._otel_endpoint.host}:{self._otel_endpoint.port}"
        self._otlp_exporter = OTLPLogExporter(
            endpoint=endpoint,
            insecure=True,
        )
        self._logger_provider.add_log_record_processor(
            BatchLogRecordProcessor(self._otlp_exporter)
        )

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True)
        )
        self._meter_provider = MeterProvider(
            resource=resource, metric_readers=[metric_reader]
        )

        self._console_exporter = ConsoleLogRecordExporter(formatter=self._format_log)

        self._logger_provider.add_log_record_processor(
            SimpleLogRecordProcessor(self._console_exporter)
        )

        self._handler = LoggingHandler(
            level=logging.NOTSET, logger_provider=self._logger_provider
        )

        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        self.logger.addHandler(self._handler)

    @staticmethod
    def _format_log(record: ReadableLogRecord) -> str:
        assert record.log_record.timestamp is not None
        timestamp = datetime.datetime.fromtimestamp(
            record.log_record.timestamp / 1e9
        ).isoformat()
        return (
            f"[{record.log_record.severity_text}] {timestamp}: {record.log_record.body}"
            + linesep
        )

    def getLogger(self) -> logging.Logger:
        return self.logger

    def get_meter_provider(self) -> MeterProvider:
        return self._meter_provider

    def shutdown(self) -> None:
        self._meter_provider.shutdown()
        self._logger_provider.shutdown()
