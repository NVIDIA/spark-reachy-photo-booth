# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib

from .base import BaseDetector


def create_detector(model_type: str, config: dict | None = None) -> BaseDetector:
    """Create a detector instance from a fully-qualified class path.

    Example: "models.detector.byte_track.ByteTrackDetector".

    Args:
        model_type (str): The fully-qualified class path of the detector.
        config (dict | None): The configuration for the detector.

    Returns:
        BaseDetector: The detector instance.

    """
    if not model_type or "." not in model_type:
        raise ValueError(
            "model_type must be a fully qualified class path like "
            "'models.detector.byte_track.ByteTrackDetector'"
        )

    module_path, _, class_name = model_type.rpartition(".")
    module = importlib.import_module(module_path)
    class_: type[BaseDetector] = getattr(module, class_name)
    if not issubclass(class_, BaseDetector):
        raise TypeError(f"{class_.__name__} is not a subclass of BaseDetector")

    return class_(**(config or {}))
