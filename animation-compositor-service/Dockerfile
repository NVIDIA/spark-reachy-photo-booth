# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.8.24-python3.12-bookworm-slim AS builder

# Needed for audio processing

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y \
    alsa-utils \
    build-essential \
    ffmpeg \
    git \
    libasound2-dev \
    libpulse0 \
    pipewire \
    pipewire-alsa \
    portaudio19-dev \
    pulseaudio-utils

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=./workmesh,target=/app/workmesh \
    uv sync --frozen --no-install-project --package animation-compositor --no-editable

FROM python:3.12-slim-bookworm

# Install audio streaming dependencies

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y \
    alsa-utils \
    ffmpeg \
    libasound2-dev \
    libpulse0 \
    pipewire \
    pipewire-alsa \
    portaudio19-dev \
    pulseaudio-utils

ENV PYTHONUNBUFFERED=1

RUN useradd -ms /bin/bash -G audio app

WORKDIR /app/src

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app ./animation-compositor-service /app

USER app

ENTRYPOINT [ "bash", "-c" ]
CMD [ "python main.py" ]
