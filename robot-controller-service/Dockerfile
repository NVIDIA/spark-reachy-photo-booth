# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.8.24-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

ARG UV_FLAGS=--no-dev

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y git

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=./uv.lock,target=uv.lock \
    --mount=type=bind,source=./pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=./workmesh,target=/app/workmesh \
    uv sync --frozen --no-install-project --package robot-controller ${UV_FLAGS} --no-editable 

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1

RUN useradd -ms /bin/bash -G dialout,video,plugdev app

WORKDIR /app/src

ENV PATH="/app/.venv/bin:$PATH"

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y curl libgl-dev libglib2.0-0 libportaudio2 --no-install-recommends

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app ./robot-controller-service /app

USER app

ENTRYPOINT [ "bash", "-c" ]
CMD [ "python main.py" ]
