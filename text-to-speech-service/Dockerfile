# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM ghcr.io/astral-sh/uv:0.8.24-python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0

WORKDIR /app

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y \
    build-essential \
    git

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=./workmesh,target=/app/workmesh \
    uv sync --frozen --no-install-project --package text-to-speech --no-editable

RUN --mount=type=bind,source=./text-to-speech-service/scripts,target=/app/scripts \
    /app/.venv/bin/python /app/scripts/cache_kokoro_model_and_voices.py

FROM python:3.12-slim-bookworm

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean && \
    apt-get update && apt-get install -y \
    ffmpeg

ENV PYTHONUNBUFFERED=1

RUN useradd -ms /bin/bash app

WORKDIR /app/src

ENV PATH="/app/.venv/bin:$PATH"

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder --chown=app:app /root/.cache/huggingface/hub/models--hexgrad--Kokoro-82M /home/app/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
COPY --chown=app:app ./text-to-speech-service /app

ENV LD_LIBRARY_PATH=/app/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/:$LD_LIBRARY_PATH

USER app

ENTRYPOINT [ "bash", "-c" ]
CMD [ "python main.py" ]
