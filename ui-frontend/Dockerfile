# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM node:24-slim AS builder

WORKDIR /app

COPY ui-frontend/package.json ui-frontend/package-lock.json ./

RUN npm ci

COPY ui-frontend/ .

ARG VITE_UI_SERVER_WS_URL="ws://localhost:9000/ws"
ARG VITE_VIDEO_WIDTH=1920
ARG VITE_VIDEO_HEIGHT=1080

ENV VITE_UI_SERVER_WS_URL=$VITE_UI_SERVER_WS_URL
ENV VITE_VIDEO_WIDTH=$VITE_VIDEO_WIDTH
ENV VITE_VIDEO_HEIGHT=$VITE_VIDEO_HEIGHT

RUN npm run build

FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html
COPY --from=builder /app/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
