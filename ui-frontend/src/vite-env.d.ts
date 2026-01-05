/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_UI_SERVER_WS_URL: string;
  readonly VITE_VIDEO_WIDTH: number;
  readonly VITE_VIDEO_HEIGHT: number;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
