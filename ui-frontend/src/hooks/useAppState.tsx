/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from "react";
import type { AppState, BaseMessage } from "../types/api";

export default function useAppState(
  lastMessage: BaseMessage | null
): AppState | null {
  const [appState, setAppState] = useState<AppState | null>(null);
  useEffect(() => {
    if (lastMessage?.message_type === "AppState") {
      setAppState(lastMessage);
    }
  }, [lastMessage]);

  return appState;
}
