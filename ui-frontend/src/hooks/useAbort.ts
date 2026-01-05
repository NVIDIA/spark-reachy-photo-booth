/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect } from "react";
import type { AbortMessage } from "../types/api";

/**
 * Listens for the "SPACE" and "ESC" keyboard shortcuts to abort
 * the interaction. The abort signal is sent to the ui-server
 * through sendMessage function
 * @param sendMessage
 */
export default function useAbort(sendMessage: (message: AbortMessage) => void) {
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.code === "Space" || event.code === "Escape") {
        event.preventDefault();
        sendMessage({ message_type: "AbortMessage" });
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [sendMessage]);
}
