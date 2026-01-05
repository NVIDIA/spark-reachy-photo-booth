/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState, useCallback } from "react";

export default function useWebsocket<T>(url: string) {
  const [websocket, setWebsocket] = useState<WebSocket | null>(null);
  const [lastMessage, setLastMessage] = useState<T | null>(null);
  const [readyState, setReadyState] = useState<number>(WebSocket.CONNECTING);

  useEffect(() => {
    const ws = new WebSocket(url);
    setWebsocket(ws);

    const handleOpen = () => setReadyState(ws.readyState);
    const handleClose = () => setReadyState(ws.readyState);

    const handleMessage = (event: MessageEvent) => {
      try {
        const parsed = JSON.parse(event.data);
        setLastMessage(parsed);
      } catch (e) {
        console.error("Failed to parse websocket message", e);
      }
    };

    ws.addEventListener("open", handleOpen);
    ws.addEventListener("close", handleClose);
    ws.addEventListener("message", handleMessage);

    return () => {
      ws.removeEventListener("open", handleOpen);
      ws.removeEventListener("close", handleClose);
      ws.removeEventListener("message", handleMessage);
      ws.close();
    };
  }, [url]);

  const sendMessage = useCallback(
    (message: unknown) => {
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify(message));
      }
    },
    [websocket]
  );

  return {
    websocket,
    lastMessage,
    readyState,
    sendMessage
  };
}
