/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState, useRef } from "react";
import type { Signalling, BaseMessage } from "../types/api";

interface UseWebRTCOptions {
  lastMessage: BaseMessage | null;
  sendMessage: (message: Signalling) => void;
}

export default function useWebRTC({
  lastMessage,
  sendMessage
}: UseWebRTCOptions) {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const peerConnection = useRef<RTCPeerConnection | null>(null);

  useEffect(() => {
    // Initialize peer connection only once
    if (!peerConnection.current) {
      const rtcPeerConnection = new RTCPeerConnection({
        iceServers: []
      });
      peerConnection.current = rtcPeerConnection;

      rtcPeerConnection.ontrack = (ev) => {
        if (ev.streams && ev.streams[0]) {
          setStream(ev.streams[0]);
        }
      };
    }

    return () => {
      // Cleanup on unmount
      if (peerConnection.current) {
        peerConnection.current.close();
        peerConnection.current = null;
      }
    };
  }, []);

  useEffect(() => {
    const handleSignalling = async () => {
      if (!lastMessage || lastMessage.message_type !== "Signalling") return;

      const rtcPeerConnection = peerConnection.current;
      if (!rtcPeerConnection) return;

      const { type, sdp } = lastMessage;

      try {
        await rtcPeerConnection.setRemoteDescription({ type, sdp });

        if (type === "offer") {
          const answer = await rtcPeerConnection.createAnswer();
          await rtcPeerConnection.setLocalDescription(answer);

          const message: Signalling = {
            message_type: "Signalling",
            sdp: rtcPeerConnection.localDescription!.sdp,
            type: "answer"
          };
          sendMessage(message);
        }
      } catch (err) {
        console.error("Error handling signalling message:", err);
      }
    };

    handleSignalling();
  }, [lastMessage, sendMessage]);

  return stream;
}
