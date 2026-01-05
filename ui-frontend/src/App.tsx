/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState, useRef } from "react";
import useAppState from "./hooks/useAppState";
import useWebsocket from "./hooks/useWebsocket";
import useWebRTC from "./hooks/useWebRTC";
import ImageGeneration from "./ImageGeneration";
import VideoFeed from "./VideoFeed";
import Transcript from "./Transcript";
import BigMessage from "./BigMessage";
import type { BaseMessage } from "./types/api";
import FullScreen from "./FullScreen";
import UserTracking from "./UserTracking";
import Crosshair from "./Crosshair";
import LightBorder from "./LightBorder";
import Binoculars from "./Binoculars";
import CapturedImage from "./CapturedImage";
import Sleeping from "./Sleeping";
import QRCode from "./QRCode";
import Countdown from "./Countdown";
import WakeUpAnimation from "./WakeUpAnimation";

const WS_URL =
  import.meta.env.VITE_UI_SERVER_WS_URL || "ws://localhost:9000/ws";

function App() {
  const { lastMessage, sendMessage, readyState } =
    useWebsocket<BaseMessage>(WS_URL);
  const stream = useWebRTC({ lastMessage, sendMessage });
  const appState = useAppState(lastMessage);
  const [showWakeUp, setShowWakeUp] = useState(false);
  const prevToolTypeRef = useRef<string | null>(null);

  useEffect(() => {
    if (!appState) return;

    const currentToolType = appState.tool?.type;
    const prevToolType = prevToolTypeRef.current;

    // Transition from falsy (no tool) to "AskHuman"
    if (!prevToolType && currentToolType === "AskHuman") {
      setShowWakeUp(true);
      // Remove the animation after it's done (roughly 2s based on animation duration)
      setTimeout(() => setShowWakeUp(false), 2000);
    }

    prevToolTypeRef.current = currentToolType || null;
  }, [appState]);

  if (readyState === WebSocket.CLOSED) {
    return (
      <FullScreen>
        <BigMessage>
          Websocket closed or failed to connect. Please reload the page.
        </BigMessage>
      </FullScreen>
    );
  }

  if (readyState === WebSocket.CONNECTING) {
    return (
      <FullScreen>
        <BigMessage>Connecting to server...</BigMessage>
      </FullScreen>
    );
  }

  if (!appState) {
    return (
      <FullScreen>
        <BigMessage>Waiting for app state...</BigMessage>
      </FullScreen>
    );
  }

  const {
    tool,
    transcript,
    animation,
    tracking_data,
    qr_code,
    countdown_started_at,
    countdown_duration
  } = appState;

  if (!tool) {
    return (
      <FullScreen>
        <Sleeping className="h-full w-full" />
        <Binoculars className="absolute h-full w-full" />
      </FullScreen>
    );
  }

  const showVideoFeed =
    tool.type === "AskHuman" ||
    tool.type === "GreetUser" ||
    (tool.type === "LookAtHuman" && !tool.captured_image);
  const showImageGeneration = tool.type === "GenerateImage";
  const capturedImageUrl =
    tool.type === "LookAtHuman" ? tool.captured_image : null;

  return (
    <FullScreen>
      {showVideoFeed && (
        <>
          <VideoFeed className="h-full w-full" stream={stream} />
          <UserTracking
            className="absolute h-full w-full"
            circles={tracking_data}
          />
          {tool.type === "LookAtHuman" && (
            <Crosshair className="absolute h-full w-full" />
          )}
          {showWakeUp && <WakeUpAnimation className="absolute h-full w-full" />}
          <Binoculars className="absolute h-full w-full" />
          <Countdown
            className="absolute h-full w-full"
            timestamp={countdown_started_at}
            duration={countdown_duration ?? 0}
          />
        </>
      )}
      {showImageGeneration && (
        <ImageGeneration
          className="h-full w-full"
          capturedImageUrl={tool.captured_image}
          generatedImageUrl={tool.generated_image}
        />
      )}
      {capturedImageUrl && (
        <CapturedImage
          className="h-full w-full"
          capturedImageUrl={capturedImageUrl}
        />
      )}
      {animation && (
        <LightBorder className="absolute inset-0" animation={animation} />
      )}
      {qr_code ? (
        <div className="absolute bottom-10 w-3/4 left-1/2 -translate-x-1/2 flex flex-row items-center gap-8">
          <Transcript className="w-full" transcript={transcript} />
          <QRCode className="w-1/5 shrink-0" url={qr_code} />
        </div>
      ) : (
        <Transcript
          className="absolute bottom-10 w-3/4 left-1/2 -translate-x-1/2"
          transcript={transcript}
        />
      )}
    </FullScreen>
  );
}

export default App;
