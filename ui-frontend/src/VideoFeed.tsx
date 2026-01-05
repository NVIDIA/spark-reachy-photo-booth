/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";
import { useEffect, useRef } from "react";

interface VideoFeedProps {
  className?: string;
  stream: MediaStream | null;
}

export default function VideoFeed({ className, stream }: VideoFeedProps) {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (videoRef.current && stream) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  return (
    <video
      autoPlay
      playsInline
      muted
      ref={videoRef}
      className={classNames("object-cover", className)}
    />
  );
}
