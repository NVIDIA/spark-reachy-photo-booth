/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import type { BoundingCircle } from "./types/api";

interface UserTrackingProps {
  className?: string;
  circles: BoundingCircle[];
}

const PRIMARY_COLOR = "#00ff00"; // tracked user
const SECONDARY_COLOR = "#c0c0c0"; // non-tracked user
const PRIMARY_THICKNESS = 12; // tracked user
const SECONDARY_THICKNESS = 2.5; // non-tracked user

// The backend sends circles coordinates assuming a video size of 1920x1080
// We need to know the video size to correctly display the circles
const VIDEO_WIDTH = import.meta.env.VITE_VIDEO_WIDTH || 1920;
const VIDEO_HEIGHT = import.meta.env.VITE_VIDEO_HEIGHT || 1080;
const RADIUS_SCALE = 1.3;

export default function UserTracking({
  circles,
  className
}: UserTrackingProps) {
  return (
    <svg
      className={className}
      viewBox={`0 0 ${VIDEO_WIDTH} ${VIDEO_HEIGHT}`}
      preserveAspectRatio="xMidYMid slice"
      xmlns="http://www.w3.org/2000/svg"
    >
      {circles.map((circle, i) => (
        <circle
          key={i}
          cx={circle.center_x * VIDEO_WIDTH}
          cy={circle.center_y * VIDEO_HEIGHT}
          r={circle.radius * VIDEO_WIDTH * RADIUS_SCALE}
          fill="none"
          stroke={circle.is_primary ? PRIMARY_COLOR : SECONDARY_COLOR}
          strokeWidth={
            circle.is_primary ? PRIMARY_THICKNESS : SECONDARY_THICKNESS
          }
          opacity={0.8}
          className="transition-all duration-100 ease-out"
        />
      ))}
    </svg>
  );
}
