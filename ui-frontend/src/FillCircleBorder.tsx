/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useRef } from "react";
import StaticBorder from "./StaticBorder";
import classNames from "classnames";
import type { FillCircleAnimation } from "./types/api";

interface FillCircleBorderProps {
  animation: FillCircleAnimation;
  className?: string;
}

export default function FillCircleBorder({
  animation,
  className
}: FillCircleBorderProps) {
  const primaryRef = useRef<HTMLDivElement>(null);
  const secondaryRef = useRef<HTMLDivElement>(null);

  // Animate the conic gradients so that the primary color fills and the secondary color empties
  useEffect(() => {
    let animationFrameId: number;
    const startTime = performance.now();
    const durationMs = animation.duration * 1000;

    const update = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / durationMs, 1);
      const angle = progress * 360;

      if (primaryRef.current) {
        const mask = `conic-gradient(black 0deg ${angle}deg, transparent ${angle}deg)`;
        primaryRef.current.style.maskImage = mask;
      }

      if (secondaryRef.current) {
        const mask = `conic-gradient(transparent 0deg ${angle}deg, black ${angle}deg)`;
        secondaryRef.current.style.maskImage = mask;
      }

      if (progress < 1) {
        animationFrameId = requestAnimationFrame(update);
      }
    };

    animationFrameId = requestAnimationFrame(update);

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [animation.duration]);

  return (
    <>
      <div
        ref={secondaryRef}
        className={classNames("absolute inset-0", className)}
        style={{
          maskImage: "conic-gradient(transparent 0deg 0deg, black 0deg)"
        }}
      >
        <StaticBorder
          animation={{
            type: "Static",
            color: animation.secondary_color,
            in_transition: animation.in_transition
          }}
        />
      </div>
      <div
        ref={primaryRef}
        className={classNames("absolute inset-0", className)}
        style={{
          maskImage: "conic-gradient(black 0deg 0deg, transparent 0deg)"
        }}
      >
        <StaticBorder
          animation={{
            type: "Static",
            color: animation.primary_color,
            in_transition: animation.in_transition
          }}
        />
      </div>
    </>
  );
}
