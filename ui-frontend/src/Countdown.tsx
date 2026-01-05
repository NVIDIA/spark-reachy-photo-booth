/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useEffect, useState } from "react";
import classNames from "classnames";

type CountdownProps = {
  timestamp: number | null;
  duration: number;
  className?: string;
};

export default function Countdown({
  timestamp,
  duration,
  className
}: CountdownProps) {
  const [timeLeft, setTimeLeft] = useState<number | null>(null);

  useEffect(() => {
    if (!timestamp) {
      setTimeLeft(null);
      return;
    }

    const updateTimer = () => {
      const now = Date.now() / 1000;
      const elapsed = now - timestamp;
      const remaining = duration - elapsed;

      if (remaining <= 0) {
        setTimeLeft(0);
      } else {
        setTimeLeft(remaining);
      }
    };

    updateTimer();
    const interval = setInterval(updateTimer, 100);
    return () => clearInterval(interval);
  }, [timestamp, duration]);

  if (timeLeft === null || timeLeft <= 0) {
    return null;
  }

  let content = null;
  if (timeLeft > 3) {
    content = "Get ready";
  } else if (timeLeft > 0) {
    content = Math.ceil(timeLeft).toString();
  }

  return (
    <div
      className={classNames(
        "flex justify-center text-white/60 font-bold drop-shadow-md overflow-hidden",
        timeLeft > 3
          ? "text-[15vh] pt-[1.5%] items-start"
          : "text-[50vh] items-center",
        className
      )}
    >
      {timeLeft <= 3 && (
        <div
          className="absolute w-96 h-96 border-white/50 border-4 rounded-full animate-ping duration-1000"
          // The "ping" animation and the "3,2,1" changes last both for one second, so they should normally
          // be in sync. However, due to the slight delay in the animation, we need to add a small offset
          // to avoid visual glitches
          style={{ animationDelay: "20ms" }}
        />
      )}
      {content}
    </div>
  );
}
