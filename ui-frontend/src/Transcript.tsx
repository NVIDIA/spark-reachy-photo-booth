/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classnames from "classnames";
import { useEffect, useState } from "react";
import type { Transcript } from "./types/api";

interface TranscriptProps {
  className?: string;
  transcript: Transcript | null;
}

const COLORS = {
  User: "#80FF80", // INTENSE_GREEN
  Bot: "#80BFFF" // INTENSE_BLUE
};

const WORDS_PER_SECOND = 2.3;

function getScrollDuration(text: string | undefined) {
  if (!text) return 0;
  const wordCount = text.trim().split(/\s+/).length;
  return wordCount / WORDS_PER_SECOND;
}

export default function Transcript({ className, transcript }: TranscriptProps) {
  const [displayedTranscript, setDisplayedTranscript] =
    useState<Transcript | null>(null);

  useEffect(() => {
    if (transcript) {
      setDisplayedTranscript(transcript);
    } else {
      // clear the displayed transcript after the fade out animation
      const timer = setTimeout(() => {
        setDisplayedTranscript(null);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [transcript]);

  const scrollDuration = getScrollDuration(displayedTranscript?.text);

  const opacity = transcript
    ? "opacity-100 duration-0"
    : "opacity-0 duration-1000";

  return (
    <div
      style={
        {
          color: displayedTranscript
            ? COLORS[displayedTranscript.author]
            : COLORS.User,
          WebkitTextStroke: "0.2vh rgba(0, 0, 0, 0.5)"
        } as React.CSSProperties
      }
      className={classnames(
        "font-bold text-[4.5vh] transition-opacity overflow-hidden h-[3.6em] leading-[1.2em]",
        "mask-[linear-gradient(to_bottom,transparent_0%,black_5%,black_95%,transparent_100%)]",
        opacity,
        className
      )}
    >
      <div
        key={displayedTranscript?.text}
        className={classnames({
          "animate-scroll-transcript": scrollDuration > 0
        })}
        style={{ animationDuration: `${scrollDuration}s` }}
      >
        {displayedTranscript?.text}
      </div>
    </div>
  );
}
