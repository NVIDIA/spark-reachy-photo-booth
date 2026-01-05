/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";
import type { StaticAnimation, COLOR } from "./types/api";

interface AnimationProps {
  animation: StaticAnimation;
  className?: string;
}

const COLORS: Record<COLOR, string> = {
  TRANSPARENT: "transparent",
  GRAY: "rgb(128, 128, 128)",
  WHITE: "rgb(255, 255, 255)",
  BLUE: "#004080",
  INTENSE_BLUE: "#0080FF",
  GREEN: "#006600",
  INTENSE_GREEN: "#00C800",
  RED: "#670000",
  INTENSE_RED: "#EB0000"
};

export default function StaticBorder({ animation, className }: AnimationProps) {
  const color = COLORS[animation.color] || "transparent";
  const isIntense = animation.color.startsWith("INTENSE");
  const transitionDuration = `${animation.in_transition}s`;

  const borderLayer = (
    <div
      className="absolute inset-0 w-full h-full border-[0.3vw] transition-colors"
      style={{
        borderColor: color,
        boxShadow: `inset 0 0 0.2vw 0.15vw ${color}`,
        transitionDuration
      }}
    />
  );

  const shadowLayer = (
    <div
      className="absolute inset-0 w-full h-full transition-colors mix-blend-plus-lighter"
      style={{
        boxShadow: `inset 0 0 3vw 1vw ${color}`,
        transitionDuration
      }}
    />
  );

  const extraGlowLayer = (
    <div
      className={classNames(
        "absolute inset-0 w-full h-full transition-opacity mix-blend-plus-lighter",
        {
          "opacity-100": isIntense,
          "opacity-0": !isIntense
        }
      )}
      style={{
        boxShadow: "inset 0 0 0.6vw 0.05vw #FFF7CC80",
        transitionDuration
      }}
    />
  );

  return (
    <div className={classNames("absolute inset-0", className)}>
      {shadowLayer}
      {borderLayer}
      {extraGlowLayer}
    </div>
  );
}
