/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import StaticBorder from "./StaticBorder";
import FillCircleBorder from "./FillCircleBorder";
import type { Animation } from "./types/api";

interface LightBorderProps {
  animation: Animation;
  className?: string;
}

export default function LightBorder({
  animation,
  className
}: LightBorderProps) {
  if (animation.type === "Static") {
    return <StaticBorder animation={animation} className={className} />;
  } else if (animation.type === "FillCircle") {
    return <FillCircleBorder animation={animation} className={className} />;
  }
}
