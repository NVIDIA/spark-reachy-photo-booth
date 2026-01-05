/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";
import crosshairImg from "./assets/camera-crosshair.png";

interface CrosshairProps {
  className?: string;
}

export default function Crosshair({ className }: CrosshairProps) {
  return (
    <img
      src={crosshairImg}
      className={classNames("object-contain opacity-50 scale-85", className)}
      alt="Crosshair"
    />
  );
}
