/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";
import binocularsImg from "./assets/binoculars.svg";

interface BinocularsProps {
  className?: string;
}

export default function Binoculars({ className }: BinocularsProps) {
  return (
    <img
      src={binocularsImg}
      className={classNames("object-cover", className)}
      alt="Binoculars"
    />
  );
}
