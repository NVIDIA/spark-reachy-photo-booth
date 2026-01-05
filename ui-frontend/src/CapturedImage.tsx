/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";

interface CapturedImageProps {
  capturedImageUrl: string;
  className?: string;
}

export default function CapturedImage({
  capturedImageUrl,
  className
}: CapturedImageProps) {
  return (
    <img
      src={capturedImageUrl}
      className={classNames(
        "object-contain grayscale -rotate-[5deg] scale-75",
        className
      )}
    />
  );
}
