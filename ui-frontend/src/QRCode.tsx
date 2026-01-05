/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";

interface QRCodeProps {
  url: string;
  className?: string;
}

export default function QRCode({ url, className }: QRCodeProps) {
  return (
    <div className={classNames("bg-black p-2 rounded-lg shadow-lg", className)}>
      <img src={url} alt="QR Code" className="invert-100" />
    </div>
  );
}
