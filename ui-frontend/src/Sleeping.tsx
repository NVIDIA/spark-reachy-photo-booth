/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classnames from "classnames";
import FloatingImages from "./FloatingImages";
import mic from "./assets/mic.png";

export default function Sleeping({ className }: { className: string }) {
  return (
    <div
      className={classnames(
        "flex flex-col items-center justify-center relative overflow-hidden",
        className
      )}
    >
      <FloatingImages className="absolute top-0 left-0 z-0" />
      <img src={mic} className="h-1/3 opacity-80 mb-[2vh]" />
      <p className="text-[4.5vh] text-center z-10 leading-[1em] mb-[2vh]">
        Wake up Reachy
        <br />
        with the mic
      </p>
    </div>
  );
}
