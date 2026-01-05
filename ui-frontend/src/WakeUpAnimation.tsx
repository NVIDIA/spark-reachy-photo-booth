/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";
import { useEffect, useState } from "react";
import { wait } from "./utils/wait";

export default function WakeUpAnimation({ className }: { className: string }) {
  const [isOpen, setIsOpen] = useState(false);
  const [blurClass, setBlurClass] = useState("backdrop-blur-2xl");

  useEffect(() => {
    const controller = new AbortController();
    const signal = controller.signal;

    const runSequence = async () => {
      try {
        // Initial open
        await wait(100, signal);
        setIsOpen(true);

        // First Blink
        await wait(1000, signal);
        setIsOpen(false);
        setBlurClass("backdrop-blur-md"); // reduce blur

        await wait(50, signal);
        setIsOpen(true);

        // Second Blink
        await wait(500, signal);
        setIsOpen(false);
        setBlurClass("backdrop-blur-sm"); // reduce blur

        await wait(50, signal);
        setIsOpen(true);

        // Third Blink
        await wait(100, signal);
        setIsOpen(false);
        setBlurClass("backdrop-blur-none"); // remove blur

        await wait(50, signal);
        setIsOpen(true);
      } catch (error) {
        if ((error as Error).name !== "AbortError") {
          throw error;
        }
        // Ignore AbortError
      }
    };

    runSequence();

    return () => {
      controller.abort();
    };
  }, []);

  return (
    <div
      className={classNames(className, "absolute overflow-hidden", blurClass)}
    >
      <div
        className={classNames(
          "bg-linear-to-b from-black to-transparent h-1/2 from-85% to-95%",
          "transition-transform ease-in-out",
          isOpen
            ? "duration-100 -translate-y-full"
            : "duration-50 translate-y-0"
        )}
      />
      <div
        className={classNames(
          "bg-linear-to-b from-transparent to-black h-1/2 from-5% to-15%",
          "transition-transform ease-in-out",
          isOpen ? "duration-100 translate-y-full" : "duration-50 translate-y-0"
        )}
      />
    </div>
  );
}
