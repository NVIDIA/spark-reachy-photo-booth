/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useEffect } from "react";

interface TextCarouselProps {
  texts: string[];
  interval?: number;
  className?: string;
}

export default function TextCarousel({
  texts,
  interval = 3000,
  className
}: TextCarouselProps) {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (texts.length === 0) return;

    const timer = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % texts.length);
    }, interval);

    return () => clearInterval(timer);
  }, [texts.length, interval]);

  if (texts.length === 0) return null;

  return <div className={className}>{texts[currentIndex]}</div>;
}

