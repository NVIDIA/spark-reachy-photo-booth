/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { useState, useRef, useEffect } from "react";

const canvasDimensions = [5, 6, 7, 8, 9, 10, 11, 12];

export function usePixelatedImage(imageUrl: string | null) {
  const [pixelIndex, setPixelIndex] = useState(0);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setPixelIndex((prev) => (prev + 1) % canvasDimensions.length);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    if (!imageRef.current || imageRef.current.src !== imageUrl) {
      const img = new Image();
      if (imageUrl) {
        img.src = imageUrl;
      }
      imageRef.current = img;

      img.onload = () => {
        drawPixelated();
      };
    } else {
      drawPixelated();
    }

    function drawPixelated() {
      if (!canvas || !ctx || !imageRef.current) return;

      const dimension = canvasDimensions[pixelIndex];
      const img = imageRef.current;

      const aspectRatio = img.width / img.height;
      const canvasWidth = Math.round(dimension * aspectRatio);
      const canvasHeight = dimension;

      canvas.width = canvasWidth;
      canvas.height = canvasHeight;

      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(img, 0, 0, canvasWidth, canvasHeight);
    }
  }, [pixelIndex, imageUrl]);

  return canvasRef;
}
