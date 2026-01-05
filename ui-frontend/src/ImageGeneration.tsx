/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classnames from "classnames";
import { useMemo } from "react";
import { usePixelatedImage } from "./hooks/usePixelatedImage";
import TextCarousel from "./TextCarousel";
import { shuffleArray } from "./utils/shuffle";

interface ImageGenerationProps {
  capturedImageUrl: string;
  generatedImageUrl: string | null;
  className?: string;
}

const TEXTS = [
  "I'm creating your image with black-forest-labs/FLUX.1-Kontext-dev...",
  "My ears? That's nvidia/riva-parakeet-ctc-1.1B listening!",
  "I'm listening with nvidia/riva-parakeet-ctc-1.1B",
  "My voice? That's hexgrad/Kokoro-82M speaking!",
  "I'm speaking with hexgrad/Kokoro-82M",
  "My brain runs on openai/gpt-oss-20b!",
  "I'm thinking with openai/gpt-oss-20b",
  "My eyes use facebookresearch/detectron2!",
  "I'm observing you with facebookresearch/detectron2",
  "I'm following your moves with ByteTrack!"
];

export default function ImageGeneration({
  capturedImageUrl,
  generatedImageUrl,
  className
}: ImageGenerationProps) {
  const canvasRef = usePixelatedImage(capturedImageUrl);
  const shuffledTexts = useMemo(() => shuffleArray(TEXTS), []);

  return (
    <div className={classnames("relative", className)}>
      {generatedImageUrl ? (
        <img
          src={generatedImageUrl}
          className={classnames("object-cover", className)}
        />
      ) : (
        <>
          <canvas
            ref={canvasRef}
            className={classnames(
              "object-cover opacity-30 [image-rendering:pixelated]",
              className
            )}
          />
          <div className="absolute inset-0 flex flex-col items-center justify-center text-white [text-shadow:0_2px_8px_rgba(0,0,0,0.8)]">
            <div className="text-7xl font-bold">Generating image...</div>
            <TextCarousel
              texts={shuffledTexts}
              className="text-4xl mt-4"
            />
          </div>
        </>
      )}
    </div>
  );
}
