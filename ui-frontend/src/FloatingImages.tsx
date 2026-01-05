/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import classNames from "classnames";
import { useMemo } from "react";
import preGeneratedImage1 from "./assets/pre-generated-images/3D Animation Character.jpg";
import preGeneratedImage2 from "./assets/pre-generated-images/3D Cartoon.png";
import preGeneratedImage3 from "./assets/pre-generated-images/3D Character.jpg";
import preGeneratedImage4 from "./assets/pre-generated-images/American Comic Book.png";
import preGeneratedImage5 from "./assets/pre-generated-images/Anime.png";
import preGeneratedImage6 from "./assets/pre-generated-images/Art Nouveu.png";
import preGeneratedImage7 from "./assets/pre-generated-images/Cartoon.jpg";
import preGeneratedImage8 from "./assets/pre-generated-images/Character Concept Art.png";
import preGeneratedImage9 from "./assets/pre-generated-images/Chibi.png";
import preGeneratedImage10 from "./assets/pre-generated-images/Children's Book Illustration.png";
import preGeneratedImage11 from "./assets/pre-generated-images/Cinematic Photography.png";
import preGeneratedImage12 from "./assets/pre-generated-images/Clay Animation.png";
import preGeneratedImage13 from "./assets/pre-generated-images/Comic Book.png";
import preGeneratedImage14 from "./assets/pre-generated-images/Doll.png";
import preGeneratedImage15 from "./assets/pre-generated-images/Embroidery.png";
import preGeneratedImage16 from "./assets/pre-generated-images/Film Noir.png";
import preGeneratedImage17 from "./assets/pre-generated-images/Gothic.png";
import preGeneratedImage18 from "./assets/pre-generated-images/Graphic Novel.png";
import preGeneratedImage19 from "./assets/pre-generated-images/HDR Photography.png";
import preGeneratedImage20 from "./assets/pre-generated-images/High Fantasy.png";
import preGeneratedImage21 from "./assets/pre-generated-images/Impressionist.jpg";
import preGeneratedImage22 from "./assets/pre-generated-images/Infographic.png";
import preGeneratedImage23 from "./assets/pre-generated-images/Manga.jpg";
import preGeneratedImage24 from "./assets/pre-generated-images/Marble Statue.jpg";
import preGeneratedImage25 from "./assets/pre-generated-images/Minimalist.png";
import preGeneratedImage26 from "./assets/pre-generated-images/Neon Comic Book.png";
import preGeneratedImage27 from "./assets/pre-generated-images/Oil Painting.png";
import preGeneratedImage28 from "./assets/pre-generated-images/Paper Craft.png";
import preGeneratedImage29 from "./assets/pre-generated-images/Paper Doll.png";
import preGeneratedImage30 from "./assets/pre-generated-images/Pencil Sketch.png";
import preGeneratedImage31 from "./assets/pre-generated-images/Pixel Art.png";
import preGeneratedImage32 from "./assets/pre-generated-images/Plush Toy.png";
import preGeneratedImage33 from "./assets/pre-generated-images/Pop Art.png";
import preGeneratedImage34 from "./assets/pre-generated-images/Steampunk.png";
import preGeneratedImage35 from "./assets/pre-generated-images/Sticker Art.png";
import preGeneratedImage36 from "./assets/pre-generated-images/Street Art.png";
import preGeneratedImage37 from "./assets/pre-generated-images/Tattoo Art.png";
import preGeneratedImage38 from "./assets/pre-generated-images/Wanted Poster.png";
import preGeneratedImage39 from "./assets/pre-generated-images/Watercolor.jpg";

const images = [
  preGeneratedImage1,
  preGeneratedImage2,
  preGeneratedImage3,
  preGeneratedImage4,
  preGeneratedImage5,
  preGeneratedImage6,
  preGeneratedImage7,
  preGeneratedImage8,
  preGeneratedImage9,
  preGeneratedImage10,
  preGeneratedImage11,
  preGeneratedImage12,
  preGeneratedImage13,
  preGeneratedImage14,
  preGeneratedImage15,
  preGeneratedImage16,
  preGeneratedImage17,
  preGeneratedImage18,
  preGeneratedImage19,
  preGeneratedImage20,
  preGeneratedImage21,
  preGeneratedImage22,
  preGeneratedImage23,
  preGeneratedImage24,
  preGeneratedImage25,
  preGeneratedImage26,
  preGeneratedImage27,
  preGeneratedImage28,
  preGeneratedImage29,
  preGeneratedImage30,
  preGeneratedImage31,
  preGeneratedImage32,
  preGeneratedImage33,
  preGeneratedImage34,
  preGeneratedImage35,
  preGeneratedImage36,
  preGeneratedImage37,
  preGeneratedImage38,
  preGeneratedImage39
];

export default function FloatingImages({ className }: { className?: string }) {
  const shuffledImages = useMemo(() => {
    return [...images].sort(() => Math.random() - 0.5);
  }, []);

  const imageConfigs = useMemo(() => {
    return shuffledImages.map((_, index) => {
      const angle = Math.random() * 7 + 3; // 3 to 10
      const sign = Math.random() > 0.5 ? 1 : -1;
      const isLeft = index % 2;

      return {
        rotation: angle * sign,
        isLeft
      };
    });
  }, [shuffledImages]);

  return (
    <div className={classNames("w-full h-full absolute", className)}>
      {shuffledImages.map((src, index) => (
        <div
          key={index}
          className={classNames("animate-float-up w-[28%] absolute", {
            "left-[8%]": imageConfigs[index].isLeft,
            "left-[64%]": !imageConfigs[index].isLeft
          })}
          style={{ animationDelay: `${index * 6.56}s` }}
        >
          <img
            src={src}
            className="w-full rounded-[1.5vw]"
            style={{
              transform: `rotate(${imageConfigs[index].rotation}deg)`,
              boxShadow: "0 0 3.5vw -0.75vw #6C6656"
            }}
          />
        </div>
      ))}
    </div>
  );
}
