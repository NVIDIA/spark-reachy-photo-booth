/**
 * Copyright 2025 NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

type COLOR =
  | "TRANSPARENT"
  | "GRAY"
  | "WHITE"
  | "BLUE"
  | "INTENSE_BLUE"
  | "GREEN"
  | "INTENSE_GREEN"
  | "RED"
  | "INTENSE_RED";

type BoundingCircle = {
  center_x: number;
  center_y: number;
  radius: number;
  is_primary: boolean;
};

export type Transcript = {
  text: string;
  author: "Bot" | "User";
};

type StaticAnimation = {
  type: "Static";
  color: COLOR;
  in_transition: number;
};

type FillCircleAnimation = {
  type: "FillCircle";
  primary_color: COLOR;
  secondary_color: COLOR;
  in_transition: number;
  duration: number;
};

type AskHuman = {
  type: "AskHuman";
};

type GreetUser = {
  type: "GreetUser";
};

type LookAtHuman = {
  type: "LookAtHuman";
  captured_image: string | null;
};

type GenerateImage = {
  type: "GenerateImage";
  captured_image: string;
  generated_image: string | null;
};

type Tool = AskHuman | LookAtHuman | GenerateImage | GreetUser;
type Animation = StaticAnimation | FillCircleAnimation;

export type AppState = {
  message_type: "AppState";
  transcript: Transcript | null;
  animation: Animation | null;
  tool: Tool | null;
  tracking_data: BoundingCircle[];
  qr_code: string | null;
  countdown_started_at: number | null;
  countdown_duration: number | null;
};

export type Signalling = {
  message_type: "Signalling";
  type: "offer" | "answer";
  sdp: string;
};

export type AbortMessage = {
  message_type: "AbortMessage";
};

export type BaseMessage = AppState | Signalling | AbortMessage;
