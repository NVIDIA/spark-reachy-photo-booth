# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Easing functions for animations. All functions map [0,1] -> [0,1].

Based on: https://easings.net/
"""

import math


# Linear
def linear(x: float) -> float:
    return x


# Sine
def ease_in_sine(x: float) -> float:
    return 1 - math.cos((x * math.pi) / 2)


def ease_out_sine(x: float) -> float:
    return math.sin((x * math.pi) / 2)


def ease_in_out_sine(x: float) -> float:
    return -(math.cos(math.pi * x) - 1) / 2


# Quad
def ease_in_quad(x: float) -> float:
    return x * x


def ease_out_quad(x: float) -> float:
    return 1 - (1 - x) * (1 - x)


def ease_in_out_quad(x: float) -> float:
    return 2 * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 2) / 2


# Cubic
def ease_in_cubic(x: float) -> float:
    return x * x * x


def ease_out_cubic(x: float) -> float:
    return 1 - math.pow(1 - x, 3)


def ease_in_out_cubic(x: float) -> float:
    return 4 * x * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 3) / 2


# Quart
def ease_in_quart(x: float) -> float:
    return x * x * x * x


def ease_out_quart(x: float) -> float:
    return 1 - math.pow(1 - x, 4)


def ease_in_out_quart(x: float) -> float:
    return 8 * x * x * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 4) / 2


# Quint
def ease_in_quint(x: float) -> float:
    return x * x * x * x * x


def ease_out_quint(x: float) -> float:
    return 1 - math.pow(1 - x, 5)


def ease_in_out_quint(x: float) -> float:
    return 16 * x * x * x * x * x if x < 0.5 else 1 - math.pow(-2 * x + 2, 5) / 2


# Expo
def ease_in_expo(x: float) -> float:
    return 0 if x == 0 else math.pow(2, 10 * x - 10)


def ease_out_expo(x: float) -> float:
    return 1 if x == 1 else 1 - math.pow(2, -10 * x)


def ease_in_out_expo(x: float) -> float:
    if x == 0:
        return 0
    if x == 1:
        return 1
    return (
        math.pow(2, 20 * x - 10) / 2 if x < 0.5 else (2 - math.pow(2, -20 * x + 10)) / 2
    )


# Circ
def ease_in_circ(x: float) -> float:
    return 1 - math.sqrt(1 - math.pow(x, 2))


def ease_out_circ(x: float) -> float:
    return math.sqrt(1 - math.pow(x - 1, 2))


def ease_in_out_circ(x: float) -> float:
    return (
        (1 - math.sqrt(1 - math.pow(2 * x, 2))) / 2
        if x < 0.5
        else (math.sqrt(1 - math.pow(-2 * x + 2, 2)) + 1) / 2
    )


# Back
def ease_in_back(x: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return c3 * x * x * x - c1 * x * x


def ease_out_back(x: float) -> float:
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * math.pow(x - 1, 3) + c1 * math.pow(x - 1, 2)


def ease_in_out_back(x: float) -> float:
    c1 = 1.70158
    c2 = c1 * 1.525
    return (
        (math.pow(2 * x, 2) * ((c2 + 1) * 2 * x - c2)) / 2
        if x < 0.5
        else (math.pow(2 * x - 2, 2) * ((c2 + 1) * (x * 2 - 2) + c2) + 2) / 2
    )


# Elastic
def ease_in_elastic(x: float) -> float:
    c4 = (2 * math.pi) / 3
    if x == 0:
        return 0
    if x == 1:
        return 1
    return -math.pow(2, 10 * x - 10) * math.sin((x * 10 - 10.75) * c4)


def ease_out_elastic(x: float) -> float:
    c4 = (2 * math.pi) / 3
    if x == 0:
        return 0
    if x == 1:
        return 1
    return math.pow(2, -10 * x) * math.sin((x * 10 - 0.75) * c4) + 1


def ease_in_out_elastic(x: float) -> float:
    c5 = (2 * math.pi) / 4.5
    if x == 0:
        return 0
    if x == 1:
        return 1
    return (
        -(math.pow(2, 20 * x - 10) * math.sin((20 * x - 11.125) * c5)) / 2
        if x < 0.5
        else (math.pow(2, -20 * x + 10) * math.sin((20 * x - 11.125) * c5)) / 2 + 1
    )


# Bounce
def ease_in_bounce(x: float) -> float:
    return 1 - ease_out_bounce(1 - x)


def ease_out_bounce(x: float) -> float:
    n1 = 7.5625
    d1 = 2.75
    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * x * x + 0.984375


def ease_in_out_bounce(x: float) -> float:
    return (
        (1 - ease_out_bounce(1 - 2 * x)) / 2
        if x < 0.5
        else (1 + ease_out_bounce(2 * x - 1)) / 2
    )
