# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass

from workmesh.messages import (
    Color,
    FillCircleAnimation,
    LightCommand,
    StaticAnimation,
)

from workmesh import Service, light_command_topic


@dataclass()
class LightEffect:
    animation: type[StaticAnimation] | type[FillCircleAnimation]

    primary_color: Color
    secondary_color: Color = Color.TRANSPARENT

    in_transition_duration: float = 0.0
    fill_duration: float = 10.0

    priority: int = 0


class LightManager:
    def __init__(self, service: Service):
        self._service = service
        self._logger = service.logger

        self._light_effects: dict[str, LightEffect] = {}
        self._priority_effects: dict[int, str] = {}

    #########################################################
    # Light functions
    #########################################################

    async def light_on(self, name: str, light_effect: LightEffect) -> None:
        """
        Turn on the lights using the specified light effect.

        Args:
            name (str): Name identifier for this light effect.
            light_effect (LightEffect): Contains animation type, colors, durations,
                transition information, and priority.
        """
        # Store the current top-priority light effect
        previous_effect = self._get_top_effect()

        # Add new light effect
        self._add_effect(name, light_effect)

        # If the top effect has changed, send an updated light command
        await self._send_if_changed(previous_effect)

    async def light_off(self, name: str) -> None:
        """
        Turn off the light effect with the specified name.

        Args:
            name (str): Name identifier of the light effect to remove.
        """
        # Store the current top-priority light effect
        previous_effect = self._get_top_effect()

        # Remove light effect
        self._remove_effect(name)

        # If the top effect has changed, send an updated light command
        await self._send_if_changed(previous_effect)

    async def light_blink(self, light_effect: LightEffect, duration: float = 1) -> None:
        """
        Temporarily turn on the lights for a specified duration using the given
        light effect, then revert to the previous top-priority effect.

        Args:
            light_effect (LightEffect): Contains animation type, colors, durations,
                and transition information.
            duration (float, optional): Number of seconds to keep the light on before
                ending the blink. Defaults to 1.
        """

        # Turn on the new light effect
        await self._send_light_command(light_effect)

        # Wait for the duration
        await asyncio.sleep(duration)

        # Turn off the previous light effect
        top_effect = self._get_top_effect()
        if top_effect is not None:
            await self._send_light_command(top_effect)

    async def _send_if_changed(self, previous_effect: LightEffect | None) -> None:
        """Send the current top effect only if it differs from the previous effect."""
        top_effect = self._get_top_effect()
        if top_effect is not None and top_effect != previous_effect:
            await self._send_light_command(top_effect)

    async def _send_light_command(self, light_effect: LightEffect) -> None:
        """Send a light command to control the robot's lights.

        Args:
            light_effect: LightEffect instance containing animation type,
                primary color, transition durations, fill and secondary color.
        """
        try:
            if light_effect.animation is FillCircleAnimation:
                command = LightCommand(
                    fill_circle_animation=FillCircleAnimation(
                        primary_color=light_effect.primary_color,
                        secondary_color=light_effect.secondary_color,
                        in_transition_duration=light_effect.in_transition_duration,
                        fill_duration=light_effect.fill_duration,
                    )
                )
            elif light_effect.animation is StaticAnimation:
                command = LightCommand(
                    static_animation=StaticAnimation(
                        color=light_effect.primary_color,
                        in_transition_duration=light_effect.in_transition_duration,
                    )
                )
            else:
                self._logger.warning(
                    f"[LIGHT MANAGER] Unknown animation type: {light_effect.animation}"
                )
                return

            await self._service.publish(light_command_topic, command)
            secondary_color_name = (
                Color.Name(light_effect.secondary_color)
                if light_effect.secondary_color != Color.TRANSPARENT
                else "None"
            )
            self._logger.debug(
                f"[LIGHT MANAGER] Sent light command: "
                f"animation={light_effect.animation.__name__}, "
                f"primary_color={Color.Name(light_effect.primary_color)}, "
                f"secondary_color={secondary_color_name}, "
                f"fill_duration={light_effect.fill_duration}, "
                f"in_transition={light_effect.in_transition_duration}"
            )
        except Exception:
            self._logger.exception("Failed to send light command")

    #########################################################
    # Management functions
    #########################################################

    def _add_effect(self, name: str, light_effect: LightEffect) -> None:
        # Replace existing effect at the same priority
        if light_effect.priority in self._priority_effects:
            old_light_name = self._priority_effects.pop(light_effect.priority)
            self._light_effects.pop(old_light_name, None)

        # Add new effect
        self._priority_effects[light_effect.priority] = name
        self._light_effects[name] = light_effect
        self._logger.debug(
            f"[LIGHT MANAGER] Added light effect '{name}' "
            f"with priority {light_effect.priority}"
        )

    def _remove_effect(self, name: str) -> None:
        effect = self._light_effects.pop(name, None)
        if effect is not None:
            self._priority_effects.pop(effect.priority, None)
            self._logger.debug(
                f"[LIGHT MANAGER] Removed light effect '{name}' "
                f"with priority {effect.priority}"
            )

    def _get_top_effect(self) -> LightEffect | None:
        if not self._priority_effects:
            return None

        # Get the top priority effect
        top_priority = min(self._priority_effects.keys())
        top_light_name = self._priority_effects[top_priority]
        return self._light_effects[top_light_name]
