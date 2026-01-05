# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import random
from pathlib import Path

import yaml


class RobotUtteranceManager:
    def __init__(self, utterances_path: Path, logger: logging.Logger):
        self._logger = logger
        self._utterances = self._load_utterances(utterances_path)
        self._logger.info("Loaded robot utterances.")

    def _load_utterances(self, utterances_path: Path) -> dict:
        try:
            return yaml.safe_load(utterances_path.read_text())
        except Exception as e:
            raise ValueError(
                f"Failed to load utterances from '{utterances_path}': {e}"
            ) from e

    def get_robot_utterance(
        self, category: str, utterance_type: str | None = None, index: int | None = None
    ) -> str | None:
        """
        Retrieve a robot utterance message for a given category and utterance type.

        Args:
            category (str): The category to get the utterance for
                (e.g., "look_at_human", "generate_image").
            utterance_type (str, optional): The type/status of the utterance
                (e.g., "started", "completed"). For utterances without sub-categories,
                omit this parameter.
            index (int | None, optional): The index of the utterance to retrieve.
                If None or out of bounds, a random utterance is chosen.

        Returns:
            str | None: The selected utterance message, or None if not found.
        """
        # Ignore categories that are not in the utterances
        if category not in self._utterances:
            self._logger.warning(f"No utterances found for '{category}'")
            return None

        utterances_entry = self._utterances[category]

        # If the utterances for a category are organized in sub-keys (dict)
        if isinstance(utterances_entry, dict):
            if not utterance_type:
                self._logger.warning(
                    f"Missing required utterance_type for utterances under '{category}'"
                )
                return None
            utterances = utterances_entry.get(utterance_type)
            if utterances is None:
                self._logger.warning(
                    f"No utterances available for '{category}' with utterance_type "
                    f"'{utterance_type}'"
                )
                return None
        else:
            # For cases where utterances are just a list
            utterances = utterances_entry

        # If no index is provided, or the index is out of bounds, choose a random index
        if index is None or index >= len(utterances) or index < 0:
            index = random.randint(0, len(utterances) - 1)

        return utterances[index]
