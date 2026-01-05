# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


class SuggestionMessage(Exception):
    """
    Base exception class that provides a helpful suggestion message
    to guide recovery from the error.
    """

    def __init__(self, message: str, suggestion: str | None = None):
        """
        Args:
            message: The error message describing what went wrong
            suggestion: Optional suggestion on how to fix or recover from the error
        """
        self.message = message
        self.suggestion = suggestion

        full_message = message
        if suggestion:
            full_message = f"{message} Suggestion: {suggestion}"

        super().__init__(full_message)


class InvalidImageError(SuggestionMessage):
    """
    Raised when an image is required but missing, invalid, or unusable
    for image generation operations.
    """

    def __init__(self, message: str, suggestion: str | None = None):
        default_suggestion = (
            "Capture the user first using `look_at_human` and only then "
            "use `generate_image` to apply the transformation."
        )
        super().__init__(message, suggestion or default_suggestion)


class NoResponseError(SuggestionMessage):
    """
    Raised when a tool times out without receiving a response from the user.
    """

    def __init__(self, message: str, suggestion: str | None = None):
        default_suggestion = (
            "The user did not respond to the prompt in time. "
            "Use the `farewell_user` tool to end the conversation."
        )
        super().__init__(message, suggestion or default_suggestion)
