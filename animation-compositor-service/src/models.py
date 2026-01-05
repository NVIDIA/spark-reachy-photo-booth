# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Self

from pydantic import BaseModel, Field, model_validator


class Range(BaseModel):
    min: float = Field(default=-180)
    max: float = Field(default=180)

    @model_validator(mode="before")
    def convert_list_to_dict(cls, value):
        """Convert list format [min, max] to dict format."""
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError(
                    f"Range list must have exactly 2 elements, got {len(value)}"
                )
            return {"min": value[0], "max": value[1]}
        return value

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.min > self.max:
            raise ValueError(f"Invalid range: {self.min} > {self.max}")
        return self

    def clamp(self, value: float) -> float:
        return max(self.min, min(value, self.max))
