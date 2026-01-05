# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import Field
from workmesh.config import BaseConfig


class MyServiceConfig(BaseConfig):
    my_super_field: float = Field(ge=1.0, lt=2.0, default=1.5)
