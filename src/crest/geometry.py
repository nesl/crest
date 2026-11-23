# Copyright (c) 2026 UCLA Networked & Embedded Systems Laboratory
# SPDX-License-Identifier: BSD-3-Clause
"""Preserve legacy geometry imports; use :mod:`crest.utils.geometry` in new code."""

from .utils.geometry import (
    euler2mat,
    euler2quat,
    mat2euler,
    mat2quat,
    quat2euler,
    quat2mat,
)

__all__ = ["euler2mat", "euler2quat", "mat2euler", "mat2quat", "quat2euler", "quat2mat"]
