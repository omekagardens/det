"""Boundary operators for DET v7."""

from __future__ import annotations

import numpy as np


def grace_injection(
    D: np.ndarray,
    F: np.ndarray,
    a: np.ndarray,
    F_min_grace: float,
    local_sum_fn,
    radius: int,
) -> np.ndarray:
    """Local grace injection used by collider boundary modules."""
    need = np.maximum(0.0, F_min_grace - F)
    weight = a * need
    denom = local_sum_fn(weight, radius) + 1e-12
    return D * weight / denom

