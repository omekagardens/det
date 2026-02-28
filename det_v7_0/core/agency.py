"""Agency laws for DET v7."""

from __future__ import annotations

import numpy as np


def coherence_gated_drive(
    P: np.ndarray,
    P_neighbor_mean: np.ndarray,
    C: np.ndarray,
    gamma_max: float,
    n: float,
) -> np.ndarray:
    """Δa_drive = γ(C) * (P - P̄_neighbors), γ(C)=γ_max*C^n."""
    gamma = gamma_max * (C ** n)
    return gamma * (P - P_neighbor_mean)


def update_agency(
    a: np.ndarray,
    a0: float,
    beta_a: float,
    drive: np.ndarray,
    noise: np.ndarray | float = 0.0,
) -> np.ndarray:
    """a⁺ = clip(a + β_a(a0-a) + drive + ξ, 0, 1)."""
    updated = a + beta_a * (a0 - a) + drive + noise
    return np.clip(updated, 0.0, 1.0)

