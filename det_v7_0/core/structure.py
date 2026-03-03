"""Structure and debt updates for DET v7.x unified mutable-q branch."""

from __future__ import annotations

import numpy as np


def update_q_from_loss(q: np.ndarray, dF: np.ndarray, alpha_q: float) -> np.ndarray:
    """q^+ = clip(q + alpha_q * max(0, -dF), 0, 1)."""
    return np.clip(q + alpha_q * np.maximum(0.0, -dF), 0.0, 1.0)


def apply_jubilee(
    q: np.ndarray,
    a: np.ndarray,
    C: np.ndarray,
    D: np.ndarray,
    delta_tau: np.ndarray,
    delta_q: float,
    n_q: float,
    D_0: float,
    F: np.ndarray,
    F_vac: float,
    energy_coupled: bool = True,
) -> np.ndarray:
    """Apply Jubilee to total q (mutable under lawful local recovery)."""
    activation = a * (C ** n_q) * (D / (D + D_0))
    dq = delta_q * activation * delta_tau
    if energy_coupled:
        F_op = np.maximum(F - F_vac, 0.0)
        energy_cap = F_op / (1.0 + F_op)
        dq = np.minimum(dq, energy_cap)
    dq = np.minimum(dq, q)
    return np.clip(q - dq, 0.0, 1.0)
