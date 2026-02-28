"""Structure and debt updates for DET v7."""

from __future__ import annotations

import numpy as np


def update_qd_from_loss(q_d: np.ndarray, dF: np.ndarray, alpha_qd: float) -> np.ndarray:
    """(q_D)^+ = clip(q_D + α_qD * max(0, -ΔF), 0, 1)."""
    return np.clip(q_d + alpha_qd * np.maximum(0.0, -dF), 0.0, 1.0)


def update_qi_from_loss(q_i: np.ndarray, dF: np.ndarray, alpha_qi: float) -> np.ndarray:
    """Optional identity locking law. Canonical default sets α_qI = 0."""
    return np.clip(q_i + alpha_qi * np.maximum(0.0, -dF), 0.0, 1.0)


def combine_debt(q_i: np.ndarray, q_d: np.ndarray) -> np.ndarray:
    """Legacy total q readout."""
    return np.clip(q_i + q_d, 0.0, 1.0)


def apply_jubilee(
    q_d: np.ndarray,
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
    """Apply Jubilee to q_D only."""
    activation = a * (C ** n_q) * (D / (D + D_0))
    dq = delta_q * activation * delta_tau
    if energy_coupled:
        F_op = np.maximum(F - F_vac, 0.0)
        energy_cap = F_op / (1.0 + F_op)
        dq = np.minimum(dq, energy_cap)
    dq = np.minimum(dq, q_d)
    return np.clip(q_d - dq, 0.0, 1.0)

