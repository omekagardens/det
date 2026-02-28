"""Gravity equations for DET v7."""

from __future__ import annotations

import numpy as np


def relative_source(q_total: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """ρ = q - b."""
    return q_total - baseline


def lattice_corrected_kappa(kappa_grav: float, eta_lattice: float) -> float:
    """κ_eff = η * κ."""
    return eta_lattice * kappa_grav

