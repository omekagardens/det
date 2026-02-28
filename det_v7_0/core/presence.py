"""Presence and time-rate laws for DET v7."""

from __future__ import annotations

import numpy as np


def compute_drag_factor(
    q_i: np.ndarray,
    q_d: np.ndarray,
    lambda_ip: float,
    lambda_dp: float,
) -> np.ndarray:
    """D = 1 / (1 + λ_DP q_D + λ_IP q_I)."""
    return 1.0 / (1.0 + lambda_dp * q_d + lambda_ip * q_i)


def compute_presence(
    a: np.ndarray,
    sigma: np.ndarray,
    F: np.ndarray,
    H: np.ndarray,
    gamma_v: float,
    q_i: np.ndarray,
    q_d: np.ndarray,
    lambda_ip: float,
    lambda_dp: float,
) -> np.ndarray:
    """P = (a*sigma/(1+F)/(1+H)/gamma_v) * D."""
    drag = compute_drag_factor(q_i=q_i, q_d=q_d, lambda_ip=lambda_ip, lambda_dp=lambda_dp)
    base = a * sigma / (1.0 + F) / (1.0 + H) / max(gamma_v, 1e-12)
    return base * drag


def compute_delta_tau(P: np.ndarray, dk: float) -> np.ndarray:
    """Δτ = P * dk."""
    return P * dk


def compute_mass(P: np.ndarray) -> np.ndarray:
    """M = 1 / P (with numerical floor)."""
    return 1.0 / np.maximum(P, 1e-12)

