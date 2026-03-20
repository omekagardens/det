"""
Self-Coherence Occupancy Field for DET v7 Speculative Selfhood Patch.

Patch ID: DET-7S-SPIRIT-HOST-1
Status: Speculative / non-canonical / readout-first

This module computes the speculative self-coherence occupancy S_i,
which models whether higher-order selfhood is stably expressed at a node.

S_i is NOT agency. S_i is NOT a hidden external force.
S_i is a local readout of whether higher-order selfhood is stably expressed.

Phase 1: readout-only (no feedback into canonical DET laws).
Phase 2: optional weak coupling to healing (disabled by default).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class SelfFieldParams:
    """Parameters for self-coherence occupancy dynamics."""
    # Self-formation
    mu_S: float = 0.05         # self-formation gain
    Theta_H: float = 0.3       # host-fitness threshold
    k_H: float = 10.0          # threshold sharpness (sigmoid steepness)

    # Self-fragmentation
    lambda_S: float = 0.03     # coherence-fragmentation loss rate
    chi_S: float = 0.02        # q-burden coupling on self-expression

    # Phase 2: optional self-assisted healing coupling
    phase2_enabled: bool = False
    eta_S: float = 0.001       # self-assisted healing gain (must be << eta_heal)


def _sigmoid(x: np.ndarray, k: float) -> np.ndarray:
    """Soft threshold: sigma(x) = 1 / (1 + exp(-k*x))."""
    # Clip to avoid overflow
    kx = np.clip(k * x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-kx))


def update_self_field(
    S: np.ndarray,
    H_host: np.ndarray,
    C_bar: np.ndarray,
    q_bar: np.ndarray,
    params: SelfFieldParams,
) -> np.ndarray:
    """Update self-coherence occupancy S_i.

    S_i^+ = clip[
        S_i
        + mu_S * sigma(H_host - Theta_H) * (1 - S_i)
        - lambda_S * (1 - C_bar) * S_i
        - chi_S * q_bar * S_i
    , 0, 1]

    Parameters
    ----------
    S : current self-coherence field
    H_host : host fitness field
    C_bar : local average coherence
    q_bar : local average structural debt
    params : SelfFieldParams

    Returns
    -------
    S_new : updated self-coherence field
    """
    # Formation: rises when host fitness exceeds threshold
    # Hard gate: zero formation when H_host is well below threshold
    # This prevents ghost insertion via sigmoid leakage at extreme mu_S
    threshold_gate = _sigmoid(H_host - params.Theta_H, params.k_H)
    hard_gate = np.where(H_host > params.Theta_H * 0.5, 1.0, 0.0)
    formation = params.mu_S * threshold_gate * hard_gate * (1.0 - S)

    # Fragmentation: falls when coherence is low
    fragmentation = params.lambda_S * (1.0 - C_bar) * S

    # Drag burden: falls when structural debt is high
    drag_burden = params.chi_S * q_bar * S

    S_new = S + formation - fragmentation - drag_burden
    return np.clip(S_new, 0.0, 1.0)


def compute_self_assisted_healing_2d(
    S: np.ndarray,
    a: np.ndarray,
    C_E: np.ndarray,
    C_S: np.ndarray,
    D: np.ndarray,
    Delta_tau: np.ndarray,
    D_mature: np.ndarray | None,
    params: SelfFieldParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Phase 2: Optional self-assisted healing coupling.

    Delta C^{heal,S}_{ij} = eta_S * S_bar_ij * g^(a)_ij * (1-C_ij) * D_bar_ij * Delta_tau_ij

    Only modifies C (never a). Disabled by default.

    Returns
    -------
    dC_heal_S_E, dC_heal_S_S : healing increments for East and South bonds
    """
    if not params.phase2_enabled:
        return np.zeros_like(C_E), np.zeros_like(C_S)

    E = lambda x: np.roll(x, -1, axis=1)
    Sf = lambda x: np.roll(x, -1, axis=0)

    # Bond-averaged self-coherence
    S_bar_E = 0.5 * (S + E(S))
    S_bar_S = 0.5 * (S + Sf(S))

    # Agency gate
    g_E = np.sqrt(a * E(a))
    g_S = np.sqrt(a * Sf(a))

    # Room for healing
    room_E = 1.0 - C_E
    room_S = 1.0 - C_S

    # Bond-averaged drag factor
    D_bar_E = 0.5 * (D + E(D))
    D_bar_S = 0.5 * (D + Sf(D))

    # Bond-averaged proper time
    Delta_tau_E = 0.5 * (Delta_tau + E(Delta_tau))
    Delta_tau_S = 0.5 * (Delta_tau + Sf(Delta_tau))

    dC_E = params.eta_S * S_bar_E * g_E * room_E * D_bar_E * Delta_tau_E
    dC_S = params.eta_S * S_bar_S * g_S * room_S * D_bar_S * Delta_tau_S

    # Optional developmental gating
    if D_mature is not None:
        D_bar_E_mat = 0.5 * (D_mature + E(D_mature))
        D_bar_S_mat = 0.5 * (D_mature + Sf(D_mature))
        dC_E = dC_E * D_bar_E_mat
        dC_S = dC_S * D_bar_S_mat

    return dC_E, dC_S
