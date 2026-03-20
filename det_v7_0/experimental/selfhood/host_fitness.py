"""
Host Fitness and Reciprocity Fields for DET v7 Speculative Selfhood Patch.

Patch ID: DET-7S-SPIRIT-HOST-1
Status: Speculative / non-canonical / readout-first

This module computes two diagnostic observables:
  - R_i : Net reciprocity field (mutual support in local neighborhood)
  - H^host_i : Host fitness field (whether a coalition can support selfhood)

These are computed AFTER the canonical DET v7 update loop (step 1-15)
and NEVER feed back into canonical dynamics (Phase 1 = readout-only).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class HostFitnessParams:
    """Parameters for host fitness computation."""
    # Host fitness weights (must be non-negative)
    # NOTE: P (presence) is naturally small in DET (typically 0.01-0.15)
    # because P = a*sigma/(1+F)/(1+H)/gamma_v * D. Using w_P=1.0 would
    # make H_host ~ 0 always. We use a reduced weight so P contributes
    # without dominating.
    w_a: float = 1.0       # agency weight
    w_P: float = 0.3       # presence weight (reduced: P is naturally small)
    w_C: float = 1.0       # coherence weight
    w_q: float = 0.8       # structural drag weight (via 1-q)
    w_R: float = 0.8       # reciprocity weight
    w_M: float = 0.5       # pointer/record stability weight

    # Developmental maturity
    developmental_enabled: bool = False
    w_D: float = 1.0       # developmental maturity weight on host fitness
    mu_D: float = 0.005    # developmental maturation rate
    lambda_D: float = 0.002  # developmental regression rate

    # Numerical
    epsilon: float = 1e-8   # denominator floor for reciprocity


def compute_neighborhood_averages_2d(
    a: np.ndarray,
    P: np.ndarray,
    C_E: np.ndarray,
    C_S: np.ndarray,
    q: np.ndarray,
) -> dict:
    """Compute local neighborhood averages for 2D periodic lattice.

    Returns dict with keys: a_bar, P_bar, C_bar, q_bar.
    Each is the average over the node and its 4 nearest neighbors.
    """
    E = lambda x: np.roll(x, -1, axis=1)
    W = lambda x: np.roll(x, 1, axis=1)
    S = lambda x: np.roll(x, -1, axis=0)
    N = lambda x: np.roll(x, 1, axis=0)

    a_bar = (a + E(a) + W(a) + S(a) + N(a)) / 5.0
    P_bar = (P + E(P) + W(P) + S(P) + N(P)) / 5.0
    q_bar = (q + E(q) + W(q) + S(q) + N(q)) / 5.0

    # Node coherence from bond coherences (average of 4 bonds touching node)
    C_bar = (C_E + W(C_E) + C_S + N(C_S)) / 4.0

    return {
        "a_bar": a_bar,
        "P_bar": P_bar,
        "C_bar": C_bar,
        "q_bar": q_bar,
    }


def compute_reciprocity_2d(
    C_E: np.ndarray,
    C_S: np.ndarray,
    F: np.ndarray,
    a: np.ndarray,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Compute net reciprocity field R_i in [0,1] for 2D lattice.

    Reciprocity measures mutuality of support in a local neighborhood.
    R_i ~ 1 means strongly reciprocal (balanced) support.
    R_i ~ 0 means one-way or fragmented exchange.

    Implementation: coherence-weighted balance metric.
    For each bond (i,j), the balance is:
        b_ij = 1 - |F_i - F_j| / (F_i + F_j + eps)
    weighted by bond coherence and geometric mean of agency:
        R_i = sum_j [C_ij * sqrt(a_i * a_j) * b_ij] / (sum_j [C_ij * sqrt(a_i * a_j)] + eps)

    This correctly gives:
    - R ~ 1 for balanced neighborhoods (uniform F, mutual support)
    - R ~ 0 for one-way extraction (large F gradients)
    - R = 0 when agency or coherence is zero

    NOTE: The original spec used min(J+_ij, J+_ji)/max(J+_ij, J+_ji),
    but this is identically zero for instantaneous flow (since flow is
    always one-directional at any instant). The balance metric captures
    the same intent: mutuality of resource sharing.
    """
    E = lambda x: np.roll(x, -1, axis=1)
    W = lambda x: np.roll(x, 1, axis=1)
    S = lambda x: np.roll(x, -1, axis=0)
    N = lambda x: np.roll(x, 1, axis=0)

    def _balance_pair(C_bond, a_i, a_j, F_i, F_j):
        """Compute weighted balance for one bond direction."""
        g = np.sqrt(np.maximum(a_i, 0.0) * np.maximum(a_j, 0.0))
        balance = 1.0 - np.abs(F_i - F_j) / (F_i + F_j + epsilon)
        weight = C_bond * g
        return weight * balance, weight

    # East bond
    wb_e, w_e = _balance_pair(C_E, a, E(a), F, E(F))
    # West bond
    wb_w, w_w = _balance_pair(W(C_E), a, W(a), F, W(F))
    # South bond
    wb_s, w_s = _balance_pair(C_S, a, S(a), F, S(F))
    # North bond
    wb_n, w_n = _balance_pair(N(C_S), a, N(a), F, N(F))

    numer = wb_e + wb_w + wb_s + wb_n
    denom = w_e + w_w + w_s + w_n + epsilon

    R = numer / denom
    return np.clip(R, 0.0, 1.0)


def compute_pointer_stability_proxy(
    q: np.ndarray,
    C_E: np.ndarray,
    C_S: np.ndarray,
) -> np.ndarray:
    """Proxy for pointer/record stability M^ptr_i.

    In the absence of an explicit pointer-record field in the current
    collider, we use a proxy: high coherence + low q indicates stable
    organizational structure that can maintain records.

    M^ptr_i = C_bar * (1 - q_bar)
    """
    W = lambda x: np.roll(x, 1, axis=1)
    N = lambda x: np.roll(x, 1, axis=0)

    C_bar = (C_E + W(C_E) + C_S + N(C_S)) / 4.0
    return C_bar * (1.0 - q)


def compute_host_fitness(
    a_bar: np.ndarray,
    P_bar: np.ndarray,
    C_bar: np.ndarray,
    q_bar: np.ndarray,
    R: np.ndarray,
    M_ptr: np.ndarray,
    params: HostFitnessParams,
    D_mature: np.ndarray | None = None,
) -> np.ndarray:
    """Compute host fitness H^host_i in [0,1].

    H^host_i = clip[ a_bar^w_a * P_bar^w_P * C_bar^w_C * (1-q_bar)^w_q
                      * R^w_R * M_ptr^w_M, 0, 1 ]

    If developmental maturity is enabled:
        H^host_i *= D_mature^w_D
    """
    # Ensure all inputs are non-negative for power operations
    a_safe = np.maximum(a_bar, 0.0)
    P_safe = np.maximum(P_bar, 0.0)
    C_safe = np.maximum(C_bar, 0.0)
    q_complement = np.maximum(1.0 - q_bar, 0.0)
    R_safe = np.maximum(R, 0.0)
    M_safe = np.maximum(M_ptr, 0.0)

    H = (
        np.power(a_safe, params.w_a)
        * np.power(P_safe, params.w_P)
        * np.power(C_safe, params.w_C)
        * np.power(q_complement, params.w_q)
        * np.power(R_safe, params.w_R)
        * np.power(M_safe, params.w_M)
    )

    if params.developmental_enabled and D_mature is not None:
        D_safe = np.maximum(D_mature, 0.0)
        H = H * np.power(D_safe, params.w_D)

    return np.clip(H, 0.0, 1.0)


def update_developmental_maturity(
    D_mature: np.ndarray,
    C_bar: np.ndarray,
    R: np.ndarray,
    P_bar: np.ndarray,
    params: HostFitnessParams,
) -> np.ndarray:
    """Update developmental maturity D^mature_i.

    D^{mature,+}_i = clip[
        D_i + mu_D * C_bar * R * (1 - D_i)
            - lambda_D * (1 - P_bar) * D_i
    , 0, 1]
    """
    growth = params.mu_D * C_bar * R * (1.0 - D_mature)
    decay = params.lambda_D * (1.0 - P_bar) * D_mature
    return np.clip(D_mature + growth - decay, 0.0, 1.0)
