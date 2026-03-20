"""
Identity Persistence Metrics for DET v7 Speculative Selfhood Patch.

Patch ID: DET-7S-SPIRIT-HOST-1
Status: Speculative / non-canonical / readout-first

This module computes the identity persistence metric I_self(t1, t2),
which asks whether a later regime is "the same self" in a graded sense.

I_self ~ 1  : strong continuity
0.3 < I_self < 0.8 : dreamlike partial continuity
I_self ~ 0  : no meaningful persistence
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class IdentityMetricParams:
    """Weights for identity persistence metric components."""
    w_S: float = 0.35    # self-coherence correlation weight
    w_C: float = 0.25    # coherence correlation weight
    w_B: float = 0.20    # bond topology overlap weight
    w_P: float = 0.20    # pointer/record correlation weight


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, returning 0 if degenerate."""
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)

    if len(x_flat) < 2:
        return 0.0

    x_std = np.std(x_flat)
    y_std = np.std(y_flat)

    if x_std < 1e-12 or y_std < 1e-12:
        # If both are constant and equal, perfect correlation
        if x_std < 1e-12 and y_std < 1e-12:
            if np.allclose(x_flat, y_flat, atol=1e-10):
                return 1.0
            return 0.0
        return 0.0

    corr = np.corrcoef(x_flat, y_flat)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def _bond_topology_overlap(
    C_E_1: np.ndarray, C_S_1: np.ndarray,
    C_E_2: np.ndarray, C_S_2: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute overlap of bond topology signatures.

    Binarize bonds as "active" if C > threshold, then compute
    Jaccard overlap between the two bond sets.
    """
    active_1_E = C_E_1 > threshold
    active_1_S = C_S_1 > threshold
    active_2_E = C_E_2 > threshold
    active_2_S = C_S_2 > threshold

    # Flatten and concatenate
    b1 = np.concatenate([active_1_E.ravel(), active_1_S.ravel()])
    b2 = np.concatenate([active_2_E.ravel(), active_2_S.ravel()])

    intersection = np.sum(b1 & b2)
    union = np.sum(b1 | b2)

    if union == 0:
        return 0.0
    return float(intersection / union)


@dataclass
class Snapshot:
    """A snapshot of the system state at a given time for identity comparison."""
    S: np.ndarray          # self-coherence field
    C_E: np.ndarray        # east bond coherence
    C_S: np.ndarray        # south bond coherence
    C_bar: np.ndarray      # node-average coherence
    M_ptr: np.ndarray      # pointer/record stability proxy
    time: float = 0.0

    def copy(self) -> "Snapshot":
        return Snapshot(
            S=self.S.copy(),
            C_E=self.C_E.copy(),
            C_S=self.C_S.copy(),
            C_bar=self.C_bar.copy(),
            M_ptr=self.M_ptr.copy(),
            time=self.time,
        )


def _value_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Compute value-level similarity: 1 - mean(|x-y|) / (mean(|x|+|y|) + eps).

    Unlike correlation (which only measures pattern shape), this also
    captures magnitude differences. Two fields with the same spatial
    pattern but different magnitudes will score < 1.
    """
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)
    diff = np.mean(np.abs(x_flat - y_flat))
    scale = np.mean(np.abs(x_flat) + np.abs(y_flat)) + 1e-12
    return float(max(0.0, 1.0 - diff / scale))


def compute_identity_persistence(
    snap1: Snapshot,
    snap2: Snapshot,
    params: IdentityMetricParams | None = None,
    bond_threshold: float = 0.5,
) -> dict:
    """Compute identity persistence metric I_self between two snapshots.

    I_self = w_S * sim(S1, S2) + w_C * sim(C1, C2)
           + w_B * overlap(B1, B2) + w_P * sim(Ptr1, Ptr2)

    We use a combined similarity metric that captures both spatial
    correlation AND magnitude similarity. Pure correlation is insufficient
    because rebuilding in the same region trivially recreates the same
    spatial pattern. The value similarity component ensures that the
    actual field magnitudes must also match.

    Each component is normalized to [0, 1].

    Returns
    -------
    dict with keys:
        I_self : float in [0, 1]
        sim_S : float (combined similarity for S)
        sim_C : float (combined similarity for C)
        overlap_B : float
        sim_P : float (combined similarity for Ptr)
    """
    if params is None:
        params = IdentityMetricParams()

    def _combined_similarity(a, b):
        """Geometric mean of correlation and value similarity."""
        corr = max(0.0, _safe_correlation(a, b))
        val_sim = _value_similarity(a, b)
        # Geometric mean penalizes if either component is low
        return float(np.sqrt(corr * val_sim))

    # Self-coherence similarity
    sim_S = _combined_similarity(snap1.S, snap2.S)

    # Coherence similarity
    sim_C = _combined_similarity(snap1.C_bar, snap2.C_bar)

    # Bond topology overlap
    overlap_B = _bond_topology_overlap(
        snap1.C_E, snap1.C_S,
        snap2.C_E, snap2.C_S,
        threshold=bond_threshold,
    )

    # Pointer/record similarity
    sim_P = _combined_similarity(snap1.M_ptr, snap2.M_ptr)

    I_self = (
        params.w_S * sim_S
        + params.w_C * sim_C
        + params.w_B * overlap_B
        + params.w_P * sim_P
    )

    return {
        "I_self": float(np.clip(I_self, 0.0, 1.0)),
        "sim_S": sim_S,
        "sim_C": sim_C,
        "overlap_B": overlap_B,
        "sim_P": sim_P,
    }
