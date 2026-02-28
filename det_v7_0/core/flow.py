"""Flow helpers for DET v7."""

from __future__ import annotations

import numpy as np


def conservative_limiter(
    total_outflow: np.ndarray,
    F: np.ndarray,
    delta_tau: np.ndarray,
    outflow_limit: float,
) -> np.ndarray:
    """Compute per-node limiter scale for conservative flux clipping."""
    max_total_out = outflow_limit * F / (delta_tau + 1e-9)
    return np.minimum(1.0, max_total_out / (total_outflow + 1e-9))

