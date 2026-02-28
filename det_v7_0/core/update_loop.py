"""Canonical DET v7 update-loop metadata."""

CANONICAL_STEPS = (
    "solve_baseline_field",
    "compute_gravitational_potential",
    "compute_presence_with_drag",
    "compute_delta_tau",
    "compute_flux_components",
    "apply_conservative_limiter",
    "update_F",
    "update_momentum",
    "update_angular_momentum",
    "update_structure_qd",
    "update_agency",
    "update_coherence",
    "update_pointer_records",
)

