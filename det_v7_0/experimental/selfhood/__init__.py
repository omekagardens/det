"""DET v7 Experimental Selfhood Module (DET-7S-SPIRIT-HOST-1).

Status: Speculative / non-canonical / readout-first.
"""

from .host_fitness import (
    HostFitnessParams,
    compute_neighborhood_averages_2d,
    compute_reciprocity_2d,
    compute_pointer_stability_proxy,
    compute_host_fitness,
    update_developmental_maturity,
)
from .self_field import (
    SelfFieldParams,
    update_self_field,
    compute_self_assisted_healing_2d,
)
from .identity_metrics import (
    IdentityMetricParams,
    Snapshot,
    compute_identity_persistence,
)
from .diagnostics import (
    SelfhoodDiagnostics,
    SelfhoodHarness,
)
