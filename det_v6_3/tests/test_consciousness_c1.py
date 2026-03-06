"""
DET-C1 falsifier-style tests on top of the v6.3 3D collider.

F_C1: Integration benefit in coherent regimes
F_C2: Word dependence decline with higher communicability
F_C3: Nonverbal readout gain with higher path coherence
F_C4: No nonlocal leakage across disconnected components
F_C5: Agency invariance (readout-only module never overwrites a)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from det_consciousness_c1 import ConsciousnessParamsC1, DETConsciousnessC1
from det_v6_3_3d_collider import DETCollider3D, DETParams3D


def _build_sim(N: int = 12) -> DETCollider3D:
    p = DETParams3D(
        N=N,
        gravity_enabled=False,
        momentum_enabled=False,
        angular_momentum_enabled=False,
        floor_enabled=False,
        boundary_enabled=True,
        grace_enabled=True,
        F_MIN_grace=0.08,
        agency_dynamic=False,
        q_enabled=True,
        alpha_q=0.015,
        q_mutable_local_enabled=True,
        alpha_q_local_resource_relief=0.08,
        alpha_q_grace_relief=0.25,
        sigma_dynamic=False,
        coherence_dynamic=False,
    )
    sim = DETCollider3D(p)
    sim.a[:] = 0.9
    sim.sigma[:] = 1.1
    sim.F[:] = 0.03
    c = N // 2
    sim.F[c - 2 : c + 2, c - 2 : c + 2, c - 2 : c + 2] = 0.006
    sim.F[c - 3 : c + 3, c - 3 : c + 3, c - 3 : c + 3] += 0.06
    sim.q[c - 2 : c + 2, c - 2 : c + 2, c - 2 : c + 2] = 0.45
    sim.C_X[:] = 0.8
    sim.C_Y[:] = 0.8
    sim.C_Z[:] = 0.8

    # Warm up the mutable-q path so all DET-C1 tests run on exploration dynamics.
    for _ in range(20):
        sim.step()

    return sim


def _regime_masks(N: int):
    z, y, x = np.indices((N, N, N))
    mask_a = x <= (N // 2 - 2)
    mask_b = x >= (N // 2 + 1)
    return mask_a, mask_b


def test_f_c1_integration_benefit():
    """F_C1: In a coherent regime, higher U should raise P_eff."""
    sim = _build_sim(N=10)
    module = DETConsciousnessC1(sim, ConsciousnessParamsC1(alpha_U=1.0, beta_U=1.0))

    mask, _ = _regime_masks(sim.p.N)

    # Coherent regime: expect integration gain > fragmentation cost.
    sim.C_X[mask] = 0.92
    sim.C_Y[mask] = 0.92
    sim.C_Z[mask] = 0.92

    low = module.compute_regime_state("A", mask, U=0.1)
    high = module.compute_regime_state("A", mask, U=0.9)
    assert high.P_eff > low.P_eff

    # In low coherence, opposite trend can appear (sanity check).
    sim.C_X[mask] = 0.15
    sim.C_Y[mask] = 0.15
    sim.C_Z[mask] = 0.15
    low_c = module.compute_regime_state("A", mask, U=0.1)
    high_c = module.compute_regime_state("A", mask, U=0.9)
    assert high_c.P_eff < low_c.P_eff


def test_f_c2_word_dependence_declines_with_gamma():
    """F_C2: As U rises on a connected path, V should fall and accuracy improve."""
    sim = _build_sim(N=12)
    mask_a, mask_b = _regime_masks(sim.p.N)
    module = DETConsciousnessC1(
        sim,
        ConsciousnessParamsC1(path_coherence_min=0.05, path_presence_min=0.01, periodic_paths=False),
    )

    low = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.2, U_b=0.2)
    high = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.9, U_b=0.9)

    assert low.path_exists
    assert high.path_exists
    assert high.Gamma > low.Gamma
    assert high.V < low.V
    assert high.nonverbal_accuracy >= low.nonverbal_accuracy


def test_f_c3_nonverbal_readout_gain():
    """F_C3: Raising path coherence should improve nonverbal channel quality."""
    sim = _build_sim(N=12)
    mask_a, mask_b = _regime_masks(sim.p.N)

    module = DETConsciousnessC1(
        sim,
        ConsciousnessParamsC1(path_coherence_min=0.01, path_presence_min=0.01, periodic_paths=False),
    )

    # Low coherence corridor
    sim.C_X[:] = 0.15
    sim.C_Y[:] = 0.15
    sim.C_Z[:] = 0.15
    low = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.8, U_b=0.8)

    # High coherence corridor
    sim.C_X[:] = 0.85
    sim.C_Y[:] = 0.85
    sim.C_Z[:] = 0.85
    high = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.8, U_b=0.8)

    assert low.path_exists and high.path_exists
    assert high.Gamma > low.Gamma
    assert high.nonverbal_accuracy > low.nonverbal_accuracy


def test_f_c4_no_nonlocal_leakage():
    """F_C4: If local bond path is severed, communicability must drop to zero."""
    sim = _build_sim(N=12)
    mask_a, mask_b = _regime_masks(sim.p.N)
    module = DETConsciousnessC1(
        sim,
        ConsciousnessParamsC1(path_coherence_min=0.05, path_presence_min=0.01, periodic_paths=False),
    )

    # Control: connected path exists.
    connected = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.7, U_b=0.7)
    assert connected.path_exists
    assert connected.Gamma > 0.0

    # Disconnect by severing all X bonds across the mid-plane.
    mid_bond_x = sim.p.N // 2 - 1
    sim.C_X[:, :, mid_bond_x] = 0.0
    disconnected = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.7, U_b=0.7)

    assert not disconnected.path_exists
    assert disconnected.Gamma == 0.0
    assert disconnected.nonverbal_accuracy == 0.0


def test_f_c5_agency_invariance():
    """F_C5: DET-C1 readouts must never overwrite agency values."""
    sim = _build_sim(N=10)
    mask_a, mask_b = _regime_masks(sim.p.N)

    module = DETConsciousnessC1(sim, ConsciousnessParamsC1())
    a_before = sim.a.copy()

    _ = module.compute_regime_state("A", mask_a, U=0.1)
    _ = module.compute_regime_state("A", mask_a, U=0.9)
    _ = module.compute_regime_states({"A": mask_a, "B": mask_b}, {"A": 0.5, "B": 0.5})
    _ = module.compute_path_state("A", "B", mask_a, mask_b, U_a=0.5, U_b=0.5)

    assert np.array_equal(a_before, sim.a)


def test_q_mutable_path_is_active_in_fixture():
    """Guardrail: DET-C1 fixture must run the q-mutable exploration path."""
    sim = _build_sim(N=12)
    assert sim.p.q_mutable_local_enabled
    assert sim.p.q_enabled
    assert np.sum(sim.last_q_relief_local) > 0.0
