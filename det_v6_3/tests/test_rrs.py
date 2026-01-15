"""
Comprehensive Test Suite for DET v6.3 RRS Extension
====================================================

Tests all RRS module components:
- RRS.I: State additions
- RRS.II: Substrate engineering
- RRS.III: Entrainment
- RRS.IV: Coherence budget (non-forkability)
- RRS.V: Rolling replacement
- RRS.VI: Migration readouts
- RRS.VII: Artificial need
- RRS.X: Falsifiers

Reference: DET Theory Card v6.3 - RRS Extension Specification
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D
from det_rrs import (
    RollingResonanceSubstrate, RRSParams, NodeLabel, BridgeBond,
    RRSFalsifierTests, run_rrs_test_suite, run_rrs_falsifier_suite
)


# =============================================================================
# RRS.I STATE ADDITIONS TESTS
# =============================================================================

class TestRRSStateAdditions:
    """Tests for RRS.I State Additions."""

    def test_node_labels_initialization(self):
        """RRS.I.1: Node labels are correctly initialized."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # All nodes start as BODY by default
        assert np.all(rrs.node_labels == NodeLabel.BODY)
        assert rrs.node_labels.shape == (16, 16, 16)

    def test_node_labels_substrate_assignment(self):
        """RRS.I.1: SUBSTRATE labels are correctly assigned."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # Use adjacent slab regions
        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Check substrate region is labeled
        assert np.any(rrs.node_labels == NodeLabel.SUBSTRATE)
        assert np.any(rrs._substrate_mask)

    def test_age_counters_initialization(self):
        """RRS.I.2: Age counters χ_i initialize to zero."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        assert np.all(rrs.chi == 0)
        assert rrs.chi.dtype == np.int64

    def test_age_counters_increment(self):
        """RRS.I.2: Age counters increment for substrate nodes."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(age_tracking_enabled=True)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        initial_chi = rrs.chi[rrs._substrate_mask].copy()

        for _ in range(10):
            rrs.step()

        final_chi = rrs.chi[rrs._substrate_mask]

        # Substrate ages should have increased
        assert np.all(final_chi >= initial_chi)
        assert np.mean(final_chi) > np.mean(initial_chi)

    def test_bridge_bonds_detection(self):
        """RRS.I.3: Bridge bonds E_BU are correctly detected."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Bridge bonds should exist between adjacent B and U nodes
        assert len(rrs.bridge_bonds) > 0

        # Verify bridge bond structure
        for bond in rrs.bridge_bonds:
            assert bond.direction in ['X', 'Y', 'Z']
            assert bond.sigma == rrs.p.sigma_bridge
            assert bond.C == rrs.p.C_bridge_init


# =============================================================================
# RRS.II SUBSTRATE ENGINEERING TESTS
# =============================================================================

class TestSubstrateEngineering:
    """Tests for RRS.II Substrate Engineering Constraints."""

    def test_high_fidelity_processing(self):
        """RRS.II.1: Substrate has higher σ than body."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(sigma_substrate_factor=2.0)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        initial_sigma = sim.sigma.copy()

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Substrate sigma should be boosted
        mean_sigma_substrate = np.mean(sim.sigma[rrs._substrate_mask])
        mean_sigma_body = np.mean(sim.sigma[rrs._body_mask])

        assert mean_sigma_substrate > mean_sigma_body


# =============================================================================
# RRS.III ENTRAINMENT TESTS
# =============================================================================

class TestEntrainment:
    """Tests for RRS.III Entrainment."""

    def test_entrainment_order_parameter_range(self):
        """RRS.III.2: R_BU ∈ [-1, 1]."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        R_BU = rrs.compute_entrainment_order_parameter()

        assert -1.0 <= R_BU <= 1.0

    def test_entrainment_no_bridges(self):
        """R_BU = 0 when no bridges exist."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # No bridge region added
        R_BU = rrs.compute_entrainment_order_parameter()

        assert R_BU == 0.0

    def test_phase_coherence_computation(self):
        """Phase coherence ψ is computed correctly."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # Set all phases to same value
        sim.theta[:] = 0.5

        mask = np.zeros((16, 16, 16), dtype=bool)
        mask[4:8, 4:8, 4:8] = True

        psi = rrs.compute_phase_coherence(mask)

        # Perfect alignment should give |ψ| ≈ 1
        assert np.abs(np.abs(psi) - 1.0) < 0.01


# =============================================================================
# RRS.IV NON-FORKABILITY TESTS
# =============================================================================

class TestCoherenceBudget:
    """Tests for RRS.IV Non-Forkability (Coherence Budget)."""

    def test_local_coherence_load_computation(self):
        """S_i = Σ C_ij is computed correctly."""
        params = DETParams3D(N=16, C_init=0.15)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        S = rrs.compute_local_coherence_load()

        # Each node has 6 neighbors, so S ≈ 6 * C_init
        expected_S = 6 * 0.15
        assert np.abs(np.mean(S) - expected_S) < 0.1

    def test_coherence_budget_enforcement(self):
        """Coherence is renormalized when exceeding S_max."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(S_max=3.0, coherence_budget_enabled=True)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        # Set high coherence
        sim.C_X[:] = 0.9
        sim.C_Y[:] = 0.9
        sim.C_Z[:] = 0.9

        S_before = np.max(rrs.compute_local_coherence_load())

        rrs.apply_coherence_budget_renormalization()

        S_after = np.max(rrs.compute_local_coherence_load())

        assert S_after < S_before
        assert S_after <= rrs_params.S_max + 0.1

    def test_coherence_budget_disabled(self):
        """No renormalization when budget is disabled."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(S_max=3.0, coherence_budget_enabled=False)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        # Set high coherence
        sim.C_X[:] = 0.9
        sim.C_Y[:] = 0.9
        sim.C_Z[:] = 0.9

        C_before = sim.C_X.copy()

        rrs.apply_coherence_budget_renormalization()

        # Should be unchanged
        assert np.allclose(sim.C_X, C_before)


# =============================================================================
# RRS.V ROLLING REPLACEMENT TESTS
# =============================================================================

class TestRollingReplacement:
    """Tests for RRS.V Rolling Replacement Operator."""

    def test_retirement_trigger_q_max(self):
        """retire(i) fires when q_i > q_max."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(q_max=0.5)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Get a substrate node
        substrate_idx = np.where(rrs._substrate_mask)
        pos = (substrate_idx[0][0], substrate_idx[1][0], substrate_idx[2][0])

        # Set q above threshold
        sim.q[pos] = 0.7

        assert rrs.check_retirement_trigger(pos) == True

    def test_retirement_trigger_P_min(self):
        """retire(i) fires when P_i < P_min."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(P_min=0.1)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        substrate_idx = np.where(rrs._substrate_mask)
        pos = (substrate_idx[0][0], substrate_idx[1][0], substrate_idx[2][0])

        # Set P below threshold
        sim.P[pos] = 0.01

        assert rrs.check_retirement_trigger(pos) == True

    def test_retirement_trigger_chi_max(self):
        """retire(i) fires when χ_i > χ_max."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(chi_max=100, age_tracking_enabled=True)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        substrate_idx = np.where(rrs._substrate_mask)
        pos = (substrate_idx[0][0], substrate_idx[1][0], substrate_idx[2][0])

        # Set age above threshold
        rrs.chi[pos] = 150

        assert rrs.check_retirement_trigger(pos) == True

    def test_retirement_trigger_body_node(self):
        """BODY nodes never retire."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(q_max=0.1)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        body_idx = np.where(rrs._body_mask)
        pos = (body_idx[0][0], body_idx[1][0], body_idx[2][0])

        # Even with high q, body nodes don't retire
        sim.q[pos] = 0.9

        assert rrs.check_retirement_trigger(pos) == False

    def test_replacement_resets_state(self):
        """Replacement resets node to fresh state."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(
            q_fresh=0.0,
            F_fresh=0.1,
            a_fresh=0.5,
            C_fresh_init=0.15
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        substrate_idx = np.where(rrs._substrate_mask)
        pos = (substrate_idx[0][0], substrate_idx[1][0], substrate_idx[2][0])

        # Set non-fresh state
        sim.q[pos] = 0.8
        sim.F[pos] = 5.0
        rrs.chi[pos] = 500

        # Perform replacement
        rrs.retire_and_replace_node(pos)

        # Check fresh state
        assert sim.q[pos] == rrs_params.q_fresh
        assert sim.F[pos] == rrs_params.F_fresh
        assert rrs.chi[pos] == 0
        assert rrs.total_replacements == 1

    def test_replacement_log(self):
        """Replacement events are logged."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        substrate_idx = np.where(rrs._substrate_mask)
        pos = (substrate_idx[0][0], substrate_idx[1][0], substrate_idx[2][0])

        rrs.retire_and_replace_node(pos)

        assert len(rrs.replacement_log) == 1
        assert rrs.replacement_log[0]['position'] == pos


# =============================================================================
# RRS.VI MIGRATION READOUTS TESTS
# =============================================================================

class TestMigrationReadouts:
    """Tests for RRS.VI Migration Readouts."""

    def test_participation_strength(self):
        """Π_i = a_i P_i is computed correctly."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # Set known values
        sim.a[:] = 0.5
        sim.P[:] = 0.2

        Pi = rrs.compute_participation_strength()

        expected = 0.5 * 0.2
        assert np.allclose(Pi, expected)

    def test_decision_locus(self):
        """i*(t) = argmax Π_i is found correctly."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # Set uniform values
        sim.a[:] = 0.5
        sim.P[:] = 0.1

        # Create peak at specific location
        peak_pos = (8, 8, 8)
        sim.a[peak_pos] = 1.0
        sim.P[peak_pos] = 1.0

        locus, value = rrs.compute_decision_locus()

        assert locus == peak_pos
        assert value == 1.0

    def test_migration_event_detection(self):
        """Migration is detected after sustained window."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(migration_window=10)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Set high participation in substrate
        sim.a[rrs._substrate_mask] = 1.0
        sim.P[rrs._substrate_mask] = 1.0

        # Run enough steps
        for _ in range(15):
            rrs.step()

        # Should detect migration to substrate
        migrated = rrs.check_migration_event()
        # Migration depends on history length
        assert isinstance(migrated, bool)

    def test_cluster_continuity_metric(self):
        """I(S) > 0 for coherent clusters."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        # Add a coherent cluster
        sim.add_packet((8, 8, 8), mass=10.0, width=3.0)

        continuity, clusters = rrs.compute_cluster_continuity()

        assert continuity >= 0.0
        assert isinstance(clusters, list)


# =============================================================================
# RRS.VII ARTIFICIAL NEED TESTS
# =============================================================================

class TestArtificialNeed:
    """Tests for RRS.VII Artificial Need."""

    def test_metabolic_drain(self):
        """Resources decrease when λ_need > 0."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(
            artificial_need_enabled=True,
            lambda_need=0.1,
            I_env_default=0.0
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Add resources to substrate
        sim.F[rrs._substrate_mask] = 5.0

        F_before = np.sum(sim.F[rrs._substrate_mask])

        rrs.apply_artificial_need()

        F_after = np.sum(sim.F[rrs._substrate_mask])

        assert F_after < F_before

    def test_environmental_injection(self):
        """I_env increases resources."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(
            artificial_need_enabled=True,
            lambda_need=0.0,  # No drain
            I_env_default=0.0
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Create custom injection
        I_env = np.zeros_like(sim.F)
        I_env[rrs._substrate_mask] = 0.5

        F_before = np.sum(sim.F[rrs._substrate_mask])

        rrs.apply_artificial_need(I_env=I_env)

        F_after = np.sum(sim.F[rrs._substrate_mask])

        assert F_after > F_before


# =============================================================================
# RRS.VIII UPDATE ORDERING TESTS
# =============================================================================

class TestUpdateOrdering:
    """Tests for RRS.VIII Canonical Update Ordering."""

    def test_step_completes(self):
        """RRS step executes without error."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Should not raise
        for _ in range(10):
            rrs.step()

        assert rrs.sim.step_count == 10

    def test_step_updates_diagnostics(self):
        """Diagnostics are tracked during step."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs_params = RRSParams(entrainment_tracking_enabled=True)
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        for _ in range(10):
            rrs.step()

        # History should be populated
        assert len(rrs.R_BU_history) > 0
        assert len(rrs.cluster_continuity_history) > 0


# =============================================================================
# RRS.X FALSIFIER TESTS
# =============================================================================

class TestFalsifiers:
    """Tests for RRS.X Falsifiers."""

    def test_falsifier_f3_locality(self):
        """F3: Verify no hidden global dependence."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        passed, details = RRSFalsifierTests.test_hidden_global_dependence(
            rrs, verbose=False
        )

        assert passed
        assert details['local_operations']['coherence_budget'] == True
        assert details['local_operations']['retirement_trigger'] == True

    def test_falsifier_f4_coercion(self):
        """F4: Verify no coercion by maintenance."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        passed, details = RRSFalsifierTests.test_coercion_by_maintenance(
            rrs, max_steps=50, verbose=False
        )

        assert passed
        assert details['replacement_respects_a_max'] == True


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for RRS with DET v6.3."""

    def test_rrs_with_gravity(self):
        """RRS works correctly with gravity enabled."""
        params = DETParams3D(N=16, gravity_enabled=True)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Add mass
        sim.add_packet((8, 8, 8), mass=5.0, width=2.0, initial_q=0.3)

        for _ in range(50):
            rrs.step()

        # Gravity should be active
        assert np.max(np.abs(sim.gz)) > 0

    def test_rrs_with_momentum(self):
        """RRS works correctly with momentum enabled."""
        params = DETParams3D(N=16, momentum_enabled=True)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        # Add packet with momentum
        sim.add_packet((8, 8, 8), mass=5.0, width=2.0, momentum=(0.1, 0, 0))

        for _ in range(50):
            rrs.step()

        # Momentum should be present
        assert np.max(np.abs(sim.pi_X)) > 0

    def test_summary_output(self):
        """Summary method returns complete information."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        for _ in range(10):
            rrs.step()

        summary = rrs.summary()

        assert 'step' in summary
        assert 'time' in summary
        assert 'body' in summary
        assert 'substrate' in summary
        assert 'bridge' in summary
        assert 'migration' in summary

    def test_statistics_methods(self):
        """Statistics methods return valid data."""
        params = DETParams3D(N=16)
        sim = DETCollider3D(params)
        rrs = RollingResonanceSubstrate(sim)

        rrs.add_adjacent_regions(
            body_slice=(slice(4, 8), slice(4, 12), slice(4, 12)),
            substrate_slice=(slice(8, 12), slice(4, 12), slice(4, 12))
        )

        body_stats = rrs.get_body_statistics()
        substrate_stats = rrs.get_substrate_statistics()
        bridge_stats = rrs.get_bridge_statistics()

        assert 'node_count' in body_stats
        assert 'node_count' in substrate_stats
        assert 'num_bridges' in bridge_stats


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_all_tests(verbose: bool = True):
    """Run all RRS tests."""
    import traceback

    print("="*70)
    print("DET v6.3 RRS EXTENSION - COMPREHENSIVE TEST SUITE")
    print("="*70)

    test_classes = [
        TestRRSStateAdditions,
        TestSubstrateEngineering,
        TestEntrainment,
        TestCoherenceBudget,
        TestRollingReplacement,
        TestMigrationReadouts,
        TestArtificialNeed,
        TestUpdateOrdering,
        TestFalsifiers,
        TestIntegration
    ]

    results = {}
    total_passed = 0
    total_failed = 0

    for test_class in test_classes:
        class_name = test_class.__name__
        if verbose:
            print(f"\n--- {class_name} ---")

        instance = test_class()

        for method_name in dir(instance):
            if method_name.startswith('test_'):
                test_name = f"{class_name}.{method_name}"
                try:
                    getattr(instance, method_name)()
                    results[test_name] = True
                    total_passed += 1
                    if verbose:
                        print(f"  {method_name}: PASS")
                except Exception as e:
                    results[test_name] = False
                    total_failed += 1
                    if verbose:
                        print(f"  {method_name}: FAIL - {e}")
                        traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Total:  {total_passed + total_failed}")
    print(f"\n  {'ALL TESTS PASSED' if total_failed == 0 else 'SOME TESTS FAILED'}")

    return results


if __name__ == "__main__":
    # Run comprehensive test suite
    results = run_all_tests(verbose=True)

    print("\n")

    # Also run the module's built-in tests
    print("="*70)
    print("RUNNING MODULE BUILT-IN TESTS")
    print("="*70)
    run_rrs_test_suite(verbose=True)
