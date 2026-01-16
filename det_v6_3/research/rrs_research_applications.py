"""
DET v6.3 RRS Extension: Research Applications and Novel Explorations
=====================================================================

This module explores the research applications, novel experiments, and
theoretical implications enabled by the Rolling Resonance Substrate (RRS)
extension for DET v6.3.

Key Research Areas:
1. Consciousness/Agency Migration Dynamics
2. Ship-of-Theseus Longevity Mechanics
3. Non-Forkability and Identity Preservation
4. Biological-Artificial Interface Modeling
5. Failure Mode Analysis (Prison Regimes, Fork Emergence)
6. Emergent Collective Phenomena

Reference: DET Theory Card v6.3 - RRS Extension
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sys
import os

# Optional matplotlib import for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D
from det_rrs import (
    RollingResonanceSubstrate, RRSParams, NodeLabel,
    RRSFalsifierTests
)


# =============================================================================
# 1. CONSCIOUSNESS/AGENCY MIGRATION DYNAMICS
# =============================================================================

@dataclass
class MigrationExperimentResult:
    """Results from a migration dynamics experiment."""
    migration_occurred: bool
    migration_step: Optional[int]
    decision_locus_trajectory: List[Tuple[int, int, int]]
    participation_history: List[float]
    R_BU_history: List[float]
    continuity_history: List[float]
    body_F_history: List[float]
    substrate_F_history: List[float]


def experiment_gradual_migration(
    resource_shift_rate: float = 0.01,
    max_steps: int = 2000,
    verbose: bool = True
) -> MigrationExperimentResult:
    """
    Experiment 1.1: Gradual Resource Migration

    Hypothesis: Gradual resource shift from B to U enables smooth migration
    of the decision locus while preserving cluster continuity.

    Setup:
    - Initialize cluster centered in BODY region
    - Slowly shift resources toward SUBSTRATE via gradient
    - Track decision locus, participation, and continuity

    Key Questions:
    - What is the critical resource ratio for migration?
    - Does entrainment (R_BU) predict migration success?
    - Is there hysteresis in the migration process?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1.1: Gradual Migration Dynamics")
        print("="*70)

    # Setup simulation
    params = DETParams3D(N=32, DT=0.02, gravity_enabled=True)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(
        migration_window=50,
        coherence_budget_enabled=True,
        S_max=6.0
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Adjacent slab regions
    rrs.add_adjacent_regions(
        body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
        substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
    )

    # Initialize cluster in BODY region
    sim.add_packet((11, 16, 16), mass=15.0, width=4.0, initial_q=0.2)

    # Tracking variables
    locus_trajectory = []
    participation_history = []
    R_BU_history = []
    continuity_history = []
    body_F_history = []
    substrate_F_history = []

    migration_step = None

    for step in range(max_steps):
        # Apply gradual resource shift: drain from body, inject to substrate
        if step > 100:  # Wait for initial equilibration
            drain = resource_shift_rate * sim.F[rrs._body_mask] * sim.Delta_tau[rrs._body_mask]
            inject = resource_shift_rate * rrs.p.F_fresh

            sim.F[rrs._body_mask] -= drain * 0.5
            sim.F[rrs._substrate_mask] += inject * 0.1

        rrs.step()

        # Track metrics
        locus, participation = rrs.compute_decision_locus()
        locus_trajectory.append(locus)
        participation_history.append(participation)

        R_BU = rrs.compute_entrainment_order_parameter()
        R_BU_history.append(R_BU)

        continuity, _ = rrs.compute_cluster_continuity()
        continuity_history.append(continuity)

        body_F = np.sum(sim.F[rrs._body_mask])
        substrate_F = np.sum(sim.F[rrs._substrate_mask])
        body_F_history.append(body_F)
        substrate_F_history.append(substrate_F)

        # Check for migration
        if migration_step is None and rrs._substrate_mask[locus]:
            migration_step = step
            if verbose:
                print(f"  Migration detected at step {step}")
                print(f"    Locus: {locus}")
                print(f"    Participation: {participation:.4f}")
                print(f"    R_BU: {R_BU:.4f}")

        if verbose and step % 400 == 0:
            print(f"  Step {step}: locus_in_substrate={rrs._substrate_mask[locus]}, "
                  f"R_BU={R_BU:.4f}, body_F={body_F:.1f}, sub_F={substrate_F:.1f}")

    migration_occurred = migration_step is not None

    if verbose:
        print(f"\n  Result: Migration {'OCCURRED' if migration_occurred else 'DID NOT OCCUR'}")
        if migration_occurred:
            print(f"    Migration step: {migration_step}")
            print(f"    Final participation: {participation_history[-1]:.4f}")

    return MigrationExperimentResult(
        migration_occurred=migration_occurred,
        migration_step=migration_step,
        decision_locus_trajectory=locus_trajectory,
        participation_history=participation_history,
        R_BU_history=R_BU_history,
        continuity_history=continuity_history,
        body_F_history=body_F_history,
        substrate_F_history=substrate_F_history
    )


def experiment_sudden_transfer(
    transfer_fraction: float = 0.8,
    max_steps: int = 1000,
    verbose: bool = True
) -> MigrationExperimentResult:
    """
    Experiment 1.2: Sudden Transfer ("Upload" Scenario)

    Hypothesis: Abrupt resource transfer may cause cluster fragmentation
    or loss of continuity, unlike gradual migration.

    Key Questions:
    - Is there a minimum "transfer time" for identity preservation?
    - What happens to entrainment during rapid transfer?
    - Can the cluster "refuse" migration if transfer is too fast?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 1.2: Sudden Transfer Dynamics")
        print("="*70)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(migration_window=30)
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    rrs.add_adjacent_regions(
        body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
        substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
    )

    # Initialize in BODY
    sim.add_packet((11, 16, 16), mass=15.0, width=4.0, initial_q=0.2)

    # Let it equilibrate
    for _ in range(100):
        rrs.step()

    # Tracking
    locus_trajectory = []
    participation_history = []
    R_BU_history = []
    continuity_history = []
    body_F_history = []
    substrate_F_history = []

    continuity_before = rrs.compute_cluster_continuity()[0]

    if verbose:
        print(f"  Pre-transfer continuity: {continuity_before:.4f}")

    # SUDDEN TRANSFER: Move resources instantly
    transfer_amount = sim.F[rrs._body_mask] * transfer_fraction
    sim.F[rrs._body_mask] -= transfer_amount
    sim.F[rrs._substrate_mask] += np.mean(transfer_amount) * np.ones(np.sum(rrs._substrate_mask))

    if verbose:
        print(f"  Transferred {np.sum(transfer_amount):.1f} resources")

    migration_step = None

    for step in range(max_steps):
        rrs.step()

        locus, participation = rrs.compute_decision_locus()
        locus_trajectory.append(locus)
        participation_history.append(participation)
        R_BU_history.append(rrs.compute_entrainment_order_parameter())

        continuity, clusters = rrs.compute_cluster_continuity()
        continuity_history.append(continuity)

        body_F_history.append(np.sum(sim.F[rrs._body_mask]))
        substrate_F_history.append(np.sum(sim.F[rrs._substrate_mask]))

        if migration_step is None and rrs._substrate_mask[locus]:
            migration_step = step

        if verbose and step % 200 == 0:
            print(f"  Step {step}: continuity={continuity:.4f}, "
                  f"n_clusters={len(clusters)}, locus_in_sub={rrs._substrate_mask[locus]}")

    if verbose:
        print(f"\n  Post-transfer continuity: {continuity_history[-1]:.4f}")
        print(f"  Continuity preserved: {continuity_history[-1] > 0.5 * continuity_before}")

    return MigrationExperimentResult(
        migration_occurred=migration_step is not None,
        migration_step=migration_step,
        decision_locus_trajectory=locus_trajectory,
        participation_history=participation_history,
        R_BU_history=R_BU_history,
        continuity_history=continuity_history,
        body_F_history=body_F_history,
        substrate_F_history=substrate_F_history
    )


# =============================================================================
# 2. SHIP-OF-THESEUS LONGEVITY MECHANICS
# =============================================================================

@dataclass
class LongevityExperimentResult:
    """Results from a longevity experiment."""
    survived: bool
    survival_duration: int
    total_replacements: int
    mean_continuity: float
    min_continuity: float
    replacement_rate_history: List[float]
    continuity_history: List[float]
    mean_age_history: List[float]


def experiment_longevity_under_churn(
    churn_rate: float = 0.1,
    max_steps: int = 5000,
    continuity_threshold: float = 1.0,
    verbose: bool = True
) -> LongevityExperimentResult:
    """
    Experiment 2.1: Cluster Survival Under Replacement Churn

    Hypothesis: A coherent cluster can survive indefinite node replacement
    if churn rate is below a critical threshold.

    Key Questions:
    - What is the maximum sustainable churn rate?
    - How does coherence budget affect longevity?
    - Is there a "regeneration" regime vs. "degradation" regime?
    """
    if verbose:
        print("\n" + "="*70)
        print(f"EXPERIMENT 2.1: Longevity Under Churn (rate={churn_rate})")
        print("="*70)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    # Configure for high churn
    rrs_params = RRSParams(
        q_max=0.3,  # Low threshold = more replacements
        chi_max=int(50 / churn_rate),  # Age limit scales with churn rate
        P_min=0.02,
        rolling_replacement_enabled=True,
        coherence_budget_enabled=True,
        S_max=6.0
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    rrs.add_adjacent_regions(
        body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
        substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
    )

    # Initialize cluster spanning both regions
    sim.add_packet((16, 16, 16), mass=20.0, width=5.0, initial_q=0.1)

    # Artificially age some substrate nodes to trigger churn
    initial_ages = np.random.randint(0, rrs_params.chi_max // 2, size=rrs.chi.shape)
    rrs.chi[rrs._substrate_mask] = initial_ages[rrs._substrate_mask]

    # Tracking
    replacement_counts = []
    continuity_history = []
    mean_age_history = []

    prev_replacements = 0
    survival_duration = max_steps
    survived = True

    for step in range(max_steps):
        rrs.step()

        # Track replacement rate
        new_replacements = rrs.total_replacements - prev_replacements
        replacement_counts.append(new_replacements)
        prev_replacements = rrs.total_replacements

        # Track continuity
        continuity, clusters = rrs.compute_cluster_continuity()
        continuity_history.append(continuity)

        # Track mean age
        if np.any(rrs._substrate_mask):
            mean_age = np.mean(rrs.chi[rrs._substrate_mask])
            mean_age_history.append(mean_age)
        else:
            mean_age_history.append(0)

        # Check for collapse
        if continuity < continuity_threshold and step > 200:
            survival_duration = step
            survived = False
            if verbose:
                print(f"  COLLAPSE at step {step}: continuity={continuity:.4f}")
            break

        if verbose and step % 1000 == 0:
            recent_rate = np.mean(replacement_counts[-100:]) if len(replacement_counts) >= 100 else 0
            print(f"  Step {step}: replacements={rrs.total_replacements}, "
                  f"rate={recent_rate:.2f}/step, continuity={continuity:.4f}")

    if verbose:
        print(f"\n  Result: {'SURVIVED' if survived else 'COLLAPSED'}")
        print(f"  Survival duration: {survival_duration}")
        print(f"  Total replacements: {rrs.total_replacements}")
        print(f"  Mean continuity: {np.mean(continuity_history):.4f}")

    return LongevityExperimentResult(
        survived=survived,
        survival_duration=survival_duration,
        total_replacements=rrs.total_replacements,
        mean_continuity=float(np.mean(continuity_history)),
        min_continuity=float(np.min(continuity_history)),
        replacement_rate_history=replacement_counts,
        continuity_history=continuity_history,
        mean_age_history=mean_age_history
    )


def experiment_debt_accumulation_regimes(
    debt_injection_rate: float = 0.01,
    max_steps: int = 3000,
    verbose: bool = True
) -> Dict:
    """
    Experiment 2.2: Debt Accumulation in BODY vs SUBSTRATE

    Hypothesis: SUBSTRATE with rolling replacement maintains lower mean debt
    than BODY, enabling indefinite operation.

    Key Questions:
    - How does mean debt evolve in each region?
    - At what debt level does agency collapse occur?
    - Can rolling replacement "export" debt via retirement?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 2.2: Debt Accumulation Regimes")
        print("="*70)

    params = DETParams3D(N=32, DT=0.02, q_enabled=True)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(
        q_max=0.6,
        rolling_replacement_enabled=True
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    rrs.add_adjacent_regions(
        body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
        substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
    )

    # Initialize with activity
    sim.add_packet((11, 16, 16), mass=10.0, width=3.0)
    sim.add_packet((21, 16, 16), mass=10.0, width=3.0)

    # Tracking
    body_q_history = []
    substrate_q_history = []
    body_a_history = []
    substrate_a_history = []

    for step in range(max_steps):
        # Inject debt uniformly (simulating metabolic byproducts)
        debt_injection = debt_injection_rate * sim.Delta_tau
        sim.q = np.clip(sim.q + debt_injection, 0, 1)

        rrs.step()

        body_q_history.append(np.mean(sim.q[rrs._body_mask]))
        substrate_q_history.append(np.mean(sim.q[rrs._substrate_mask]))
        body_a_history.append(np.mean(sim.a[rrs._body_mask]))
        substrate_a_history.append(np.mean(sim.a[rrs._substrate_mask]))

        if verbose and step % 500 == 0:
            print(f"  Step {step}: body_q={body_q_history[-1]:.4f}, "
                  f"sub_q={substrate_q_history[-1]:.4f}, "
                  f"body_a={body_a_history[-1]:.4f}, sub_a={substrate_a_history[-1]:.4f}")

    if verbose:
        print(f"\n  Final mean debt - BODY: {body_q_history[-1]:.4f}, SUBSTRATE: {substrate_q_history[-1]:.4f}")
        print(f"  Final mean agency - BODY: {body_a_history[-1]:.4f}, SUBSTRATE: {substrate_a_history[-1]:.4f}")

    return {
        'body_q_history': body_q_history,
        'substrate_q_history': substrate_q_history,
        'body_a_history': body_a_history,
        'substrate_a_history': substrate_a_history,
        'total_replacements': rrs.total_replacements
    }


# =============================================================================
# 3. NON-FORKABILITY AND IDENTITY PRESERVATION
# =============================================================================

def experiment_coherence_budget_fork_prevention(
    S_max_values: List[float] = [2.0, 4.0, 6.0, 10.0, 20.0],
    max_steps: int = 2000,
    verbose: bool = True
) -> Dict[float, Dict]:
    """
    Experiment 3.1: Coherence Budget as Fork Prevention Mechanism

    Hypothesis: Lower S_max prevents stable fork emergence by limiting
    coherence bandwidth per node.

    Key Questions:
    - At what S_max does fork emergence become possible?
    - How does S_max affect entrainment dynamics?
    - Is there a trade-off between fork prevention and migration ease?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 3.1: Coherence Budget Fork Prevention")
        print("="*70)

    results = {}

    for S_max in S_max_values:
        if verbose:
            print(f"\n  Testing S_max = {S_max}")

        params = DETParams3D(N=32, DT=0.02)
        sim = DETCollider3D(params)

        rrs_params = RRSParams(
            coherence_budget_enabled=True,
            S_max=S_max
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        # Create two separate regions to encourage forking
        rrs.add_adjacent_regions(
            body_slice=(slice(6, 14), slice(8, 24), slice(8, 24)),
            substrate_slice=(slice(18, 26), slice(8, 24), slice(8, 24))
        )

        # Initialize TWO clusters - potential fork scenario
        sim.add_packet((10, 16, 16), mass=12.0, width=3.0, initial_q=0.1)
        sim.add_packet((22, 16, 16), mass=12.0, width=3.0, initial_q=0.1)

        # Track cluster count over time
        cluster_counts = []
        dual_cluster_steps = 0

        for step in range(max_steps):
            rrs.step()

            _, clusters = rrs.compute_cluster_continuity(threshold_ratio=10.0)
            cluster_counts.append(len(clusters))

            if len(clusters) >= 2:
                dual_cluster_steps += 1

        fork_fraction = dual_cluster_steps / max_steps

        results[S_max] = {
            'mean_clusters': np.mean(cluster_counts),
            'max_clusters': max(cluster_counts),
            'fork_fraction': fork_fraction,
            'final_S_mean': np.mean(rrs.compute_local_coherence_load())
        }

        if verbose:
            print(f"    Fork fraction: {fork_fraction:.3f}")
            print(f"    Mean clusters: {results[S_max]['mean_clusters']:.2f}")

    return results


def experiment_attempted_duplication(
    copy_fidelity: float = 0.9,
    max_steps: int = 1500,
    verbose: bool = True
) -> Dict:
    """
    Experiment 3.2: Why Copying Fails (Attempted Duplication)

    Hypothesis: Attempting to "copy" a cluster's state to a second location
    will fail due to coherence budget constraints and phase decoherence.

    Key Questions:
    - What happens when we try to copy F, θ, a to a new location?
    - Does the "copy" achieve independent agency or decay?
    - Is the failure mode fork-collapse or merger?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 3.2: Attempted Duplication (Why Copying Fails)")
        print("="*70)

    params = DETParams3D(N=32, DT=0.02)
    sim = DETCollider3D(params)

    rrs_params = RRSParams(
        coherence_budget_enabled=True,
        S_max=5.0
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Single large region for both original and copy
    rrs.add_adjacent_regions(
        body_slice=(slice(4, 14), slice(6, 26), slice(6, 26)),
        substrate_slice=(slice(18, 28), slice(6, 26), slice(6, 26))
    )

    # Create original cluster in BODY
    sim.add_packet((9, 16, 16), mass=15.0, width=4.0, initial_q=0.15)

    # Let it equilibrate and develop coherent structure
    for _ in range(300):
        rrs.step()

    # Record original state
    original_region = (slice(5, 13), slice(12, 20), slice(12, 20))
    original_F = sim.F[original_region].copy()
    original_theta = sim.theta[original_region].copy()
    original_a = sim.a[original_region].copy()

    original_participation = float(np.sum(sim.a[original_region] * sim.P[original_region]))

    if verbose:
        print(f"  Original participation before copy: {original_participation:.4f}")

    # ATTEMPT TO COPY to substrate region
    copy_region = (slice(19, 27), slice(12, 20), slice(12, 20))
    sim.F[copy_region] = original_F * copy_fidelity
    sim.theta[copy_region] = original_theta  # Copy phases
    sim.a[copy_region] = original_a * copy_fidelity

    if verbose:
        print(f"  Copy created with fidelity {copy_fidelity}")

    # Track both regions
    original_part_history = []
    copy_part_history = []
    n_clusters_history = []

    for step in range(max_steps):
        rrs.step()

        orig_part = float(np.sum(sim.a[original_region] * sim.P[original_region]))
        copy_part = float(np.sum(sim.a[copy_region] * sim.P[copy_region]))

        original_part_history.append(orig_part)
        copy_part_history.append(copy_part)

        _, clusters = rrs.compute_cluster_continuity()
        n_clusters_history.append(len(clusters))

        if verbose and step % 300 == 0:
            print(f"  Step {step}: orig_Π={orig_part:.4f}, copy_Π={copy_part:.4f}, "
                  f"clusters={len(clusters)}")

    # Analyze outcome
    final_orig = original_part_history[-1]
    final_copy = copy_part_history[-1]

    # Did the copy achieve independent viability?
    copy_viable = final_copy > 0.3 * final_orig
    both_survived = final_orig > 0.5 * original_participation and copy_viable

    if verbose:
        print(f"\n  Final original participation: {final_orig:.4f}")
        print(f"  Final copy participation: {final_copy:.4f}")
        print(f"  Copy viable: {copy_viable}")
        print(f"  Both survived (fork): {both_survived}")
        print(f"  Outcome: {'FORK' if both_survived else 'COPY FAILED'}")

    return {
        'original_participation_history': original_part_history,
        'copy_participation_history': copy_part_history,
        'n_clusters_history': n_clusters_history,
        'copy_viable': copy_viable,
        'both_survived': both_survived,
        'final_original': final_orig,
        'final_copy': final_copy
    }


# =============================================================================
# 4. BIOLOGICAL-ARTIFICIAL INTERFACE MODELING
# =============================================================================

def experiment_bridge_conductivity_effects(
    sigma_bridge_values: List[float] = [0.5, 1.0, 2.0, 5.0, 10.0],
    max_steps: int = 1500,
    verbose: bool = True
) -> Dict[float, Dict]:
    """
    Experiment 4.1: Bridge Conductivity and Migration Ease

    Models brain-computer interface "bandwidth":
    - Low σ_bridge: Weak coupling (limited data transfer)
    - High σ_bridge: Strong coupling (high-bandwidth interface)

    Key Questions:
    - What is the minimum σ_bridge for successful entrainment?
    - How does σ_bridge affect migration speed?
    - Is there an optimal σ_bridge for smooth transfer?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 4.1: Bridge Conductivity (BCI Bandwidth)")
        print("="*70)

    results = {}

    for sigma_bridge in sigma_bridge_values:
        if verbose:
            print(f"\n  Testing σ_bridge = {sigma_bridge}")

        params = DETParams3D(N=32, DT=0.02)
        sim = DETCollider3D(params)

        rrs_params = RRSParams(
            sigma_bridge=sigma_bridge,
            C_bridge_init=0.1,
            migration_window=50
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
            substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
        )

        # Initialize in BODY
        sim.add_packet((11, 16, 16), mass=15.0, width=4.0)

        # Apply pressure toward substrate
        R_BU_history = []
        migration_step = None

        for step in range(max_steps):
            # Gentle push toward substrate
            if step > 200:
                sim.F[rrs._body_mask] *= 0.9995
                sim.F[rrs._substrate_mask] += 0.001

            rrs.step()

            R_BU = rrs.compute_entrainment_order_parameter()
            R_BU_history.append(R_BU)

            locus, _ = rrs.compute_decision_locus()
            if migration_step is None and rrs._substrate_mask[locus]:
                migration_step = step

        results[sigma_bridge] = {
            'mean_R_BU': np.mean(R_BU_history),
            'max_R_BU': max(R_BU_history),
            'migration_step': migration_step,
            'migrated': migration_step is not None
        }

        if verbose:
            print(f"    Mean R_BU: {results[sigma_bridge]['mean_R_BU']:.4f}")
            print(f"    Migrated: {results[sigma_bridge]['migrated']}")
            if migration_step:
                print(f"    Migration step: {migration_step}")

    return results


def experiment_gradual_vs_instantaneous_interface(
    interface_growth_rate: float = 0.01,
    max_steps: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Experiment 4.2: Gradual Interface Development vs. Instant Connection

    Models different BCI deployment scenarios:
    - Gradual: Interface strength grows over time (learning/adaptation)
    - Instant: Full interface from step 0

    Key Questions:
    - Does gradual interface development improve entrainment?
    - Is there a "preparation" phase before migration becomes possible?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 4.2: Gradual vs. Instant Interface")
        print("="*70)

    results = {'gradual': {}, 'instant': {}}

    for mode in ['gradual', 'instant']:
        if verbose:
            print(f"\n  Testing mode: {mode}")

        params = DETParams3D(N=32, DT=0.02)
        sim = DETCollider3D(params)

        rrs_params = RRSParams(
            sigma_bridge=0.1 if mode == 'gradual' else 2.0,
            C_bridge_init=0.01 if mode == 'gradual' else 0.2
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
            substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
        )

        sim.add_packet((11, 16, 16), mass=15.0, width=4.0)

        R_BU_history = []
        continuity_history = []

        for step in range(max_steps):
            # Gradual interface growth
            if mode == 'gradual' and step < 1000:
                growth = interface_growth_rate * step / 1000
                for bond in rrs.bridge_bonds:
                    z, y, x = bond.i
                    if bond.direction == 'X':
                        sim.C_X[z, y, x] = min(0.5, 0.01 + growth * 0.5)
                    elif bond.direction == 'Y':
                        sim.C_Y[z, y, x] = min(0.5, 0.01 + growth * 0.5)
                    elif bond.direction == 'Z':
                        sim.C_Z[z, y, x] = min(0.5, 0.01 + growth * 0.5)

            # Apply migration pressure
            if step > 500:
                sim.F[rrs._body_mask] *= 0.999
                sim.F[rrs._substrate_mask] += 0.002

            rrs.step()

            R_BU_history.append(rrs.compute_entrainment_order_parameter())
            continuity_history.append(rrs.compute_cluster_continuity()[0])

        results[mode] = {
            'R_BU_history': R_BU_history,
            'continuity_history': continuity_history,
            'final_R_BU': R_BU_history[-1],
            'mean_continuity': np.mean(continuity_history),
            'min_continuity': min(continuity_history)
        }

        if verbose:
            print(f"    Final R_BU: {results[mode]['final_R_BU']:.4f}")
            print(f"    Min continuity: {results[mode]['min_continuity']:.4f}")

    return results


# =============================================================================
# 5. FAILURE MODE ANALYSIS
# =============================================================================

def experiment_prison_regime_conditions(
    coherence_boost_rates: List[float] = [0.0, 0.01, 0.05, 0.1],
    agency_suppression_rates: List[float] = [0.0, 0.01, 0.05, 0.1],
    max_steps: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Experiment 5.1: Conditions for Prison Regime (High C, Low a)

    The "prison regime" is a pathological state where coherence grows
    while agency collapses - the system is "captured" but not agentive.

    Key Questions:
    - Under what parameter combinations does prison emerge?
    - Is the prison regime stable or transient?
    - Can rolling replacement prevent prison formation?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 5.1: Prison Regime Conditions")
        print("="*70)

    results = {}

    for C_boost in coherence_boost_rates:
        for a_suppress in agency_suppression_rates:
            key = f"C_boost={C_boost}_a_suppress={a_suppress}"

            params = DETParams3D(N=32, DT=0.02, agency_dynamic=True)
            sim = DETCollider3D(params)

            rrs_params = RRSParams(rolling_replacement_enabled=True, q_max=0.7)
            rrs = RollingResonanceSubstrate(sim, rrs_params)

            rrs.add_adjacent_regions(
                body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
                substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
            )

            sim.add_packet((16, 16, 16), mass=15.0, width=5.0, initial_q=0.3)

            C_history = []
            a_history = []
            prison_steps = 0

            for step in range(max_steps):
                # Artificially manipulate to induce prison
                sim.C_X[rrs._substrate_mask] = np.clip(
                    sim.C_X[rrs._substrate_mask] + C_boost * 0.01, 0.15, 1.0)
                sim.C_Y[rrs._substrate_mask] = np.clip(
                    sim.C_Y[rrs._substrate_mask] + C_boost * 0.01, 0.15, 1.0)
                sim.C_Z[rrs._substrate_mask] = np.clip(
                    sim.C_Z[rrs._substrate_mask] + C_boost * 0.01, 0.15, 1.0)

                # Suppress agency (simulating external constraint)
                sim.a[rrs._substrate_mask] *= (1.0 - a_suppress * 0.01)

                rrs.step()

                mean_C = np.mean([
                    np.mean(sim.C_X[rrs._substrate_mask]),
                    np.mean(sim.C_Y[rrs._substrate_mask]),
                    np.mean(sim.C_Z[rrs._substrate_mask])
                ])
                mean_a = np.mean(sim.a[rrs._substrate_mask])

                C_history.append(mean_C)
                a_history.append(mean_a)

                # Prison: high C, low a
                if mean_C > 0.7 and mean_a < 0.2:
                    prison_steps += 1

            prison_fraction = prison_steps / max_steps

            results[key] = {
                'final_C': C_history[-1],
                'final_a': a_history[-1],
                'prison_fraction': prison_fraction,
                'is_prison': prison_fraction > 0.5
            }

            if verbose and prison_fraction > 0.1:
                print(f"  {key}: prison_frac={prison_fraction:.3f}, "
                      f"C={C_history[-1]:.3f}, a={a_history[-1]:.3f}")

    return results


def experiment_fork_emergence_conditions(
    resource_injection_rates: List[float] = [0.0, 0.05, 0.1, 0.2],
    max_steps: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Experiment 5.2: Fork Emergence Conditions

    Under what conditions can a single cluster bifurcate into two
    independently viable clusters (violating non-forkability)?

    Key Questions:
    - Does high resource injection enable forking?
    - Can coherence budget prevent fork even with high resources?
    - What is the signature of incipient fork formation?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 5.2: Fork Emergence Conditions")
        print("="*70)

    results = {}

    for injection_rate in resource_injection_rates:
        if verbose:
            print(f"\n  Testing injection_rate = {injection_rate}")

        params = DETParams3D(N=32, DT=0.02)
        sim = DETCollider3D(params)

        rrs_params = RRSParams(
            coherence_budget_enabled=True,
            S_max=6.0
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        # Wide regions to allow forking
        rrs.add_adjacent_regions(
            body_slice=(slice(4, 14), slice(4, 28), slice(4, 28)),
            substrate_slice=(slice(18, 28), slice(4, 28), slice(4, 28))
        )

        # Start with single cluster
        sim.add_packet((9, 16, 16), mass=20.0, width=5.0)

        n_clusters_history = []
        fork_detected = False
        fork_step = None

        for step in range(max_steps):
            # Inject resources (could enable forking)
            sim.F += injection_rate * sim.Delta_tau

            rrs.step()

            _, clusters = rrs.compute_cluster_continuity(threshold_ratio=15.0)
            n_clusters_history.append(len(clusters))

            # Sustained dual cluster = fork
            if len(clusters) >= 2 and not fork_detected:
                if len(n_clusters_history) > 50 and np.mean(n_clusters_history[-50:]) >= 1.5:
                    fork_detected = True
                    fork_step = step
                    if verbose:
                        print(f"    Fork detected at step {step}")

        results[injection_rate] = {
            'fork_detected': fork_detected,
            'fork_step': fork_step,
            'mean_clusters': np.mean(n_clusters_history),
            'max_clusters': max(n_clusters_history)
        }

        if verbose:
            print(f"    Fork: {fork_detected}, mean_clusters: {results[injection_rate]['mean_clusters']:.2f}")

    return results


# =============================================================================
# 6. EMERGENT COLLECTIVE PHENOMENA
# =============================================================================

def experiment_multi_agent_substrate(
    n_agents: int = 3,
    max_steps: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Experiment 6.1: Multiple Agents on Shared Substrate

    What happens when multiple coherent clusters attempt to use
    the same substrate region?

    Key Questions:
    - Do agents compete for substrate "bandwidth"?
    - Can multiple agents coexist on shared substrate?
    - Is there emergent cooperation or interference?
    """
    if verbose:
        print("\n" + "="*70)
        print(f"EXPERIMENT 6.1: Multi-Agent Substrate ({n_agents} agents)")
        print("="*70)

    params = DETParams3D(N=48, DT=0.02)  # Larger grid for multiple agents
    sim = DETCollider3D(params)

    rrs_params = RRSParams(
        coherence_budget_enabled=True,
        S_max=5.0,
        rolling_replacement_enabled=True,
        q_max=0.5
    )
    rrs = RollingResonanceSubstrate(sim, rrs_params)

    # Large shared substrate region
    rrs.add_adjacent_regions(
        body_slice=(slice(4, 20), slice(4, 44), slice(4, 44)),
        substrate_slice=(slice(28, 44), slice(4, 44), slice(4, 44))
    )

    # Initialize multiple agents in BODY region
    agent_positions = [
        (12, 16, 16),
        (12, 16, 32),
        (12, 32, 24)
    ][:n_agents]

    for i, pos in enumerate(agent_positions):
        sim.add_packet(pos, mass=10.0, width=3.0, initial_q=0.1)

    # Track each agent's cluster
    n_clusters_history = []
    total_continuity_history = []
    mean_participation_history = []

    for step in range(max_steps):
        # Apply migration pressure toward substrate
        if step > 200:
            sim.F[rrs._body_mask] *= 0.9995

        rrs.step()

        continuity, clusters = rrs.compute_cluster_continuity(threshold_ratio=8.0)
        n_clusters_history.append(len(clusters))
        total_continuity_history.append(continuity)

        Pi = rrs.compute_participation_strength()
        mean_participation_history.append(np.mean(Pi[rrs._substrate_mask]))

        if verbose and step % 400 == 0:
            print(f"  Step {step}: n_clusters={len(clusters)}, "
                  f"continuity={continuity:.4f}, mean_Π={mean_participation_history[-1]:.6f}")

    # Analyze: did agents merge, compete, or coexist?
    final_n_clusters = n_clusters_history[-1]
    coexistence = final_n_clusters >= n_agents
    merger = final_n_clusters == 1 and n_agents > 1
    competition = final_n_clusters < n_agents and final_n_clusters > 1

    if verbose:
        print(f"\n  Final n_clusters: {final_n_clusters}")
        print(f"  Outcome: {'COEXISTENCE' if coexistence else ('MERGER' if merger else 'COMPETITION')}")

    return {
        'n_clusters_history': n_clusters_history,
        'total_continuity_history': total_continuity_history,
        'mean_participation_history': mean_participation_history,
        'final_n_clusters': final_n_clusters,
        'coexistence': coexistence,
        'merger': merger,
        'competition': competition
    }


def experiment_substrate_capacity_limit(
    cluster_masses: List[float] = [5.0, 10.0, 20.0, 40.0],
    max_steps: int = 1500,
    verbose: bool = True
) -> Dict[float, Dict]:
    """
    Experiment 6.2: Substrate Carrying Capacity

    Is there a maximum "size" of coherent cluster that a given
    substrate can support?

    Key Questions:
    - How does cluster mass affect migration success?
    - Is there a substrate "carrying capacity"?
    - What happens when cluster exceeds capacity?
    """
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT 6.2: Substrate Carrying Capacity")
        print("="*70)

    results = {}

    for mass in cluster_masses:
        if verbose:
            print(f"\n  Testing cluster mass = {mass}")

        params = DETParams3D(N=32, DT=0.02)
        sim = DETCollider3D(params)

        rrs_params = RRSParams(
            coherence_budget_enabled=True,
            S_max=6.0
        )
        rrs = RollingResonanceSubstrate(sim, rrs_params)

        rrs.add_adjacent_regions(
            body_slice=(slice(6, 16), slice(8, 24), slice(8, 24)),
            substrate_slice=(slice(16, 26), slice(8, 24), slice(8, 24))
        )

        sim.add_packet((11, 16, 16), mass=mass, width=4.0)

        # Apply migration pressure
        substrate_F_history = []
        continuity_history = []

        for step in range(max_steps):
            if step > 200:
                sim.F[rrs._body_mask] *= 0.998
                sim.F[rrs._substrate_mask] += 0.005

            rrs.step()

            substrate_F_history.append(np.sum(sim.F[rrs._substrate_mask]))
            continuity_history.append(rrs.compute_cluster_continuity()[0])

        max_substrate_F = max(substrate_F_history)
        migration_success = max_substrate_F > 0.3 * mass

        results[mass] = {
            'max_substrate_F': max_substrate_F,
            'final_continuity': continuity_history[-1],
            'migration_success': migration_success
        }

        if verbose:
            print(f"    Max substrate F: {max_substrate_F:.2f}")
            print(f"    Migration success: {migration_success}")

    return results


# =============================================================================
# MAIN RESEARCH RUNNER
# =============================================================================

def run_all_experiments(verbose: bool = True):
    """Run all research experiments."""
    print("="*80)
    print("DET v6.3 RRS EXTENSION - RESEARCH EXPERIMENTS")
    print("="*80)

    results = {}

    # 1. Migration Dynamics
    print("\n" + "#"*80)
    print("# SECTION 1: MIGRATION DYNAMICS")
    print("#"*80)

    results['1.1_gradual_migration'] = experiment_gradual_migration(
        resource_shift_rate=0.005, max_steps=1500, verbose=verbose)

    results['1.2_sudden_transfer'] = experiment_sudden_transfer(
        transfer_fraction=0.7, max_steps=800, verbose=verbose)

    # 2. Longevity Mechanics
    print("\n" + "#"*80)
    print("# SECTION 2: LONGEVITY MECHANICS")
    print("#"*80)

    results['2.1_longevity_churn'] = experiment_longevity_under_churn(
        churn_rate=0.1, max_steps=3000, verbose=verbose)

    results['2.2_debt_accumulation'] = experiment_debt_accumulation_regimes(
        debt_injection_rate=0.005, max_steps=2000, verbose=verbose)

    # 3. Non-Forkability
    print("\n" + "#"*80)
    print("# SECTION 3: NON-FORKABILITY")
    print("#"*80)

    results['3.1_coherence_budget_forks'] = experiment_coherence_budget_fork_prevention(
        S_max_values=[3.0, 6.0, 12.0], max_steps=1500, verbose=verbose)

    results['3.2_attempted_duplication'] = experiment_attempted_duplication(
        copy_fidelity=0.9, max_steps=1000, verbose=verbose)

    # 4. Biological-Artificial Interface
    print("\n" + "#"*80)
    print("# SECTION 4: BIO-ARTIFICIAL INTERFACE")
    print("#"*80)

    results['4.1_bridge_conductivity'] = experiment_bridge_conductivity_effects(
        sigma_bridge_values=[0.5, 2.0, 5.0], max_steps=1200, verbose=verbose)

    # 5. Failure Modes
    print("\n" + "#"*80)
    print("# SECTION 5: FAILURE MODES")
    print("#"*80)

    results['5.2_fork_emergence'] = experiment_fork_emergence_conditions(
        resource_injection_rates=[0.0, 0.1], max_steps=1500, verbose=verbose)

    # 6. Collective Phenomena
    print("\n" + "#"*80)
    print("# SECTION 6: COLLECTIVE PHENOMENA")
    print("#"*80)

    results['6.1_multi_agent'] = experiment_multi_agent_substrate(
        n_agents=2, max_steps=1500, verbose=verbose)

    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)

    return results


if __name__ == "__main__":
    results = run_all_experiments(verbose=True)
