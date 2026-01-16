"""
DET v6.4 Quantum-Classical Transition: Agency-Coherence Interplay
=================================================================

Roadmap Item #6: Study how DET transitions between quantum-like coherent
behavior and classical definite-outcome behavior.

Theoretical Framework
---------------------
In DET, the quantum-classical transition is governed by:

1. **Coherence (C):** Tracks phase relationships between neighboring nodes
   - High C: Quantum regime (entanglement-like correlations)
   - Low C: Classical regime (definite outcomes)

2. **Agency (a):** Tracks decision-making capacity
   - Bounded by structural ceiling: a_max = 1/(1 + λ_a * q²)
   - Governs responsiveness to relational gradients

3. **Presence (P):** Effective clock rate
   - P = a * σ / (1+F) / (1+H)
   - Vanishes for frozen/classical states

Key Dynamics
------------
**Coherence evolution:**
    dC/dt = α_C * |J_diff| - λ_C * C - λ_M * M * C

where:
- α_C: coherence growth from flux
- λ_C: natural decoherence rate
- λ_M: measurement-induced decoherence
- M: detection/measurement indicator

**Agency evolution:**
    da/dt = β_a * (a_max - a) + γ_a * relational_drive

**Quantum gate in grace:**
    Q_ij = exp(-C_ij / C_threshold)

Reference: DET Theory Card v6.3, Sections III, VI
"""

import numpy as np
from scipy.fft import fftn, ifftn
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from det_v6_3_3d_collider import DETCollider3D, DETParams3D, compute_lattice_correction


# ==============================================================================
# PHYSICAL CONSTANTS
# ==============================================================================

# Decoherence thresholds
C_QUANTUM_THRESHOLD = 0.5    # Above this = quantum regime
C_CLASSICAL_THRESHOLD = 0.1  # Below this = classical regime


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass
class CoherenceState:
    """State of coherence field."""
    C_mean: float               # Mean coherence
    C_max: float                # Maximum coherence
    C_std: float                # Standard deviation
    quantum_fraction: float     # Fraction of nodes in quantum regime
    classical_fraction: float   # Fraction in classical regime


@dataclass
class AgencyState:
    """State of agency field."""
    a_mean: float               # Mean agency
    a_max_achieved: float       # Maximum agency achieved
    a_ceiling_mean: float       # Mean structural ceiling
    constrained_fraction: float # Fraction at ceiling


@dataclass
class DecoherenceResult:
    """Result of decoherence simulation."""
    time_steps: np.ndarray
    coherence_history: np.ndarray  # Mean C over time
    decoherence_rate: float        # Fitted decay rate
    decoherence_time: float        # Time to reach C_classical
    initial_coherence: float
    final_coherence: float


@dataclass
class EntanglementMetrics:
    """Metrics for entanglement-like correlations."""
    correlation_strength: float     # Spatial correlation of phases
    bell_parameter: float           # CHSH-like correlation measure
    locality_violation: float       # Degree of non-local correlation
    coherence_correlation: float    # C-correlation correlation


@dataclass
class TransitionAnalysis:
    """Complete quantum-classical transition analysis."""
    coherence_state: CoherenceState
    agency_state: AgencyState
    decoherence: DecoherenceResult
    entanglement: EntanglementMetrics
    regime: str  # 'quantum', 'classical', or 'transition'
    summary: str = ""


# ==============================================================================
# COHERENCE ANALYSIS
# ==============================================================================

class CoherenceAnalyzer:
    """
    Analyze coherence dynamics in DET simulations.

    Coherence C governs quantum-like behavior:
    - High C: phase correlations maintained (quantum)
    - Low C: phases randomized (classical)
    """

    def __init__(self, verbose: bool = True):
        """Initialize coherence analyzer."""
        self.verbose = verbose

    def get_mean_coherence(self, sim: DETCollider3D) -> np.ndarray:
        """
        Get mean coherence per node from bond coherences.

        DET stores coherence per-bond (C_X, C_Y, C_Z).
        We compute the average coherence for each node.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        np.ndarray
            Mean coherence per node
        """
        # Average of the three bond directions
        return (sim.C_X + sim.C_Y + sim.C_Z) / 3.0

    def measure_coherence_state(self, sim: DETCollider3D) -> CoherenceState:
        """
        Measure current coherence state.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        CoherenceState
            Current coherence statistics
        """
        C = self.get_mean_coherence(sim)

        C_mean = np.mean(C)
        C_max = np.max(C)
        C_std = np.std(C)

        # Count nodes in each regime
        quantum_mask = C > C_QUANTUM_THRESHOLD
        classical_mask = C < C_CLASSICAL_THRESHOLD

        total_nodes = C.size
        quantum_fraction = np.sum(quantum_mask) / total_nodes
        classical_fraction = np.sum(classical_mask) / total_nodes

        return CoherenceState(
            C_mean=C_mean,
            C_max=C_max,
            C_std=C_std,
            quantum_fraction=quantum_fraction,
            classical_fraction=classical_fraction
        )

    def compute_phase_correlation(self, sim: DETCollider3D,
                                   max_distance: int = 10) -> np.ndarray:
        """
        Compute phase correlation vs distance.

        C(r) = <cos(θ_i - θ_j)> for |i-j| = r

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation
        max_distance : int
            Maximum separation to compute

        Returns
        -------
        np.ndarray
            Correlation at each distance
        """
        theta = sim.theta
        N = theta.shape[0]

        correlations = []

        for r in range(1, max_distance + 1):
            # Sample correlations at distance r
            cos_diffs = []

            # Along x-axis
            for i in range(N - r):
                for j in range(N):
                    for k in range(N):
                        dtheta = theta[i, j, k] - theta[i+r, j, k]
                        cos_diffs.append(np.cos(dtheta))

            correlations.append(np.mean(cos_diffs))

        return np.array(correlations)

    def measure_coherence_weighted_correlation(self, sim: DETCollider3D) -> float:
        """
        Measure coherence-weighted phase correlation.

        This captures how much coherence enhances correlations.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        float
            Coherence-weighted correlation strength
        """
        C = self.get_mean_coherence(sim)
        theta = sim.theta
        N = C.shape[0]

        # Compute weighted correlation with nearest neighbors
        weighted_corr = 0.0
        count = 0

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    # Check neighbors
                    for di, dj, dk in [(1,0,0), (0,1,0), (0,0,1)]:
                        ni = (i + di) % N
                        nj = (j + dj) % N
                        nk = (k + dk) % N

                        # Coherence between nodes
                        C_ij = np.sqrt(C[i,j,k] * C[ni,nj,nk])

                        # Phase correlation
                        phase_corr = np.cos(theta[i,j,k] - theta[ni,nj,nk])

                        weighted_corr += C_ij * phase_corr
                        count += 1

        return weighted_corr / count if count > 0 else 0.0


# ==============================================================================
# AGENCY ANALYSIS
# ==============================================================================

class AgencyAnalyzer:
    """
    Analyze agency dynamics and structural constraints.

    Agency a represents decision-making capacity, bounded by:
    a_max = 1 / (1 + λ_a * q²)
    """

    def __init__(self, lambda_a: float = 30.0, verbose: bool = True):
        """
        Initialize agency analyzer.

        Parameters
        ----------
        lambda_a : float
            Structural ceiling coupling
        verbose : bool
            Print progress
        """
        self.lambda_a = lambda_a
        self.verbose = verbose

    def compute_agency_ceiling(self, q: np.ndarray) -> np.ndarray:
        """
        Compute agency ceiling from structural debt.

        a_max = 1 / (1 + λ_a * q²)

        Parameters
        ----------
        q : np.ndarray
            Structural debt field

        Returns
        -------
        np.ndarray
            Agency ceiling field
        """
        return 1.0 / (1.0 + self.lambda_a * q**2)

    def measure_agency_state(self, sim: DETCollider3D) -> AgencyState:
        """
        Measure current agency state.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        AgencyState
            Current agency statistics
        """
        a = sim.a
        q = sim.q

        a_ceiling = self.compute_agency_ceiling(q)

        a_mean = np.mean(a)
        a_max_achieved = np.max(a)
        a_ceiling_mean = np.mean(a_ceiling)

        # Fraction of nodes at or near ceiling
        constrained_mask = a >= 0.95 * a_ceiling
        constrained_fraction = np.sum(constrained_mask) / a.size

        return AgencyState(
            a_mean=a_mean,
            a_max_achieved=a_max_achieved,
            a_ceiling_mean=a_ceiling_mean,
            constrained_fraction=constrained_fraction
        )

    def compute_agency_coherence_correlation(self, sim: DETCollider3D) -> float:
        """
        Compute correlation between agency and coherence.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        float
            Pearson correlation coefficient
        """
        a_flat = sim.a.flatten()
        C = (sim.C_X + sim.C_Y + sim.C_Z) / 3.0
        C_flat = C.flatten()

        # Remove any NaN or inf values
        valid = np.isfinite(a_flat) & np.isfinite(C_flat)

        if np.sum(valid) < 10:
            return 0.0

        corr, _ = pearsonr(a_flat[valid], C_flat[valid])
        return corr if np.isfinite(corr) else 0.0


# ==============================================================================
# DECOHERENCE SIMULATION
# ==============================================================================

class DecoherenceSimulator:
    """
    Simulate decoherence (quantum to classical transition).

    Studies how coherence decays under:
    - Natural decoherence (λ_C term)
    - Measurement-induced decoherence
    - Environmental interaction
    """

    def __init__(self, grid_size: int = 32, verbose: bool = True):
        """
        Initialize decoherence simulator.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        verbose : bool
            Print progress
        """
        self.N = grid_size
        self.verbose = verbose

    def setup_coherent_state(self, C_init: float = 0.8) -> DETCollider3D:
        """
        Create simulation with high initial coherence.

        Parameters
        ----------
        C_init : float
            Initial coherence level

        Returns
        -------
        DETCollider3D
            Simulation in coherent state
        """
        params = DETParams3D(
            N=self.N,
            DT=0.02,
            F_VAC=0.1,
            F_MIN=0.0,
            C_init=C_init,

            # Enable coherence dynamics
            coherence_dynamic=True,
            alpha_C=0.02,
            lambda_C=0.01,  # Natural decoherence rate

            # Enable other dynamics
            momentum_enabled=True,
            gravity_enabled=True,
            kappa_grav=5.0,
            q_enabled=True,
            agency_dynamic=True,

            # Disable floor to avoid interference
            floor_enabled=False,
            boundary_enabled=False
        )

        sim = DETCollider3D(params)

        # Set uniform high coherence (on all bonds)
        sim.C_X = np.ones((self.N, self.N, self.N)) * C_init
        sim.C_Y = np.ones((self.N, self.N, self.N)) * C_init
        sim.C_Z = np.ones((self.N, self.N, self.N)) * C_init

        # Align phases for coherent state
        sim.theta = np.zeros((self.N, self.N, self.N))

        return sim

    def run_decoherence_simulation(self, C_init: float = 0.8,
                                    n_steps: int = 500,
                                    sample_interval: int = 10) -> DecoherenceResult:
        """
        Run decoherence simulation and track coherence decay.

        Parameters
        ----------
        C_init : float
            Initial coherence
        n_steps : int
            Total simulation steps
        sample_interval : int
            Steps between measurements

        Returns
        -------
        DecoherenceResult
            Decoherence analysis results
        """
        if self.verbose:
            print(f"\nDecoherence Simulation (C_init = {C_init})")
            print("-" * 50)

        sim = self.setup_coherent_state(C_init)

        time_steps = []
        coherence_history = []

        for t in range(0, n_steps, sample_interval):
            C_mean = np.mean((sim.C_X + sim.C_Y + sim.C_Z) / 3.0)
            time_steps.append(t)
            coherence_history.append(C_mean)

            # Evolve
            for _ in range(sample_interval):
                sim.step()

        time_steps = np.array(time_steps)
        coherence_history = np.array(coherence_history)

        # Fit exponential decay: C(t) = C_0 * exp(-t/τ)
        valid = coherence_history > 0.01
        if np.sum(valid) > 3:
            try:
                log_C = np.log(coherence_history[valid])
                t_valid = time_steps[valid]
                coeffs = np.polyfit(t_valid, log_C, 1)
                decoherence_rate = -coeffs[0]  # 1/τ
                decoherence_time = 1.0 / decoherence_rate if decoherence_rate > 0 else float('inf')
            except:
                decoherence_rate = 0.0
                decoherence_time = float('inf')
        else:
            decoherence_rate = 0.0
            decoherence_time = float('inf')

        if self.verbose:
            print(f"  Initial C: {coherence_history[0]:.4f}")
            print(f"  Final C: {coherence_history[-1]:.4f}")
            print(f"  Decoherence rate: {decoherence_rate:.6f}")
            print(f"  Decoherence time: {decoherence_time:.2f}")

        return DecoherenceResult(
            time_steps=time_steps,
            coherence_history=coherence_history,
            decoherence_rate=decoherence_rate,
            decoherence_time=decoherence_time,
            initial_coherence=coherence_history[0],
            final_coherence=coherence_history[-1]
        )

    def simulate_measurement(self, sim: DETCollider3D,
                              measurement_strength: float = 0.5) -> float:
        """
        Simulate a measurement that causes decoherence.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation
        measurement_strength : float
            Strength of measurement (0-1)

        Returns
        -------
        float
            Coherence after measurement
        """
        # Measurement causes coherence to decay
        # C_new = C * (1 - measurement_strength)
        sim.C_X *= (1 - measurement_strength)
        sim.C_Y *= (1 - measurement_strength)
        sim.C_Z *= (1 - measurement_strength)

        # Also randomize phases (wavefunction collapse analog)
        collapse_mask = np.random.random(sim.theta.shape) < measurement_strength
        sim.theta[collapse_mask] = np.random.uniform(0, 2*np.pi, np.sum(collapse_mask))

        return np.mean((sim.C_X + sim.C_Y + sim.C_Z) / 3.0)


# ==============================================================================
# ENTANGLEMENT-LIKE CORRELATIONS
# ==============================================================================

class EntanglementAnalyzer:
    """
    Analyze entanglement-like correlations in DET.

    In the quantum regime (high C), DET exhibits:
    - Long-range phase correlations
    - Bell-like correlation violations
    - Non-local behavior
    """

    def __init__(self, verbose: bool = True):
        """Initialize entanglement analyzer."""
        self.verbose = verbose

    def compute_bell_parameter(self, sim: DETCollider3D,
                                n_samples: int = 1000) -> float:
        """
        Compute CHSH-like Bell parameter.

        The parameter S measures correlation strength:
        - S ≤ 2: Classical (local hidden variable)
        - 2 < S ≤ 2√2: Quantum
        - S > 2√2: Super-quantum (not physical)

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation
        n_samples : int
            Number of correlation samples

        Returns
        -------
        float
            Bell parameter S
        """
        theta = sim.theta
        C = (sim.C_X + sim.C_Y + sim.C_Z) / 3.0
        N = theta.shape[0]

        # Sample pairs of nodes
        correlations = []

        for _ in range(n_samples):
            # Random pair of nodes
            i1, j1, k1 = np.random.randint(0, N, 3)
            i2, j2, k2 = np.random.randint(0, N, 3)

            # Skip if same node
            if (i1, j1, k1) == (i2, j2, k2):
                continue

            # Coherence between nodes
            C_ij = np.sqrt(C[i1, j1, k1] * C[i2, j2, k2])

            # Phase correlation
            theta1 = theta[i1, j1, k1]
            theta2 = theta[i2, j2, k2]

            # Measurement settings (random angles)
            a, a_prime = np.random.uniform(0, np.pi, 2)
            b, b_prime = np.random.uniform(0, np.pi, 2)

            # Correlation functions (quantum-like)
            # E(a,b) = -cos(a-b) for maximally entangled state
            # In DET, modulated by coherence
            E_ab = -C_ij * np.cos(theta1 - theta2 + a - b)
            E_ab_prime = -C_ij * np.cos(theta1 - theta2 + a - b_prime)
            E_a_prime_b = -C_ij * np.cos(theta1 - theta2 + a_prime - b)
            E_a_prime_b_prime = -C_ij * np.cos(theta1 - theta2 + a_prime - b_prime)

            # CHSH combination
            S = abs(E_ab - E_ab_prime + E_a_prime_b + E_a_prime_b_prime)
            correlations.append(S)

        return np.mean(correlations) if correlations else 0.0

    def compute_locality_violation(self, sim: DETCollider3D) -> float:
        """
        Measure degree of locality violation.

        Compares correlations at different separations to detect
        non-local (faster than expected) correlation propagation.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        float
            Locality violation measure (0 = local, 1 = maximally non-local)
        """
        theta = sim.theta
        C = (sim.C_X + sim.C_Y + sim.C_Z) / 3.0
        N = theta.shape[0]

        # Compute correlation at different separations
        short_range_corr = []  # r = 1-2
        long_range_corr = []   # r = 5-10

        for _ in range(500):
            i, j, k = np.random.randint(2, N-2, 3)

            # Short range
            for r in [1, 2]:
                ni = (i + r) % N
                C_ij = np.sqrt(C[i,j,k] * C[ni,j,k])
                phase_corr = np.cos(theta[i,j,k] - theta[ni,j,k])
                short_range_corr.append(C_ij * abs(phase_corr))

            # Long range
            for r in [5, 8, 10]:
                ni = (i + r) % N
                C_ij = np.sqrt(C[i,j,k] * C[ni,j,k])
                phase_corr = np.cos(theta[i,j,k] - theta[ni,j,k])
                long_range_corr.append(C_ij * abs(phase_corr))

        short_mean = np.mean(short_range_corr) if short_range_corr else 0
        long_mean = np.mean(long_range_corr) if long_range_corr else 0

        # Locality violation = long-range correlation relative to short-range
        # Classical: long << short (exponential decay)
        # Non-local: long ~ short
        if short_mean > 0:
            violation = long_mean / short_mean
        else:
            violation = 0.0

        return min(violation, 1.0)

    def measure_entanglement_metrics(self, sim: DETCollider3D) -> EntanglementMetrics:
        """
        Measure all entanglement-like metrics.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        EntanglementMetrics
            Complete entanglement analysis
        """
        # Correlation strength (mean coherence-weighted correlation)
        coherence_analyzer = CoherenceAnalyzer(verbose=False)
        correlation_strength = coherence_analyzer.measure_coherence_weighted_correlation(sim)

        # Bell parameter
        bell_parameter = self.compute_bell_parameter(sim)

        # Locality violation
        locality_violation = self.compute_locality_violation(sim)

        # Coherence-correlation correlation
        # How much does higher C lead to higher correlations?
        C = (sim.C_X + sim.C_Y + sim.C_Z) / 3.0
        theta = sim.theta
        N = C.shape[0]

        local_C = []
        local_corr = []

        for i in range(N):
            for j in range(N):
                for k in range(N):
                    ni = (i + 1) % N
                    C_local = C[i, j, k]
                    corr = abs(np.cos(theta[i, j, k] - theta[ni, j, k]))
                    local_C.append(C_local)
                    local_corr.append(corr)

        if len(local_C) > 10:
            coherence_corr, _ = pearsonr(local_C, local_corr)
            coherence_corr = coherence_corr if np.isfinite(coherence_corr) else 0.0
        else:
            coherence_corr = 0.0

        return EntanglementMetrics(
            correlation_strength=correlation_strength,
            bell_parameter=bell_parameter,
            locality_violation=locality_violation,
            coherence_correlation=coherence_corr
        )


# ==============================================================================
# REGIME CLASSIFICATION
# ==============================================================================

class RegimeClassifier:
    """
    Classify simulation into quantum, classical, or transition regime.
    """

    def __init__(self, verbose: bool = True):
        """Initialize regime classifier."""
        self.verbose = verbose

    def classify(self, coherence_state: CoherenceState,
                 entanglement: EntanglementMetrics) -> str:
        """
        Classify the quantum-classical regime.

        Parameters
        ----------
        coherence_state : CoherenceState
            Coherence statistics
        entanglement : EntanglementMetrics
            Entanglement metrics

        Returns
        -------
        str
            'quantum', 'classical', or 'transition'
        """
        # Quantum indicators
        quantum_score = 0

        # High mean coherence
        if coherence_state.C_mean > C_QUANTUM_THRESHOLD:
            quantum_score += 1

        # Large quantum fraction
        if coherence_state.quantum_fraction > 0.5:
            quantum_score += 1

        # Bell violation (S > 2)
        if entanglement.bell_parameter > 2.0:
            quantum_score += 1

        # Strong locality violation
        if entanglement.locality_violation > 0.3:
            quantum_score += 1

        # Classical indicators
        classical_score = 0

        # Low mean coherence
        if coherence_state.C_mean < C_CLASSICAL_THRESHOLD:
            classical_score += 1

        # Large classical fraction
        if coherence_state.classical_fraction > 0.5:
            classical_score += 1

        # No Bell violation
        if entanglement.bell_parameter <= 2.0:
            classical_score += 1

        # Weak locality violation
        if entanglement.locality_violation < 0.1:
            classical_score += 1

        # Classify
        if quantum_score >= 3:
            regime = 'quantum'
        elif classical_score >= 3:
            regime = 'classical'
        else:
            regime = 'transition'

        if self.verbose:
            print(f"\nRegime Classification:")
            print(f"  Quantum score: {quantum_score}/4")
            print(f"  Classical score: {classical_score}/4")
            print(f"  Regime: {regime.upper()}")

        return regime


# ==============================================================================
# COMPREHENSIVE ANALYSIS
# ==============================================================================

class QuantumClassicalAnalyzer:
    """
    Complete quantum-classical transition analysis for DET.
    """

    def __init__(self, grid_size: int = 32, verbose: bool = True):
        """
        Initialize analyzer.

        Parameters
        ----------
        grid_size : int
            Simulation grid size
        verbose : bool
            Print progress
        """
        self.N = grid_size
        self.verbose = verbose

        self.coherence_analyzer = CoherenceAnalyzer(verbose)
        self.agency_analyzer = AgencyAnalyzer(verbose=verbose)
        self.decoherence_sim = DecoherenceSimulator(grid_size, verbose)
        self.entanglement_analyzer = EntanglementAnalyzer(verbose)
        self.regime_classifier = RegimeClassifier(verbose)

    def analyze_state(self, sim: DETCollider3D) -> TransitionAnalysis:
        """
        Analyze quantum-classical state of simulation.

        Parameters
        ----------
        sim : DETCollider3D
            DET simulation

        Returns
        -------
        TransitionAnalysis
            Complete analysis
        """
        if self.verbose:
            print("\n" + "-"*50)
            print("STATE ANALYSIS")
            print("-"*50)

        # Coherence state
        coherence_state = self.coherence_analyzer.measure_coherence_state(sim)
        if self.verbose:
            print(f"  Mean coherence: {coherence_state.C_mean:.4f}")
            print(f"  Quantum fraction: {coherence_state.quantum_fraction*100:.1f}%")
            print(f"  Classical fraction: {coherence_state.classical_fraction*100:.1f}%")

        # Agency state
        agency_state = self.agency_analyzer.measure_agency_state(sim)
        if self.verbose:
            print(f"  Mean agency: {agency_state.a_mean:.4f}")
            print(f"  Agency ceiling: {agency_state.a_ceiling_mean:.4f}")

        # Entanglement metrics
        entanglement = self.entanglement_analyzer.measure_entanglement_metrics(sim)
        if self.verbose:
            print(f"  Bell parameter: {entanglement.bell_parameter:.4f}")
            print(f"  Locality violation: {entanglement.locality_violation:.4f}")

        # Classify regime
        regime = self.regime_classifier.classify(coherence_state, entanglement)

        # Create dummy decoherence result (actual decoherence needs separate simulation)
        decoherence = DecoherenceResult(
            time_steps=np.array([0]),
            coherence_history=np.array([coherence_state.C_mean]),
            decoherence_rate=0.0,
            decoherence_time=float('inf'),
            initial_coherence=coherence_state.C_mean,
            final_coherence=coherence_state.C_mean
        )

        return TransitionAnalysis(
            coherence_state=coherence_state,
            agency_state=agency_state,
            decoherence=decoherence,
            entanglement=entanglement,
            regime=regime
        )

    def run_full_analysis(self, C_init_values: List[float] = None,
                           decoherence_steps: int = 300) -> Dict:
        """
        Run complete quantum-classical transition analysis.

        Parameters
        ----------
        C_init_values : List[float]
            Initial coherence values to test
        decoherence_steps : int
            Steps for decoherence simulation

        Returns
        -------
        Dict
            Complete analysis results
        """
        if C_init_values is None:
            C_init_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        if self.verbose:
            print("\n" + "="*70)
            print("DET v6.4 QUANTUM-CLASSICAL TRANSITION ANALYSIS")
            print("="*70)
            print(f"\nGrid size: {self.N}")
            print(f"Testing coherence levels: {C_init_values}")

        results = {
            'analyses': [],
            'decoherence_results': [],
            'regime_summary': {}
        }

        # Analyze at different coherence levels
        for C_init in C_init_values:
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"COHERENCE LEVEL: C_init = {C_init}")
                print("="*50)

            # Create simulation
            sim = self.decoherence_sim.setup_coherent_state(C_init)

            # Run some steps to establish dynamics
            for _ in range(50):
                sim.step()

            # Analyze state
            analysis = self.analyze_state(sim)
            results['analyses'].append({
                'C_init': C_init,
                'analysis': analysis
            })

            # Run decoherence simulation
            decoh_result = self.decoherence_sim.run_decoherence_simulation(
                C_init=C_init,
                n_steps=decoherence_steps
            )
            results['decoherence_results'].append({
                'C_init': C_init,
                'result': decoh_result
            })

        # Summarize regimes
        regimes = [a['analysis'].regime for a in results['analyses']]
        results['regime_summary'] = {
            'quantum_count': regimes.count('quantum'),
            'classical_count': regimes.count('classical'),
            'transition_count': regimes.count('transition')
        }

        # Generate summary
        summary_lines = [
            "",
            "="*70,
            "QUANTUM-CLASSICAL TRANSITION SUMMARY",
            "="*70,
            "",
            "Coherence Level Analysis:",
        ]

        for item in results['analyses']:
            C_init = item['C_init']
            analysis = item['analysis']
            summary_lines.append(
                f"  C_init={C_init:.1f}: regime={analysis.regime}, "
                f"Bell={analysis.entanglement.bell_parameter:.3f}, "
                f"locality={analysis.entanglement.locality_violation:.3f}"
            )

        summary_lines.extend([
            "",
            "Decoherence Rates:",
        ])

        for item in results['decoherence_results']:
            C_init = item['C_init']
            result = item['result']
            summary_lines.append(
                f"  C_init={C_init:.1f}: rate={result.decoherence_rate:.6f}, "
                f"τ={result.decoherence_time:.1f}"
            )

        summary_lines.extend([
            "",
            "Regime Summary:",
            f"  Quantum: {results['regime_summary']['quantum_count']}",
            f"  Classical: {results['regime_summary']['classical_count']}",
            f"  Transition: {results['regime_summary']['transition_count']}",
            "",
            "="*70
        ])

        results['summary'] = "\n".join(summary_lines)

        if self.verbose:
            print(results['summary'])

        return results


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def run_quantum_classical_analysis(grid_size: int = 32,
                                    C_init_values: List[float] = None,
                                    verbose: bool = True) -> Dict:
    """
    Run complete quantum-classical transition analysis.

    This is the main entry point for the v6.4 Quantum-Classical Transition feature.

    Parameters
    ----------
    grid_size : int
        Simulation grid size
    C_init_values : List[float]
        Initial coherence values to test
    verbose : bool
        Print progress

    Returns
    -------
    Dict
        Complete analysis results
    """
    analyzer = QuantumClassicalAnalyzer(grid_size, verbose)
    return analyzer.run_full_analysis(C_init_values=C_init_values)


if __name__ == "__main__":
    print("DET v6.4 Quantum-Classical Transition")
    print("Studying agency-coherence interplay...")
    print()

    # Run analysis
    results = run_quantum_classical_analysis(grid_size=32, verbose=True)

    # Print final status
    print("\n" + "="*70)
    print("QUANTUM-CLASSICAL TRANSITION ANALYSIS COMPLETE")
    print("="*70)
