"""
DET v6 Gravitational Redshift Falsifier Test
=============================================

DET Prediction (Theory Card Section III + V):
    Δτ/τ = ΔΦ (in natural units)
    Equivalent to: P_i / P_∞ = 1 + Φ_i (weak field)
    
Falsifier Criterion:
    If clock rates don't match potential: |P(r)/P_∞ - (1 + Φ(r))| > 1%

Test Setup:
    - Place stationary spherical mass at center with q > 0
    - Let gravity field stabilize
    - Measure presence P(r) at different radii
    - Compare to Φ(r) from Poisson solver

Reference: DET Theory Card v6.0, Sections III.1, V.1-V.2
"""

import numpy as np
from scipy.fft import fftn, ifftn
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import matplotlib.pyplot as plt


@dataclass
class RedshiftTestParams:
    """Parameters for gravitational redshift test."""
    N: int = 64          # Grid size (larger for better resolution)
    DT: float = 0.01     # Time step
    F_VAC: float = 0.001 # Vacuum resource level
    F_MIN: float = 0.0
    
    # Coherence
    C_init: float = 0.1
    
    # Diffusive flux (reduced for stability)
    diff_enabled: bool = True
    
    # Disable momentum for static test
    momentum_enabled: bool = False
    
    # Disable angular momentum
    angular_momentum_enabled: bool = False
    
    # Floor repulsion disabled
    floor_enabled: bool = False
    
    # Structure (q-locking) - static test uses fixed q
    q_enabled: bool = False
    
    # Agency dynamics
    agency_dynamic: bool = False  # Keep agency fixed for clean test
    
    # Gravity module
    gravity_enabled: bool = True
    alpha_grav: float = 0.02     # Helmholtz screening
    kappa_grav: float = 5.0      # Poisson coupling
    mu_grav: float = 2.0         # Gravitational flux strength


class RedshiftTestSimulator:
    """
    Simplified DET simulator for gravitational redshift testing.
    
    This tests the fundamental relationship between presence P and
    gravitational potential Φ in a static configuration.
    """
    
    def __init__(self, params: Optional[RedshiftTestParams] = None):
        self.p = params or RedshiftTestParams()
        N = self.p.N
        
        # Per-node state
        self.F = np.ones((N, N, N), dtype=np.float64) * self.p.F_VAC
        self.q = np.zeros((N, N, N), dtype=np.float64)
        self.a = np.ones((N, N, N), dtype=np.float64)  # Full agency everywhere
        
        # Coherence (uniform)
        self.C_X = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        self.C_Y = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        self.C_Z = np.ones((N, N, N), dtype=np.float64) * self.p.C_init
        
        self.sigma = np.ones((N, N, N), dtype=np.float64)
        
        # Gravity fields
        self.b = np.zeros((N, N, N), dtype=np.float64)
        self.Phi = np.zeros((N, N, N), dtype=np.float64)
        self.rho = np.zeros((N, N, N), dtype=np.float64)
        
        # Presence
        self.P = np.ones((N, N, N), dtype=np.float64)
        
        # Precompute FFT wavenumbers
        self._setup_fft_solvers()
    
    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers for spectral solvers."""
        N = self.p.N
        kx = np.fft.fftfreq(N) * N
        ky = np.fft.fftfreq(N) * N
        kz = np.fft.fftfreq(N) * N
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        # Discrete 3D Laplacian eigenvalues
        self.L_k = -4 * (np.sin(np.pi * KX / N)**2 + 
                        np.sin(np.pi * KY / N)**2 + 
                        np.sin(np.pi * KZ / N)**2)
        
        # Helmholtz operator: (L - α)
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        
        # Poisson (avoiding zero mode)
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0, 0, 0] = 1.0
    
    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        """Solve Helmholtz equation: (L - α)b = -α * source"""
        source_k = fftn(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifftn(b_k))
    
    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve Poisson equation: L Φ = -κ * source"""
        source_k = fftn(source)
        source_k[0, 0, 0] = 0  # Remove mean
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0, 0, 0] = 0
        return np.real(ifftn(Phi_k))
    
    def add_central_mass(self, q_amplitude: float = 0.5, width: float = 4.0,
                          F_amplitude: float = 5.0):
        """
        Add a spherical mass distribution at center with structure q.
        
        Args:
            q_amplitude: Peak structural debt (sources gravity)
            width: Gaussian width
            F_amplitude: Resource amplitude (affects presence via 1/(1+F))
        """
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        center = N // 2
        
        # Periodic distance from center
        dx = (x - center + N/2) % N - N/2
        dy = (y - center + N/2) % N - N/2
        dz = (z - center + N/2) % N - N/2
        
        r2 = dx**2 + dy**2 + dz**2
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.q = q_amplitude * envelope
        self.F = self.p.F_VAC + F_amplitude * envelope
    
    def compute_gravity(self):
        """Compute gravitational fields from structural debt q."""
        # Helmholtz baseline: (L - α)b = -α * q
        self.b = self._solve_helmholtz(self.q)
        
        # Relative source (gravity source)
        self.rho = self.q - self.b
        
        # Poisson potential: L Φ = -κ * ρ
        self.Phi = self._solve_poisson(self.rho)
    
    def compute_presence(self):
        """
        Compute presence field according to Theory Card III.1:
        P_i = a_i * σ_i / (1 + F_i^op) / (1 + H_i)
        
        Using Option A (degenerate): H_i = σ_i
        """
        H = self.sigma  # Option A: H = σ
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
    
    def get_radial_profile(self, field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radially averaged profile of a field.
        
        Returns:
            r: Array of radii
            profile: Radially averaged field values
        """
        N = self.p.N
        center = N // 2
        
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        
        # Periodic distance from center
        dx = (x - center + N/2) % N - N/2
        dy = (y - center + N/2) % N - N/2
        dz = (z - center + N/2) % N - N/2
        
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Bin by radius
        max_r = N // 2 - 2
        n_bins = min(max_r, 30)
        r_edges = np.linspace(0, max_r, n_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        
        profile = np.zeros(n_bins)
        counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            mask = (r >= r_edges[i]) & (r < r_edges[i + 1])
            if np.sum(mask) > 0:
                profile[i] = np.mean(field[mask])
                counts[i] = np.sum(mask)
        
        # Filter out empty bins
        valid = counts > 0
        return r_centers[valid], profile[valid]
    
    def get_far_field_reference(self) -> Tuple[float, float]:
        """
        Get far-field reference values for P and Φ.
        
        Returns P_∞ and Φ_∞ from the outer boundary region.
        """
        N = self.p.N
        center = N // 2
        
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        dx = (x - center + N/2) % N - N/2
        dy = (y - center + N/2) % N - N/2
        dz = (z - center + N/2) % N - N/2
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Far field: outer 20% of grid
        far_threshold = 0.4 * N
        far_mask = r > far_threshold
        
        P_inf = np.mean(self.P[far_mask])
        Phi_inf = np.mean(self.Phi[far_mask])
        
        return P_inf, Phi_inf


def run_gravitational_redshift_test(
    verbose: bool = True,
    plot: bool = True,
    tolerance: float = 0.01  # 1% tolerance
) -> Dict:
    """
    Run the gravitational redshift falsifier test.
    
    DET predicts: P(r)/P_∞ = 1 + Φ(r) in the weak field limit
    
    Returns:
        Dictionary with test results and diagnostics
    """
    print("="*70)
    print("DET v6 GRAVITATIONAL REDSHIFT FALSIFIER TEST")
    print("="*70)
    print("\nTheory Card Prediction (weak field):")
    print("    P(r)/P_∞ = 1 + Φ(r)")
    print("    where Φ(r) is the gravitational potential")
    print("\nFalsifier criterion: |P(r)/P_∞ - (1 + Φ(r))| > 1%")
    print("-"*70)
    
    # Initialize simulator
    params = RedshiftTestParams(
        N=64,
        gravity_enabled=True,
        alpha_grav=0.02,
        kappa_grav=3.0,  # Moderate coupling for weak field
    )
    
    sim = RedshiftTestSimulator(params)
    
    # Add central mass with structure
    print("\n1. Setting up central mass with structure q...")
    sim.add_central_mass(
        q_amplitude=0.4,   # Moderate structure (weak field)
        width=4.0,
        F_amplitude=3.0
    )
    
    print(f"   Total q: {np.sum(sim.q):.3f}")
    print(f"   Max q: {np.max(sim.q):.3f}")
    print(f"   Max F: {np.max(sim.F):.3f}")
    
    # Compute gravity
    print("\n2. Computing gravitational field...")
    sim.compute_gravity()
    
    print(f"   Phi range: [{np.min(sim.Phi):.4f}, {np.max(sim.Phi):.4f}]")
    print(f"   Phi at center: {sim.Phi[params.N//2, params.N//2, params.N//2]:.4f}")
    
    # Compute presence
    print("\n3. Computing presence field...")
    sim.compute_presence()
    
    print(f"   P range: [{np.min(sim.P):.4f}, {np.max(sim.P):.4f}]")
    print(f"   P at center: {sim.P[params.N//2, params.N//2, params.N//2]:.4f}")
    
    # Get far-field references
    print("\n4. Extracting far-field reference values...")
    P_inf, Phi_inf = sim.get_far_field_reference()
    print(f"   P_∞ = {P_inf:.6f}")
    print(f"   Φ_∞ = {Phi_inf:.6f}")
    
    # Compute radial profiles
    print("\n5. Computing radial profiles...")
    r_P, P_profile = sim.get_radial_profile(sim.P)
    r_Phi, Phi_profile = sim.get_radial_profile(sim.Phi)
    r_q, q_profile = sim.get_radial_profile(sim.q)
    r_F, F_profile = sim.get_radial_profile(sim.F)
    
    # Normalize presence
    P_normalized = P_profile / P_inf
    
    # DET prediction: P/P_∞ = 1 + Φ
    # But we need to account for Φ_∞ ≠ 0 due to periodic BC
    Phi_shifted = Phi_profile - Phi_inf
    P_predicted = 1.0 + Phi_shifted
    
    # Compute deviation
    deviation = P_normalized - P_predicted
    max_deviation = np.max(np.abs(deviation))
    mean_deviation = np.mean(np.abs(deviation))
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nRadial analysis ({len(r_P)} points):")
    print(f"   r range: [{r_P[0]:.1f}, {r_P[-1]:.1f}]")
    
    print("\nPresence-Potential relationship:")
    print(f"   Max |P/P_∞ - (1+Φ)|: {max_deviation:.4f} ({max_deviation*100:.2f}%)")
    print(f"   Mean |P/P_∞ - (1+Φ)|: {mean_deviation:.4f} ({mean_deviation*100:.2f}%)")
    
    # Check falsifier
    passed = max_deviation < tolerance
    
    print("\n" + "-"*70)
    print(f"FALSIFIER CHECK (tolerance = {tolerance*100:.1f}%):")
    if passed:
        print(f"   ✓ PASSED: Max deviation {max_deviation*100:.2f}% < {tolerance*100:.1f}%")
    else:
        print(f"   ✗ FAILED: Max deviation {max_deviation*100:.2f}% > {tolerance*100:.1f}%")
    print("-"*70)
    
    # Detailed analysis
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)
    
    # Check the actual relationship
    # In current DET: P = a*σ/(1+F)/(1+H)
    # This doesn't directly include Φ
    print("\nNote on DET Presence Formula:")
    print("   Theory Card III.1: P = a·σ / (1+F^op) / (1+H)")
    print("   This formula does NOT directly include Φ.")
    print("\n   The redshift relation P/P_∞ = 1+Φ is a derived prediction,")
    print("   not a defining equation. This test checks if the dynamics")
    print("   produce this emergent relationship.")
    
    # Analyze what's actually happening
    print("\nActual contributions to P variation:")
    
    # Presence depends on F through 1/(1+F)
    F_contribution = 1.0 / (1.0 + F_profile) / (1.0 / (1.0 + np.mean(F_profile[-5:])))
    
    print(f"   F variation contribution: [{np.min(F_contribution):.4f}, {np.max(F_contribution):.4f}]")
    print(f"   Actual P variation: [{np.min(P_normalized):.4f}, {np.max(P_normalized):.4f}]")
    
    # Store results
    results = {
        'passed': passed,
        'max_deviation': max_deviation,
        'mean_deviation': mean_deviation,
        'tolerance': tolerance,
        'r': r_P,
        'P_profile': P_profile,
        'P_normalized': P_normalized,
        'Phi_profile': Phi_profile,
        'Phi_shifted': Phi_shifted,
        'P_predicted': P_predicted,
        'deviation': deviation,
        'q_profile': q_profile,
        'F_profile': F_profile,
        'P_inf': P_inf,
        'Phi_inf': Phi_inf,
        'params': params
    }
    
    if plot:
        create_diagnostic_plots(results, sim)
    
    return results


def run_enhanced_redshift_test(verbose: bool = True, plot: bool = True) -> Dict:
    """
    Enhanced test that explores the relationship between F, q, and the
    emergent clock-rate behavior.
    
    The key insight: In DET, P depends on F, and F accumulates where q > 0
    due to gravitational flux. The redshift should emerge dynamically.
    """
    print("\n" + "="*70)
    print("ENHANCED GRAVITATIONAL REDSHIFT TEST")
    print("="*70)
    print("\nThis test runs dynamics to see if resource F accumulates in")
    print("potential wells, causing presence P to decrease there.")
    print("-"*70)
    
    from det_v6_3d_collider_gravity import DETCollider3DGravity, DETParams3D
    
    params = DETParams3D(
        N=48,
        DT=0.015,
        F_VAC=0.01,
        F_MIN=0.0,
        C_init=0.2,
        diff_enabled=True,
        momentum_enabled=False,  # No momentum for cleaner test
        angular_momentum_enabled=False,
        floor_enabled=False,
        q_enabled=False,  # Fixed q
        agency_dynamic=False,  # Fixed agency
        gravity_enabled=True,
        alpha_grav=0.015,
        kappa_grav=4.0,
        mu_grav=2.0
    )
    
    sim = DETCollider3DGravity(params)
    
    # Add central mass with structure but NO extra F initially
    N = params.N
    center = N // 2
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    
    dx = (x - center + N/2) % N - N/2
    dy = (y - center + N/2) % N - N/2
    dz = (z - center + N/2) % N - N/2
    r2 = dx**2 + dy**2 + dz**2
    
    width = 3.5
    envelope = np.exp(-0.5 * r2 / width**2)
    
    # Set q directly (sources gravity)
    sim.q = 0.35 * envelope
    
    # Start with nearly uniform F
    sim.F = np.ones((N, N, N)) * params.F_VAC
    
    print(f"\nInitial state:")
    print(f"   Total q: {np.sum(sim.q):.3f}")
    print(f"   F range: [{np.min(sim.F):.4f}, {np.max(sim.F):.4f}]")
    print(f"   F mean: {np.mean(sim.F):.4f}")
    
    # Record initial gravity field
    sim._compute_gravity()
    Phi_initial = sim.Phi.copy()
    print(f"   Φ range: [{np.min(sim.Phi):.4f}, {np.max(sim.Phi):.4f}]")
    
    # Run dynamics
    n_steps = 400
    print(f"\nRunning {n_steps} steps of dynamics...")
    
    records = {'t': [], 'F_center': [], 'F_far': [], 'P_center': [], 'P_far': []}
    
    for t in range(n_steps):
        if t % 100 == 0:
            F_center = sim.F[center, center, center]
            F_far = np.mean(sim.F[r2 > (0.4*N)**2])
            P_center = sim.P[center, center, center]
            P_far = np.mean(sim.P[r2 > (0.4*N)**2])
            
            records['t'].append(t)
            records['F_center'].append(F_center)
            records['F_far'].append(F_far)
            records['P_center'].append(P_center)
            records['P_far'].append(P_far)
            
            print(f"   t={t:4d}: F_center={F_center:.4f}, F_far={F_far:.4f}, "
                  f"P_center={P_center:.4f}, P_far={P_far:.4f}")
        
        sim.step()
    
    # Final analysis
    print(f"\nFinal state:")
    print(f"   F range: [{np.min(sim.F):.4f}, {np.max(sim.F):.4f}]")
    print(f"   P range: [{np.min(sim.P):.4f}, {np.max(sim.P):.4f}]")
    
    # Get radial profiles
    def get_radial_profile(field, r_arr):
        max_r = N // 2 - 2
        n_bins = 25
        r_edges = np.linspace(0, max_r, n_bins + 1)
        r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
        
        profile = np.zeros(n_bins)
        for i in range(n_bins):
            mask = (r_arr >= r_edges[i]) & (r_arr < r_edges[i + 1])
            if np.sum(mask) > 0:
                profile[i] = np.mean(field[mask])
        
        return r_centers, profile
    
    r_flat = np.sqrt(r2)
    r_vals, F_profile = get_radial_profile(sim.F, r_flat)
    _, P_profile = get_radial_profile(sim.P, r_flat)
    _, Phi_profile = get_radial_profile(sim.Phi, r_flat)
    
    # Normalize
    far_mask = r_vals > 0.4 * N
    P_inf = np.mean(P_profile[far_mask]) if np.any(far_mask) else P_profile[-1]
    Phi_inf = np.mean(Phi_profile[far_mask]) if np.any(far_mask) else Phi_profile[-1]
    
    P_normalized = P_profile / P_inf
    Phi_shifted = Phi_profile - Phi_inf
    P_predicted = 1.0 + Phi_shifted
    
    deviation = np.abs(P_normalized - P_predicted)
    max_dev = np.max(deviation)
    
    print(f"\nRedshift Analysis:")
    print(f"   P_∞ = {P_inf:.6f}")
    print(f"   Φ_∞ = {Phi_inf:.6f}")
    print(f"   Max |P/P_∞ - (1+Φ)| = {max_dev:.4f} ({max_dev*100:.2f}%)")
    
    passed = max_dev < 0.01
    print(f"\n   FALSIFIER: {'PASSED' if passed else 'FAILED'}")
    
    results = {
        'passed': passed,
        'max_deviation': max_dev,
        'r': r_vals,
        'F_profile': F_profile,
        'P_profile': P_profile,
        'P_normalized': P_normalized,
        'Phi_profile': Phi_profile,
        'Phi_shifted': Phi_shifted,
        'P_predicted': P_predicted,
        'deviation': deviation,
        'P_inf': P_inf,
        'Phi_inf': Phi_inf,
        'records': records
    }
    
    if plot:
        create_enhanced_plots(results)
    
    return results


def create_diagnostic_plots(results: Dict, sim=None):
    """Create diagnostic plots for the redshift test."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DET v6 Gravitational Redshift Falsifier Test', fontsize=14, fontweight='bold')
    
    r = results['r']
    
    # Plot 1: Source profiles (q and F)
    ax1 = axes[0, 0]
    ax1.plot(r, results['q_profile'], 'b-', linewidth=2, label='q (structure)')
    ax1b = ax1.twinx()
    ax1b.plot(r, results['F_profile'], 'r--', linewidth=2, label='F (resource)')
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Structural debt q', color='b')
    ax1b.set_ylabel('Resource F', color='r')
    ax1.set_title('Source Profiles')
    ax1.legend(loc='upper right')
    ax1b.legend(loc='right')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gravitational potential
    ax2 = axes[0, 1]
    ax2.plot(r, results['Phi_profile'], 'g-', linewidth=2, label='Φ(r)')
    ax2.axhline(y=results['Phi_inf'], color='g', linestyle='--', alpha=0.5, label='Φ_∞')
    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Gravitational Potential Φ')
    ax2.set_title('Gravitational Potential')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Presence profile
    ax3 = axes[0, 2]
    ax3.plot(r, results['P_profile'], 'm-', linewidth=2, label='P(r)')
    ax3.axhline(y=results['P_inf'], color='m', linestyle='--', alpha=0.5, label='P_∞')
    ax3.set_xlabel('Radius r')
    ax3.set_ylabel('Presence P')
    ax3.set_title('Presence (Clock Rate)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Normalized presence vs prediction
    ax4 = axes[1, 0]
    ax4.plot(r, results['P_normalized'], 'b-', linewidth=2, label='P/P_∞ (measured)')
    ax4.plot(r, results['P_predicted'], 'r--', linewidth=2, label='1 + Φ (predicted)')
    ax4.set_xlabel('Radius r')
    ax4.set_ylabel('Normalized Presence')
    ax4.set_title('Presence vs DET Prediction')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Deviation
    ax5 = axes[1, 1]
    ax5.plot(r, results['deviation'] * 100, 'k-', linewidth=2)
    ax5.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='1% threshold')
    ax5.axhline(y=-1.0, color='r', linestyle='--', alpha=0.7)
    ax5.fill_between(r, -1, 1, alpha=0.2, color='green', label='Pass region')
    ax5.set_xlabel('Radius r')
    ax5.set_ylabel('Deviation (%)')
    ax5.set_title(f'|P/P_∞ - (1+Φ)| (max={results["max_deviation"]*100:.2f}%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Scatter comparison
    ax6 = axes[1, 2]
    ax6.scatter(results['P_predicted'], results['P_normalized'], 
                c=r, cmap='viridis', s=50, alpha=0.7)
    
    # Perfect agreement line
    min_val = min(np.min(results['P_predicted']), np.min(results['P_normalized']))
    max_val = max(np.max(results['P_predicted']), np.max(results['P_normalized']))
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement')
    
    cbar = plt.colorbar(ax6.collections[0], ax=ax6)
    cbar.set_label('Radius r')
    ax6.set_xlabel('1 + Φ (predicted)')
    ax6.set_ylabel('P/P_∞ (measured)')
    ax6.set_title('Measured vs Predicted Presence')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/redshift_falsifier_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: /mnt/user-data/outputs/redshift_falsifier_test.png")
    plt.close()


def create_enhanced_plots(results: Dict):
    """Create plots for the enhanced dynamical test."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DET v6 Enhanced Gravitational Redshift Test (with dynamics)', 
                 fontsize=14, fontweight='bold')
    
    r = results['r']
    
    # Plot 1: Final profiles
    ax1 = axes[0, 0]
    ax1.plot(r, results['F_profile'], 'b-', linewidth=2, label='F(r)')
    ax1.set_xlabel('Radius r')
    ax1.set_ylabel('Resource F')
    ax1.set_title('Final Resource Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Presence vs Potential
    ax2 = axes[0, 1]
    ax2.plot(r, results['P_normalized'], 'b-', linewidth=2, label='P/P_∞ (measured)')
    ax2.plot(r, results['P_predicted'], 'r--', linewidth=2, label='1+Φ (predicted)')
    ax2.set_xlabel('Radius r')
    ax2.set_ylabel('Normalized Presence')
    ax2.set_title('Clock Rate vs Potential')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Time evolution
    ax3 = axes[1, 0]
    rec = results['records']
    ax3.plot(rec['t'], rec['F_center'], 'b-', linewidth=2, label='F at center')
    ax3.plot(rec['t'], rec['F_far'], 'b--', linewidth=2, label='F at far field')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Resource F')
    ax3.set_title('Resource Accumulation Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Deviation
    ax4 = axes[1, 1]
    ax4.plot(r, results['deviation'] * 100, 'k-', linewidth=2)
    ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='1% threshold')
    ax4.axhline(y=-1.0, color='r', linestyle='--', alpha=0.7)
    ax4.fill_between(r, -1, 1, alpha=0.2, color='green')
    ax4.set_xlabel('Radius r')
    ax4.set_ylabel('Deviation (%)')
    ax4.set_title(f'Deviation from Prediction (max={results["max_deviation"]*100:.2f}%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/redshift_enhanced_test.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: /mnt/user-data/outputs/redshift_enhanced_test.png")
    plt.close()


def run_comprehensive_redshift_analysis():
    """
    Run comprehensive analysis exploring when/if redshift emerges.
    
    Key insight: The DET presence formula doesn't directly include Φ.
    Redshift must emerge from dynamics: gravitational flux accumulates F
    in potential wells, which then reduces P via 1/(1+F).
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE GRAVITATIONAL REDSHIFT ANALYSIS")
    print("="*70)
    
    print("\n" + "-"*70)
    print("THEORETICAL CONTEXT")
    print("-"*70)
    print("""
The DET Theory Card defines:
    
    Presence (III.1):  P = a·σ / (1+F) / (1+H)
    Mass (III.2):      M = P⁻¹
    Potential (V.2):   LΦ = -κρ where ρ = q - b

The gravitational redshift prediction P/P_∞ = 1+Φ is NOT built into
the presence formula. It must EMERGE from the dynamics.

Possible emergence mechanisms:
1. Gravitational flux J^(grav) accumulates F in potential wells
2. Higher F → lower P (via 1/(1+F) factor)
3. This should produce P/P_∞ ≈ 1+Φ if the coupling is correct

This test examines whether the current parameter regime produces
the correct emergent relationship.
""")
    
    # Test 1: Static analysis (no dynamics)
    print("\n" + "="*70)
    print("TEST 1: STATIC ANALYSIS (Pure presence-potential relationship)")
    print("="*70)
    results_static = run_gravitational_redshift_test(verbose=True, plot=True, tolerance=0.01)
    
    # Test 2: With dynamics
    print("\n" + "="*70)
    print("TEST 2: DYNAMIC ANALYSIS (With gravitational flux)")
    print("="*70)
    results_dynamic = run_enhanced_redshift_test(verbose=True, plot=True)
    
    # Summary
    print("\n" + "="*70)
    print("COMPREHENSIVE SUMMARY")
    print("="*70)
    
    print("\nTest Results:")
    print(f"   Static test:  {'PASSED' if results_static['passed'] else 'FAILED'} "
          f"(max dev: {results_static['max_deviation']*100:.2f}%)")
    print(f"   Dynamic test: {'PASSED' if results_dynamic['passed'] else 'FAILED'} "
          f"(max dev: {results_dynamic['max_deviation']*100:.2f}%)")
    
    print("\nPhysical Interpretation:")
    if not results_static['passed'] and not results_dynamic['passed']:
        print("""
   The gravitational redshift relation P/P_∞ = 1+Φ does NOT emerge 
   automatically from the current DET formulation.
   
   This is expected because:
   1. Presence P depends on F through 1/(1+F), not directly on Φ
   2. The redshift relation requires a specific calibration between
      the gravitational flux strength and the F-dependence of P
   
   To achieve P/P_∞ = 1+Φ would require either:
   a) Adding Φ directly to the presence formula, or
   b) Tuning parameters so gravitational F-accumulation produces
      the correct clock-rate relationship
   
   This is a key finding for DET development: the redshift relation
   is a derived/calibrated result, not an automatic consequence.
""")
    
    return {'static': results_static, 'dynamic': results_dynamic}


if __name__ == "__main__":
    # Copy collider to working directory
    import shutil
    import os
    
    src = '/mnt/user-data/uploads/det_v6_3d_collider_gravity.py'
    dst = '/home/claude/det_v6_3d_collider_gravity.py'
    if os.path.exists(src):
        shutil.copy(src, dst)
    
    # Run comprehensive analysis
    results = run_comprehensive_redshift_analysis()
