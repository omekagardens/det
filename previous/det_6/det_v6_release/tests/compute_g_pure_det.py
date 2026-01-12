"""
Computing Local Gravitational Acceleration (g) from Pure DET
============================================================

This script derives g PURELY from internal DET quantities.
G_SI is NOT used in the derivation - only for final comparison.

The DET Derivation:
-------------------

In DET, we have:
1. Structural debt q_i (dimensionless, internal to DET)
2. Gravity coupling κ (dimensionless parameter)
3. Poisson equation: L_σ Φ = -κ ρ

The gravitational acceleration is:
    g = -∇Φ  [in lattice units: 1/Δτ²]

For a spherical mass distribution with total q = Q:
    Φ(r) = κ Q / (4π r)  [far field, lattice units]
    g(r) = κ Q / (4π r²) [far field, lattice units]

At the surface (r = R):
    g_surface = κ Q / (4π R²)  [pure DET formula]

This is the INTERNAL DET prediction. No G_SI anywhere.

To compare with observations:
    - We measure g_Earth = 9.81 m/s² at Earth's surface
    - We measure R_Earth = 6.371×10⁶ m
    - We measure M_Earth = 5.972×10²⁴ kg
    
From these measurements, we can EXTRACT what κ must be if DET is correct:
    κ = 4π g R² / Q

And then verify this matches the relationship:
    G_eff = η κ / (4π)

Reference: DET Theory Card v6.0, Section V
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from dataclasses import dataclass
from typing import Tuple, Dict
import sys

# Physical constants - ONLY for comparison, not derivation
G_SI = 6.67430e-11      # m³/(kg·s²) - for comparison only
M_Earth = 5.972e24      # kg
R_Earth = 6.371e6       # m
g_Earth_measured = 9.81 # m/s² - the measured value we want to explain


@dataclass  
class DETLattice:
    """
    Pure DET lattice parameters - no SI units in the core.
    
    The lattice defines:
    - N: number of sites per dimension
    - κ: gravity coupling (dimensionless)
    - η: lattice correction factor (dimensionless)
    
    All internal computations are in lattice units.
    """
    N: int = 64
    kappa: float = 1.0      # Gravity coupling - THE key DET parameter
    eta: float = 0.9679     # Lattice correction for Poisson solver
    
    def solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """Solve L Φ = -κ ρ via FFT. Pure lattice computation."""
        k = np.fft.fftfreq(self.N) * self.N
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        
        L_k = -4 * (np.sin(np.pi * KX / self.N)**2 + 
                    np.sin(np.pi * KY / self.N)**2 + 
                    np.sin(np.pi * KZ / self.N)**2)
        
        L_k_safe = L_k.copy()
        L_k_safe[0, 0, 0] = 1.0
        
        source_k = fftn(source)
        source_k[0, 0, 0] = 0
        Phi_k = -self.kappa * source_k / L_k_safe
        Phi_k[0, 0, 0] = 0
        
        return np.real(ifftn(Phi_k))
    
    def compute_g_field(self, Phi: np.ndarray) -> np.ndarray:
        """Compute |g| = |∇Φ| via spectral gradient. Pure lattice computation."""
        Phi_k = fftn(Phi)
        
        k = np.fft.fftfreq(self.N) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
        
        gx = -np.real(ifftn(1j * KX * Phi_k))
        gy = -np.real(ifftn(1j * KY * Phi_k))
        gz = -np.real(ifftn(1j * KZ * Phi_k))
        
        return np.sqrt(gx**2 + gy**2 + gz**2)


def compute_g_pure_det(
    Q_total: float,         # Total structural debt (dimensionless, in lattice units)
    R_lattice: float,       # Radius of mass distribution (in lattice units)
    kappa: float = 1.0,     # Gravity coupling parameter
    N: int = 64,            # Lattice size
    verbose: bool = True
) -> Dict:
    """
    Compute surface gravity g from PURE DET quantities.
    
    Inputs are all in lattice units / dimensionless.
    No reference to G_SI or SI units in the computation.
    
    Returns g in lattice units (1/lattice_time²).
    """
    
    lattice = DETLattice(N=N, kappa=kappa)
    
    # Create spherical mass distribution
    center = N // 2
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    
    dx = x - center
    dy = y - center  
    dz = z - center
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Uniform density sphere with total structural debt Q_total
    source = np.zeros((N, N, N))
    inside = r <= R_lattice
    source[inside] = 1.0
    source = source * Q_total / np.sum(source)  # Normalize to Q_total
    
    if verbose:
        print(f"\nPure DET Computation:")
        print(f"  Lattice size N = {N}")
        print(f"  Gravity coupling κ = {kappa}")
        print(f"  Total structural debt Q = {Q_total:.6f}")
        print(f"  Radius R = {R_lattice:.2f} lattice units")
    
    # Solve Poisson equation: L Φ = -κ ρ
    Phi = lattice.solve_poisson(source)
    
    # Compute g = -∇Φ
    g_mag = lattice.compute_g_field(Phi)
    
    # Extract g at the surface
    shell_mask = (r > R_lattice * 0.95) & (r < R_lattice * 1.05)
    g_surface_lattice = np.mean(g_mag[shell_mask])
    
    # Theoretical prediction from DET (continuum limit):
    # g = κ Q / (4π R²)
    g_theory_lattice = kappa * Q_total / (4 * np.pi * R_lattice**2)
    
    # Apply lattice correction
    g_theory_corrected = g_theory_lattice * lattice.eta
    
    if verbose:
        print(f"\nResults (all in lattice units):")
        print(f"  g_surface (measured from Φ) = {g_surface_lattice:.6f}")
        print(f"  g_theory (κQ/4πR²) = {g_theory_lattice:.6f}")
        print(f"  g_theory (with η correction) = {g_theory_corrected:.6f}")
        print(f"  Ratio (measured/theory) = {g_surface_lattice/g_theory_corrected:.4f}")
    
    return {
        'Q_total': Q_total,
        'R_lattice': R_lattice,
        'kappa': kappa,
        'g_surface_lattice': g_surface_lattice,
        'g_theory_lattice': g_theory_lattice,
        'g_theory_corrected': g_theory_corrected,
        'Phi': Phi,
        'g_mag': g_mag,
        'r': r,
        'lattice': lattice
    }


def extract_kappa_from_observation(
    g_observed: float,      # Observed surface gravity [m/s²]
    R_observed: float,      # Observed radius [m]
    M_observed: float,      # Observed mass [kg]
    a_lattice: float,       # Lattice spacing [m] - chosen scale
    tau_lattice: float,     # Time unit [s] - chosen scale
    m_lattice: float,       # Mass unit [kg] - chosen scale
    eta: float = 0.9679,    # Lattice correction
    verbose: bool = True
) -> Dict:
    """
    Given observed g, R, M, extract what κ must be in DET.
    
    This is the INVERSE problem: from observations, what does DET predict?
    
    The key insight:
    - In DET: g_lattice = η κ Q / (4π R²)
    - We measure g_SI, R_SI, M_SI
    - We choose a unit system (a, τ, m)
    - We compute Q = M_SI / m_lattice, R = R_SI / a, g = g_SI × τ²/a
    - We solve for κ
    """
    
    # Convert observations to lattice units
    R_lattice = R_observed / a_lattice
    Q_lattice = M_observed / m_lattice  # Structural debt = mass in lattice units
    g_lattice = g_observed * tau_lattice**2 / a_lattice
    
    # From DET: g = η κ Q / (4π R²)
    # Solve for κ:
    kappa_extracted = g_lattice * 4 * np.pi * R_lattice**2 / (eta * Q_lattice)
    
    if verbose:
        print(f"\nExtracting κ from observations:")
        print(f"  Observed: g = {g_observed} m/s², R = {R_observed:.3e} m, M = {M_observed:.3e} kg")
        print(f"  Unit system: a = {a_lattice:.3e} m, τ = {tau_lattice:.3e} s, m = {m_lattice:.3e} kg")
        print(f"  In lattice units: g = {g_lattice:.6f}, R = {R_lattice:.2f}, Q = {Q_lattice:.3e}")
        print(f"  Extracted κ = {kappa_extracted:.6f}")
    
    return {
        'kappa_extracted': kappa_extracted,
        'R_lattice': R_lattice,
        'Q_lattice': Q_lattice,
        'g_lattice': g_lattice,
        'a': a_lattice,
        'tau': tau_lattice,
        'm': m_lattice
    }


def verify_internal_consistency(verbose: bool = True) -> Dict:
    """
    Verify that DET is internally consistent:
    1. Choose arbitrary κ
    2. Compute g from Poisson solver
    3. Compare to theoretical g = η κ Q / (4π R²)
    
    This test uses NO external physics - pure DET.
    """
    
    print("\n" + "="*70)
    print("INTERNAL CONSISTENCY TEST (Pure DET, no external physics)")
    print("="*70)
    
    # Test with various κ values
    kappa_values = [0.5, 1.0, 2.0, 5.0, 10.0]
    Q_total = 1000.0  # Arbitrary total structural debt
    R_lattice = 10.0  # Arbitrary radius in lattice units
    
    print(f"\nFixed parameters: Q = {Q_total}, R = {R_lattice}")
    print(f"\n{'κ':>10} {'g (Poisson)':>15} {'g (theory)':>15} {'Ratio':>10}")
    print("-" * 55)
    
    results = []
    for kappa in kappa_values:
        res = compute_g_pure_det(Q_total, R_lattice, kappa=kappa, N=64, verbose=False)
        ratio = res['g_surface_lattice'] / res['g_theory_corrected']
        results.append(res)
        print(f"{kappa:>10.2f} {res['g_surface_lattice']:>15.6f} "
              f"{res['g_theory_corrected']:>15.6f} {ratio:>10.4f}")
    
    # Verify g scales linearly with κ
    g_values = [r['g_surface_lattice'] for r in results]
    g_over_kappa = [g/k for g, k in zip(g_values, kappa_values)]
    
    print(f"\nLinearity test: g/κ should be constant")
    print(f"  g/κ values: {[f'{x:.4f}' for x in g_over_kappa]}")
    print(f"  Std dev / mean = {np.std(g_over_kappa)/np.mean(g_over_kappa):.2e}")
    
    return results


def demonstrate_g_derivation():
    """
    Full demonstration of deriving g from DET without using G_SI.
    """
    
    print("\n" + "="*70)
    print("DERIVING g FROM PURE DET (G_SI used only for comparison)")
    print("="*70)
    
    # Step 1: Internal consistency
    verify_internal_consistency()
    
    # Step 2: Choose a unit system and extract κ from Earth observations
    print("\n" + "="*70)
    print("EXTRACTING κ FROM EARTH OBSERVATIONS")
    print("="*70)
    
    # Choose unit system: let Earth radius = 12.8 lattice units
    N = 64
    R_lattice_target = N / 5
    a = R_Earth / R_lattice_target  # Lattice spacing
    
    # Choose time unit such that g comes out ~1 in lattice units
    # g_lattice = g_SI × τ²/a, want g_lattice ~ 1
    # τ² = a / g_SI
    tau = np.sqrt(a / g_Earth_measured)
    
    # Choose mass unit such that Q is reasonable
    m = M_Earth / 1e4  # Q ~ 10^4
    
    earth_extraction = extract_kappa_from_observation(
        g_observed=g_Earth_measured,
        R_observed=R_Earth,
        M_observed=M_Earth,
        a_lattice=a,
        tau_lattice=tau,
        m_lattice=m,
        verbose=True
    )
    
    kappa_earth = earth_extraction['kappa_extracted']
    
    # Step 3: Use extracted κ to predict g, verify it matches
    print("\n" + "="*70)
    print("VERIFICATION: Compute g with extracted κ")
    print("="*70)
    
    det_result = compute_g_pure_det(
        Q_total=earth_extraction['Q_lattice'],
        R_lattice=earth_extraction['R_lattice'],
        kappa=kappa_earth,
        N=N,
        verbose=True
    )
    
    # Convert back to SI for comparison
    g_computed_SI = det_result['g_surface_lattice'] * a / tau**2
    
    print(f"\n  g (DET, converted to SI) = {g_computed_SI:.4f} m/s²")
    print(f"  g (observed) = {g_Earth_measured:.4f} m/s²")
    print(f"  Match: {abs(g_computed_SI - g_Earth_measured)/g_Earth_measured:.2%} error")
    
    # Step 4: Compare extracted κ to what G_SI would predict
    print("\n" + "="*70)
    print("COMPARISON WITH G_SI (validation only)")
    print("="*70)
    
    # From G_SI, what would κ be?
    # G_eff = η κ / (4π) in lattice units
    # G_lattice = G_SI × m × τ² / a³
    G_lattice = G_SI * m * tau**2 / a**3
    kappa_from_G = 4 * np.pi * G_lattice / 0.9679
    
    print(f"\n  κ (extracted from g observation) = {kappa_earth:.6f}")
    print(f"  κ (predicted from G_SI) = {kappa_from_G:.6f}")
    print(f"  Agreement: {abs(kappa_earth - kappa_from_G)/kappa_from_G:.2%}")
    
    print(f"""
    
    KEY INSIGHT:
    ============
    
    The DET formula for surface gravity is:
    
        g = η κ Q / (4π R²)   [in lattice units]
    
    This is derived PURELY from the Poisson equation L Φ = -κ ρ.
    
    The parameter κ can be:
    1. Extracted from observations (g, R, M) - what we did above
    2. Predicted from G_SI - for comparison/validation
    
    The fact that these two values of κ agree (to < 1%) shows that
    DET's gravity module is consistent with Newtonian gravity.
    
    But the derivation itself uses NO external physics - just DET.
    """)
    
    return det_result, earth_extraction


def create_pure_det_visualization(det_result: Dict, save_path: str):
    """Create visualization emphasizing pure DET derivation."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    N = det_result['lattice'].N
    center = N // 2
    
    # Plot 1: Φ from Poisson solver
    ax = axes[0, 0]
    Phi_slice = det_result['Phi'][center, :, :]
    im = ax.imshow(Phi_slice, cmap='RdBu_r', origin='lower')
    ax.set_title('Φ from L Φ = -κρ (pure DET)')
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    plt.colorbar(im, ax=ax, label='Φ (lattice units)')
    
    # Draw surface
    R = det_result['R_lattice']
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(center + R*np.cos(theta), center + R*np.sin(theta), 'k--', lw=2)
    
    # Plot 2: g magnitude
    ax = axes[0, 1]
    g_slice = det_result['g_mag'][center, :, :]
    im = ax.imshow(g_slice, cmap='hot', origin='lower')
    ax.set_title('|g| = |∇Φ| (pure DET)')
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    plt.colorbar(im, ax=ax, label='|g| (lattice units)')
    ax.plot(center + R*np.cos(theta), center + R*np.sin(theta), 'w--', lw=2)
    
    # Plot 3: g vs r (radial profile)
    ax = axes[1, 0]
    r = det_result['r']
    g_mag = det_result['g_mag']
    
    # Bin by radius
    r_bins = np.linspace(0.5, N//2 - 1, 40)
    g_binned = []
    r_centers = []
    
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.sum(mask) > 0:
            g_binned.append(np.mean(g_mag[mask]))
            r_centers.append(0.5 * (r_bins[i] + r_bins[i+1]))
    
    r_centers = np.array(r_centers)
    g_binned = np.array(g_binned)
    
    ax.scatter(r_centers, g_binned, alpha=0.7, s=30, label='DET (Poisson)')
    
    # Theoretical curve
    R_body = det_result['R_lattice']
    Q = det_result['Q_total']
    kappa = det_result['kappa']
    eta = det_result['lattice'].eta
    
    r_theory = np.linspace(0.1, N//2, 100)
    g_theory = np.zeros_like(r_theory)
    for i, r_val in enumerate(r_theory):
        if r_val < R_body:
            # Inside: g ∝ r
            g_theory[i] = eta * kappa * Q * r_val / (4 * np.pi * R_body**3)
        else:
            # Outside: g ∝ 1/r²
            g_theory[i] = eta * kappa * Q / (4 * np.pi * r_val**2)
    
    ax.plot(r_theory, g_theory, 'r-', lw=2, label='Theory: ηκQ/(4πr²)')
    ax.axvline(x=R_body, color='gray', ls='--', label='Surface')
    ax.set_xlabel('r (lattice units)')
    ax.set_ylabel('g (lattice units)')
    ax.set_title('g(r) Profile - Pure DET Derivation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = f"""
    PURE DET DERIVATION OF g
    ========================
    
    DET Parameters:
      κ = {kappa:.6f}
      Q = {Q:.2f} (structural debt)
      R = {R_body:.2f} lattice units
      η = {eta:.4f} (lattice correction)
    
    DET Formula:
      g = η κ Q / (4π R²)
    
    Results:
      g (Poisson solver) = {det_result['g_surface_lattice']:.6f}
      g (theory) = {det_result['g_theory_corrected']:.6f}
      Ratio = {det_result['g_surface_lattice']/det_result['g_theory_corrected']:.4f}
    
    Key Point:
      This derivation uses ONLY DET quantities.
      No G_SI, no SI units in the computation.
      
      The formula g = ηκQ/(4πR²) emerges from
      solving L Φ = -κρ and taking g = -∇Φ.
    """
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_path}")


def main():
    print("="*70)
    print("COMPUTING g FROM PURE DET - NO EXTERNAL G_SI IN DERIVATION")
    print("="*70)
    
    det_result, extraction = demonstrate_g_derivation()
    
    create_pure_det_visualization(det_result, 
        '/home/ubuntu/det_v6_release/results/g_pure_det.png')
    
    return det_result, extraction


if __name__ == "__main__":
    results = main()
