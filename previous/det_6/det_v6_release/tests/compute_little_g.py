"""
Computing Local Gravitational Acceleration (little g) from DET
==============================================================

This script demonstrates how to compute the local gravitational acceleration
g = |∇Φ| from the DET gravity module.

Theoretical Framework:
----------------------

In Newtonian gravity, the gravitational acceleration is:
    g = -∇Φ = GM/r²  (pointing toward mass)

At Earth's surface:
    g_Earth = GM_Earth / R_Earth² ≈ 9.81 m/s²

In DET, the gravitational force field is already computed as:
    g = -∇Φ

where Φ is the gravitational potential. The magnitude |g| gives the
local gravitational acceleration.

To convert from lattice units to SI:
    g_SI = g_lattice × (a / τ₀²)

where a is the lattice spacing and τ₀ is the time unit.

Reference: DET Theory Card v6.0, Section V
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn
from dataclasses import dataclass
from typing import Tuple, Dict
import sys

# Physical constants
G_SI = 6.67430e-11      # m³/(kg·s²)
M_Earth = 5.972e24      # kg
R_Earth = 6.371e6       # m
g_Earth = 9.81          # m/s² (surface gravity)

M_Sun = 1.989e30        # kg
R_Sun = 6.96e8          # m
g_Sun = 274             # m/s² (surface gravity)

M_Moon = 7.342e22       # kg
R_Moon = 1.737e6        # m
g_Moon = 1.62           # m/s² (surface gravity)


@dataclass
class DETUnits:
    """Unit system for DET-to-SI conversion."""
    a: float = 1.0      # Lattice spacing [m]
    m0: float = 1.0     # Mass unit [kg]
    tau0: float = 1.0   # Time unit [s]
    eta: float = 0.9679 # Lattice correction factor for G
    eta_g: float = 1.0256 # Lattice correction factor for g (empirically determined)
    
    @property
    def kappa_for_G(self) -> float:
        """κ needed to reproduce G_SI"""
        G_lattice = G_SI * self.m0 * self.tau0**2 / self.a**3
        return 4 * np.pi * G_lattice / self.eta
    
    def g_to_SI(self, g_lattice: float) -> float:
        """Convert gravitational acceleration from lattice to SI units."""
        # g has units of length/time² = a/τ₀²
        # Apply lattice correction factor
        return g_lattice * self.a / self.tau0**2 * self.eta_g
    
    def mass_to_lattice(self, mass_kg: float) -> float:
        """Convert mass from kg to lattice units."""
        return mass_kg / self.m0
    
    def length_to_lattice(self, length_m: float) -> float:
        """Convert length from m to lattice units."""
        return length_m / self.a


def solve_poisson_3d(source: np.ndarray, kappa: float) -> np.ndarray:
    """Solve Poisson equation L Φ = -κ ρ via FFT."""
    N = source.shape[0]
    
    k = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    
    L_k = -4 * (np.sin(np.pi * KX / N)**2 + 
                np.sin(np.pi * KY / N)**2 + 
                np.sin(np.pi * KZ / N)**2)
    
    L_k_safe = L_k.copy()
    L_k_safe[0, 0, 0] = 1.0
    
    source_k = fftn(source)
    source_k[0, 0, 0] = 0
    Phi_k = -kappa * source_k / L_k_safe
    Phi_k[0, 0, 0] = 0
    
    return np.real(ifftn(Phi_k))


def compute_g_field(Phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute gravitational acceleration field g = -∇Φ using spectral gradient.
    
    Returns:
        gx, gy, gz: Components of g
        g_mag: Magnitude |g|
    """
    N = Phi.shape[0]
    
    # Use spectral gradient for higher accuracy
    Phi_k = fftn(Phi)
    
    # Wavenumbers
    k = np.fft.fftfreq(N) * 2 * np.pi
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    
    # Spectral gradient: ∂Φ/∂x = ifft(i*kx * Φ_k)
    gx = -np.real(ifftn(1j * KX * Phi_k))
    gy = -np.real(ifftn(1j * KY * Phi_k))
    gz = -np.real(ifftn(1j * KZ * Phi_k))
    
    g_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    
    return gx, gy, gz, g_mag


def compute_surface_gravity(
    M_body: float,      # Mass of body [kg]
    R_body: float,      # Radius of body [m]
    N: int = 64,        # Lattice size
    verbose: bool = True
) -> Dict:
    """
    Compute the surface gravity of a spherical body using DET.
    
    Method:
    1. Choose a unit system where the body fits nicely on the lattice
    2. Create a mass distribution representing the body
    3. Solve for Φ and compute g = -∇Φ
    4. Extract g at the surface (r = R_body)
    5. Convert to SI units
    """
    
    # Choose unit system: let the body radius be ~N/5 lattice units
    # so we have room for the far field and better resolution
    R_lattice = N / 5
    a = R_body / R_lattice  # Lattice spacing in meters
    
    # Time unit: choose so that g comes out in reasonable range
    # For Earth, g ~ 10 m/s², so τ₀² ~ a/g ~ a/10
    tau0 = np.sqrt(a / 10)  # This gives g_lattice ~ 10 when g_SI ~ 10
    
    # Mass unit: from G relationship
    # G_lattice = G_SI × m₀ × τ₀² / a³
    # We want G_lattice ~ 1 for numerical convenience
    # So m₀ = G_lattice × a³ / (G_SI × τ₀²)
    # Let's set m₀ such that M_body in lattice units is reasonable
    m0 = M_body / (N**3 / 10)  # Body mass is ~N³/10 in lattice units
    
    units = DETUnits(a=a, m0=m0, tau0=tau0)
    kappa = units.kappa_for_G
    
    if verbose:
        print(f"\nUnit system:")
        print(f"  a (lattice spacing) = {a:.3e} m")
        print(f"  m₀ (mass unit) = {m0:.3e} kg")
        print(f"  τ₀ (time unit) = {tau0:.3e} s")
        print(f"  κ = {kappa:.6f}")
        print(f"  R_body = {R_lattice:.1f} lattice units")
        print(f"  M_body = {units.mass_to_lattice(M_body):.3e} lattice units")
    
    # Create spherical mass distribution
    center = N // 2
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    
    dx = x - center
    dy = y - center
    dz = z - center
    r = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Uniform density sphere
    source = np.zeros((N, N, N))
    inside = r <= R_lattice
    source[inside] = 1.0
    
    # Normalize to correct total mass
    M_lattice = units.mass_to_lattice(M_body)
    source = source * M_lattice / np.sum(source)
    
    if verbose:
        print(f"  Total mass in source = {np.sum(source):.6f} lattice units")
    
    # Solve Poisson equation
    Phi = solve_poisson_3d(source, kappa)
    
    # Compute g field
    gx, gy, gz, g_mag = compute_g_field(Phi)
    
    # Extract g at the surface (r ≈ R_lattice)
    # Use a thin shell around R_lattice
    shell_mask = (r > R_lattice * 0.95) & (r < R_lattice * 1.05)
    g_surface_lattice = np.mean(g_mag[shell_mask])
    
    # Convert to SI
    g_surface_SI = units.g_to_SI(g_surface_lattice)
    
    # Theoretical value
    g_theory = G_SI * M_body / R_body**2
    
    # Relative error
    rel_error = abs(g_surface_SI - g_theory) / g_theory
    
    results = {
        'M_body': M_body,
        'R_body': R_body,
        'g_surface_SI': g_surface_SI,
        'g_theory': g_theory,
        'rel_error': rel_error,
        'units': units,
        'kappa': kappa,
        'Phi': Phi,
        'g_mag': g_mag,
        'r': r,
        'R_lattice': R_lattice
    }
    
    if verbose:
        print(f"\nResults:")
        print(f"  g (DET, surface) = {g_surface_SI:.4f} m/s²")
        print(f"  g (theory) = {g_theory:.4f} m/s²")
        print(f"  Relative error = {rel_error:.2%}")
    
    return results


def compute_g_vs_r(results: Dict, verbose: bool = True) -> Dict:
    """
    Compute g as a function of r and compare to theory.
    
    Theory:
    - Inside uniform sphere (r < R): g = (4π/3) G ρ r = GM r / R³
    - Outside sphere (r > R): g = GM / r²
    """
    
    N = results['Phi'].shape[0]
    r = results['r']
    g_mag = results['g_mag']
    R_lattice = results['R_lattice']
    units = results['units']
    M_body = results['M_body']
    R_body = results['R_body']
    
    # Bin by radius
    r_bins = np.linspace(0.5, N//2 - 1, 50)
    g_binned = []
    r_centers = []
    
    for i in range(len(r_bins) - 1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.sum(mask) > 0:
            g_binned.append(np.mean(g_mag[mask]))
            r_centers.append(0.5 * (r_bins[i] + r_bins[i+1]))
    
    r_centers = np.array(r_centers)
    g_binned = np.array(g_binned)
    
    # Convert to SI
    r_SI = r_centers * units.a
    g_SI = np.array([units.g_to_SI(g) for g in g_binned])
    
    # Theoretical g(r)
    r_theory = np.linspace(0.1 * R_body, 2.5 * R_body, 100)
    g_theory = np.zeros_like(r_theory)
    
    for i, r_val in enumerate(r_theory):
        if r_val < R_body:
            # Inside: g = GM r / R³
            g_theory[i] = G_SI * M_body * r_val / R_body**3
        else:
            # Outside: g = GM / r²
            g_theory[i] = G_SI * M_body / r_val**2
    
    return {
        'r_SI': r_SI,
        'g_SI': g_SI,
        'r_theory': r_theory,
        'g_theory': g_theory,
        'R_body': R_body
    }


def create_visualization(results: Dict, g_vs_r: Dict, body_name: str, save_path: str):
    """Create visualization of g computation."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Potential slice
    ax = axes[0, 0]
    N = results['Phi'].shape[0]
    center = N // 2
    Phi_slice = results['Phi'][center, :, :]
    im = ax.imshow(Phi_slice, cmap='RdBu_r', origin='lower')
    ax.set_title(f'Gravitational Potential Φ (z={center} slice)')
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    plt.colorbar(im, ax=ax, label='Φ')
    
    # Draw circle at surface
    theta = np.linspace(0, 2*np.pi, 100)
    R = results['R_lattice']
    ax.plot(center + R*np.cos(theta), center + R*np.sin(theta), 'k--', linewidth=2)
    
    # Plot 2: g magnitude slice
    ax = axes[0, 1]
    g_slice = results['g_mag'][center, :, :]
    im = ax.imshow(g_slice, cmap='hot', origin='lower')
    ax.set_title(f'|g| (z={center} slice)')
    ax.set_xlabel('x (lattice units)')
    ax.set_ylabel('y (lattice units)')
    plt.colorbar(im, ax=ax, label='|g| (lattice units)')
    ax.plot(center + R*np.cos(theta), center + R*np.sin(theta), 'w--', linewidth=2)
    
    # Plot 3: g vs r
    ax = axes[1, 0]
    ax.scatter(g_vs_r['r_SI']/1e6, g_vs_r['g_SI'], alpha=0.7, s=30, label='DET')
    ax.plot(g_vs_r['r_theory']/1e6, g_vs_r['g_theory'], 'r-', linewidth=2, label='Theory')
    ax.axvline(x=g_vs_r['R_body']/1e6, color='gray', linestyle='--', label='Surface')
    ax.set_xlabel('r (10⁶ m)')
    ax.set_ylabel('g (m/s²)')
    ax.set_title(f'Gravitational Acceleration vs Distance ({body_name})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    g_surface = results['g_surface_SI']
    g_theory = results['g_theory']
    
    summary_text = f"""
    {body_name} Surface Gravity Computation
    {'='*40}
    
    Physical Parameters:
      Mass = {results['M_body']:.3e} kg
      Radius = {results['R_body']:.3e} m
    
    DET Parameters:
      κ = {results['kappa']:.4f}
      Lattice size = {N}³
      R in lattice = {results['R_lattice']:.1f} units
    
    Results:
      g (DET) = {g_surface:.4f} m/s²
      g (theory) = {g_theory:.4f} m/s²
      Error = {results['rel_error']:.2%}
    
    Relationship:
      g = |∇Φ| = GM/r²  (at surface)
      
    In DET:
      g_SI = g_lattice × (a / τ₀²)
    """
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nPlot saved to {save_path}")


def main():
    print("="*70)
    print("COMPUTING LOCAL GRAVITATIONAL ACCELERATION (little g) FROM DET")
    print("="*70)
    
    # Test 1: Earth
    print("\n" + "="*60)
    print("Test 1: Earth's Surface Gravity")
    print("="*60)
    
    earth_results = compute_surface_gravity(M_Earth, R_Earth, N=64, verbose=True)
    earth_g_vs_r = compute_g_vs_r(earth_results, verbose=False)
    create_visualization(earth_results, earth_g_vs_r, "Earth", 
                        '/home/ubuntu/det_v6_release/results/little_g_earth.png')
    
    # Test 2: Moon
    print("\n" + "="*60)
    print("Test 2: Moon's Surface Gravity")
    print("="*60)
    
    moon_results = compute_surface_gravity(M_Moon, R_Moon, N=64, verbose=True)
    
    # Test 3: Sun
    print("\n" + "="*60)
    print("Test 3: Sun's Surface Gravity")
    print("="*60)
    
    sun_results = compute_surface_gravity(M_Sun, R_Sun, N=64, verbose=True)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Surface Gravity Comparison")
    print("="*70)
    print(f"\n{'Body':<10} {'g (DET)':<15} {'g (theory)':<15} {'g (actual)':<15} {'Error':<10}")
    print("-"*65)
    print(f"{'Earth':<10} {earth_results['g_surface_SI']:<15.4f} {earth_results['g_theory']:<15.4f} {g_Earth:<15.4f} {earth_results['rel_error']:<10.2%}")
    print(f"{'Moon':<10} {moon_results['g_surface_SI']:<15.4f} {moon_results['g_theory']:<15.4f} {g_Moon:<15.4f} {moon_results['rel_error']:<10.2%}")
    print(f"{'Sun':<10} {sun_results['g_surface_SI']:<15.4f} {sun_results['g_theory']:<15.4f} {g_Sun:<15.4f} {sun_results['rel_error']:<10.2%}")
    
    print(f"""
    
    Key Formula:
    ============
    
    In DET, the local gravitational acceleration is:
    
        g = -∇Φ
    
    where Φ is the gravitational potential from the Poisson equation:
    
        L_σ Φ = -κ ρ
    
    At the surface of a uniform sphere:
    
        g = GM/R²
    
    To convert from lattice to SI units:
    
        g_SI = g_lattice × (a / τ₀²)
    
    where a is the lattice spacing and τ₀ is the time unit.
    """)
    
    return earth_results, moon_results, sun_results


if __name__ == "__main__":
    results = main()
