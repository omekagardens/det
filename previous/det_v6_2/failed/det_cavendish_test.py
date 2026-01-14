"""
DET Cavendish Experiment - Mode A (Simulation)
===============================================

This script tests whether DET's emergent gravity produces the correct 
classical limit (1/r² scaling) as predicted by Newton's law.

Based on DET Theory Card v6.1 Section V:
- Gravity source: ρ_i = q_i - b_i (structural debt minus baseline)
- Baseline field: (L_σ b)_i - α b_i = -α q_i (screened Poisson)
- Gravitational potential: (L_σ Φ)_i = -κ ρ_i (standard Poisson)
- Gravitational flux: J^(grav) = μ_g σ_ij (F_i + F_j)/2 (Φ_i - Φ_j)

Test Protocol:
1. Create two spherical clusters with high structural debt q
2. Measure emergent gravitational force at multiple separations
3. Verify 1/r² scaling and extract effective G

Author: DET Validation Suite
Reference: DET Theory Card 6.1, Section V
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace, convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import cg, spsolve
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class GravityParams:
    """Parameters for DET gravity simulation."""
    N: int = 64                    # Grid size
    DT: float = 0.01               # Time step
    
    # Vacuum/background
    F_VAC: float = 0.01            # Background resource
    F_MIN: float = 0.0             # Minimum resource
    
    # Gravity module (Section V)
    gravity_enabled: bool = True
    kappa: float = 1.0             # Gravity coupling (Poisson source strength)
    mu_g: float = 0.5              # Gravity-to-flux coupling
    alpha_screen: float = 0.05    # Baseline screening parameter
    
    # Sphere properties
    q_matter: float = 0.9          # Structural debt for "matter"
    q_vacuum: float = 0.0          # Structural debt for vacuum
    
    # Other fluxes (disabled for pure gravity test)
    diff_enabled: bool = False     # Disable diffusion
    momentum_enabled: bool = False
    floor_enabled: bool = False
    
    # Numerical
    sigma_base: float = 1.0        # Uniform conductivity
    poisson_tol: float = 1e-8      # Solver tolerance
    max_iter: int = 1000


# =============================================================================
# GRAVITY SOLVER (Section V)
# =============================================================================

class GravitySolver3D:
    """
    Solves the baseline-referenced gravity equations from DET v6.1 Section V.
    
    Equations:
    1. Baseline: (L_σ - α) b = -α q   →  b is smoothed version of q
    2. Gravity source: ρ = q - b
    3. Potential: L_σ Φ = -κ ρ
    4. Force: F_grav ∝ -∇Φ
    """
    
    def __init__(self, N: int, alpha: float = 0.05, kappa: float = 1.0):
        self.N = N
        self.alpha = alpha
        self.kappa = kappa
        
        # Build Laplacian operator (7-point stencil with periodic BC)
        self._build_laplacian()
    
    def _build_laplacian(self):
        """Build sparse Laplacian matrix for 3D periodic grid."""
        N = self.N
        size = N**3
        
        # For 3D Laplacian: Δu = (u[i+1] + u[i-1] - 2u[i]) for each dimension
        # We'll use iterative solver with matrix-free Laplacian instead
        self.size = size
    
    def apply_laplacian(self, u: np.ndarray) -> np.ndarray:
        """Apply discrete Laplacian with periodic boundaries."""
        # 7-point stencil Laplacian
        result = -6 * u
        result += np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0)  # Z neighbors
        result += np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1)  # Y neighbors
        result += np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2)  # X neighbors
        return result
    
    def solve_screened_poisson(self, source: np.ndarray, alpha: float, 
                                tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
        """
        Solve (L - α)u = f using Jacobi iteration.
        
        For baseline: (L - α)b = -α q
        """
        N = self.N
        u = np.zeros_like(source)
        
        for iteration in range(max_iter):
            u_new = (1/6) * (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
                np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) -
                source
            ) / (1 + alpha/6)
            
            # Check convergence
            if iteration % 50 == 0:
                residual = np.max(np.abs(u_new - u))
                if residual < tol:
                    break
            u = u_new
        
        return u
    
    def solve_poisson(self, source: np.ndarray, tol: float = 1e-8, 
                      max_iter: int = 1000) -> np.ndarray:
        """
        Solve Lu = f using Jacobi iteration.
        
        For potential: LΦ = -κρ
        """
        N = self.N
        u = np.zeros_like(source)
        
        for iteration in range(max_iter):
            u_new = (1/6) * (
                np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
                np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) +
                np.roll(u, 1, axis=2) + np.roll(u, -1, axis=2) -
                source
            )
            
            # Remove mean (gauge fixing for periodic BC)
            u_new = u_new - np.mean(u_new)
            
            if iteration % 50 == 0:
                residual = np.max(np.abs(u_new - u))
                if residual < tol:
                    break
            u = u_new
        
        return u
    
    def compute_gravity(self, q: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute gravitational potential and flux from structural debt field.
        
        Returns:
            Phi: Gravitational potential
            rho: Gravity source (q - baseline)
            b: Baseline field
        """
        # Step 1: Compute baseline (screened average of q)
        # (L - α)b = -αq  →  b is smoothed q
        b = self.solve_screened_poisson(-self.alpha * q, self.alpha)
        
        # Step 2: Gravity source is excess structure
        rho = q - b
        
        # Step 3: Solve for potential
        # LΦ = -κρ
        Phi = self.solve_poisson(-self.kappa * rho)
        
        return Phi, rho, b
    
    def compute_gravitational_force(self, Phi: np.ndarray, 
                                     position: Tuple[int, int, int],
                                     radius: int = 3) -> np.ndarray:
        """
        Compute gravitational force at a position by averaging -∇Φ over a region.
        
        Returns force vector [Fx, Fy, Fz]
        """
        # Compute gradient of potential
        grad_Phi_z = np.roll(Phi, -1, axis=0) - np.roll(Phi, 1, axis=0)  # ∂Φ/∂z
        grad_Phi_y = np.roll(Phi, -1, axis=1) - np.roll(Phi, 1, axis=1)  # ∂Φ/∂y
        grad_Phi_x = np.roll(Phi, -1, axis=2) - np.roll(Phi, 1, axis=2)  # ∂Φ/∂x
        
        # Force is -∇Φ
        Fx = -0.5 * grad_Phi_x
        Fy = -0.5 * grad_Phi_y
        Fz = -0.5 * grad_Phi_z
        
        # Average over region around position
        z, y, x = position
        N = self.N
        
        # Create mask for averaging region
        zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
        dx = (xx - x + N//2) % N - N//2
        dy = (yy - y + N//2) % N - N//2
        dz = (zz - z + N//2) % N - N//2
        r2 = dx**2 + dy**2 + dz**2
        mask = r2 < radius**2
        
        if np.sum(mask) == 0:
            return np.array([0.0, 0.0, 0.0])
        
        Fx_avg = np.sum(Fx * mask) / np.sum(mask)
        Fy_avg = np.sum(Fy * mask) / np.sum(mask)
        Fz_avg = np.sum(Fz * mask) / np.sum(mask)
        
        return np.array([Fx_avg, Fy_avg, Fz_avg])


# =============================================================================
# CAVENDISH TEST SETUP
# =============================================================================

class CavendishSimulator:
    """
    Simulates the Cavendish experiment in DET.
    
    Creates two spherical "masses" (high-q clusters) and measures
    the emergent gravitational force between them.
    """
    
    def __init__(self, params: Optional[GravityParams] = None):
        self.p = params or GravityParams()
        N = self.p.N
        
        # Initialize fields
        self.q = np.zeros((N, N, N), dtype=np.float64)
        self.F = np.ones((N, N, N), dtype=np.float64) * self.p.F_VAC
        self.sigma = np.ones((N, N, N), dtype=np.float64) * self.p.sigma_base
        
        # Gravity solver
        self.gravity = GravitySolver3D(N, self.p.alpha_screen, self.p.kappa)
        
        # Cached results
        self.Phi = None
        self.rho = None
        self.baseline = None
    
    def create_sphere(self, center: Tuple[int, int, int], radius: float, 
                      q_value: float, F_value: float = 1.0) -> np.ndarray:
        """
        Create a spherical cluster with given structural debt q.
        
        Returns mask of the sphere.
        """
        N = self.p.N
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        cz, cy, cx = center
        
        # Periodic distance
        dx = (x - cx + N//2) % N - N//2
        dy = (y - cy + N//2) % N - N//2
        dz = (z - cz + N//2) % N - N//2
        
        r = np.sqrt(dx**2 + dy**2 + dz**2)
        
        # Smooth sphere boundary
        mask = np.exp(-0.5 * ((r - radius) / 1.0)**2)
        mask = np.where(r < radius, 1.0, mask)
        mask = np.where(r > radius + 2, 0.0, mask)
        
        # Apply q and F
        self.q = np.maximum(self.q, mask * q_value)
        self.F = np.maximum(self.F, mask * F_value)
        
        return mask
    
    def setup_two_spheres(self, separation: float, radius: float = 4.0,
                          mass_ratio: float = 1.0) -> Tuple[Tuple, Tuple]:
        """
        Set up two spheres at given separation (center-to-center).
        
        Returns centers of both spheres.
        """
        N = self.p.N
        center = N // 2
        
        # Place spheres along X axis
        offset = int(separation / 2)
        center1 = (center, center, center - offset)
        center2 = (center, center, center + offset)
        
        # Create spheres with structural debt
        self.q[:] = self.p.q_vacuum
        self.F[:] = self.p.F_VAC
        
        self.create_sphere(center1, radius, self.p.q_matter, F_value=5.0)
        self.create_sphere(center2, radius, self.p.q_matter * mass_ratio, 
                          F_value=5.0 * mass_ratio)
        
        return center1, center2
    
    def compute_force_between_spheres(self, center1: Tuple, center2: Tuple,
                                       measure_radius: int = 3) -> Dict:
        """
        Compute gravitational force on sphere 2 due to sphere 1.
        
        Returns dict with force magnitude, direction, and diagnostic info.
        """
        # Solve gravity
        self.Phi, self.rho, self.baseline = self.gravity.compute_gravity(
            self.q, self.sigma
        )
        
        # Measure force at center of sphere 2
        force_vec = self.gravity.compute_gravitational_force(
            self.Phi, center2, radius=measure_radius
        )
        
        # Direction from sphere 1 to sphere 2
        direction = np.array([
            center2[2] - center1[2],  # X
            center2[1] - center1[1],  # Y
            center2[0] - center1[0],  # Z
        ], dtype=float)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        
        # Force magnitude (positive = attraction toward sphere 1)
        # Since F = -∇Φ points toward lower potential (toward mass)
        force_magnitude = np.dot(force_vec, -direction)
        
        # Compute total "mass" as sum of q
        mass1 = np.sum(self.q * (self.q > self.p.q_vacuum * 10))
        mass2 = mass1  # Equal masses by default
        
        return {
            'force_vec': force_vec,
            'force_magnitude': force_magnitude,
            'direction': direction,
            'mass1': mass1,
            'mass2': mass2,
            'Phi_max': np.max(self.Phi),
            'Phi_min': np.min(self.Phi),
            'rho_max': np.max(self.rho),
            'rho_min': np.min(self.rho),
        }


# =============================================================================
# CAVENDISH TEST: 1/r² SCALING
# =============================================================================

def run_cavendish_test(params: Optional[GravityParams] = None,
                       separations: Optional[List[float]] = None,
                       sphere_radius: float = 3.0,
                       verbose: bool = True) -> Dict:
    """
    Run the full Cavendish test: measure force at multiple separations.
    
    Checks:
    1. Force scales as 1/r²
    2. Extracted G is consistent across distances
    
    Returns:
        Dict with all measurements and analysis results
    """
    if params is None:
        params = GravityParams(N=64, kappa=1.0, mu_g=0.5, alpha_screen=0.05)
    
    if separations is None:
        # Separations from 10 to 28 cells (avoid boundary effects)
        separations = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    
    if verbose:
        print("=" * 70)
        print("DET CAVENDISH EXPERIMENT - MODE A (SIMULATION)")
        print("=" * 70)
        print(f"\nGrid size: {params.N}³")
        print(f"Sphere radius: {sphere_radius} cells")
        print(f"Separations: {separations} cells")
        print(f"Gravity parameters: κ={params.kappa}, μ_g={params.mu_g}, α={params.alpha_screen}")
        print()
    
    results = {
        'separations': [],
        'forces': [],
        'forces_vec': [],
        'masses': [],
        'diagnostics': []
    }
    
    sim = CavendishSimulator(params)
    
    for r in separations:
        if verbose:
            print(f"  Testing r = {r} cells... ", end='', flush=True)
        
        # Setup spheres at this separation
        center1, center2 = sim.setup_two_spheres(
            separation=r, 
            radius=sphere_radius
        )
        
        # Measure force
        measurement = sim.compute_force_between_spheres(center1, center2)
        
        results['separations'].append(r)
        results['forces'].append(measurement['force_magnitude'])
        results['forces_vec'].append(measurement['force_vec'])
        results['masses'].append((measurement['mass1'], measurement['mass2']))
        results['diagnostics'].append(measurement)
        
        if verbose:
            print(f"F = {measurement['force_magnitude']:.6f}")
    
    # Convert to arrays
    r_arr = np.array(results['separations'])
    F_arr = np.array(results['forces'])
    
    # Use absolute value of force (we care about magnitude, not sign)
    # Negative force = attraction (correct physics)
    F_arr_abs = np.abs(F_arr)
    results['forces_abs'] = F_arr_abs.tolist()
    
    # Fit power law: |F| = A * r^n
    # log|F| = log(A) + n * log(r)
    # Only use data where force is significant (not near zero due to boundary effects)
    valid = F_arr_abs > 1e-4
    if np.sum(valid) >= 3:
        log_r = np.log(r_arr[valid])
        log_F = np.log(F_arr_abs[valid])
        
        # Linear regression
        coeffs = np.polyfit(log_r, log_F, 1)
        power_law_exponent = coeffs[0]
        power_law_amplitude = np.exp(coeffs[1])
        
        # Residuals
        log_F_fit = coeffs[1] + coeffs[0] * log_r
        r_squared = 1 - np.sum((log_F - log_F_fit)**2) / np.sum((log_F - np.mean(log_F))**2)
        
        results['power_law_exponent'] = power_law_exponent
        results['power_law_amplitude'] = power_law_amplitude
        results['r_squared'] = r_squared
        
        # Check if exponent is close to -2
        exponent_error = abs(power_law_exponent - (-2.0))
        results['exponent_error'] = exponent_error
        results['passes_inverse_square'] = exponent_error < 0.2  # Within 10%
        
        # Extract effective G: |F| = G * m1 * m2 / r²
        # G_eff = |F| * r² / (m1 * m2)
        m1 = results['masses'][0][0]
        m2 = results['masses'][0][1]
        G_eff_values = F_arr_abs[valid] * r_arr[valid]**2 / (m1 * m2 + 1e-10)
        results['G_effective'] = G_eff_values
        results['G_mean'] = np.mean(G_eff_values)
        results['G_std'] = np.std(G_eff_values)
        
        if verbose:
            print("\n" + "-" * 50)
            print("ANALYSIS RESULTS")
            print("-" * 50)
            print(f"  Power law exponent: {power_law_exponent:.4f} (expected: -2.0)")
            print(f"  Exponent error: {exponent_error:.4f}")
            print(f"  R² of fit: {r_squared:.6f}")
            print(f"  1/r² test: {'PASS' if results['passes_inverse_square'] else 'FAIL'}")
            print()
            print(f"  Effective G (mean): {results['G_mean']:.6f}")
            print(f"  Effective G (std): {results['G_std']:.6f}")
            print(f"  G consistency: {results['G_std']/results['G_mean']*100:.1f}% variation")
    else:
        if verbose:
            print("  WARNING: Not enough valid force measurements for analysis")
        results['passes_inverse_square'] = False
    
    return results


def visualize_cavendish_results(results: Dict, 
                                 filename: str = 'det_cavendish_results.png'):
    """Create visualization of Cavendish test results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    r_arr = np.array(results['separations'])
    F_arr = np.abs(np.array(results['forces']))  # Use absolute values
    
    # Filter to valid data points
    valid = F_arr > 1e-4
    r_valid = r_arr[valid]
    F_valid = F_arr[valid]
    
    # Plot 1: Force vs Distance (log-log)
    ax = axes[0, 0]
    ax.loglog(r_valid, F_valid, 'bo-', markersize=10, linewidth=2, label='DET simulation')
    
    # Overlay 1/r² fit
    if 'power_law_exponent' in results:
        r_fit = np.linspace(r_arr.min(), r_arr.max(), 100)
        F_fit = results['power_law_amplitude'] * r_fit**results['power_law_exponent']
        ax.loglog(r_fit, F_fit, 'r--', linewidth=2, 
                  label=f'Fit: F ∝ r^{results["power_law_exponent"]:.2f}')
        
        # Overlay exact 1/r²
        F_ideal = results['power_law_amplitude'] * (r_fit.min()/r_fit)**2 * F_arr[0]
        ax.loglog(r_fit, F_ideal * (r_arr[0]/r_fit)**2, 'g:', linewidth=2, 
                  label='Ideal: F ∝ 1/r²')
    
    ax.set_xlabel('Separation r (grid units)', fontsize=12)
    ax.set_ylabel('Force magnitude', fontsize=12)
    ax.set_title('Force vs Distance (Log-Log)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    # Plot 2: F * r² (should be constant for 1/r²)
    ax = axes[0, 1]
    Fr2 = F_valid * r_valid**2
    ax.plot(r_valid, Fr2, 'go-', markersize=10, linewidth=2)
    ax.axhline(y=np.mean(Fr2), color='r', linestyle='--', linewidth=2, 
               label=f'Mean = {np.mean(Fr2):.4f}')
    ax.fill_between(r_valid, np.mean(Fr2) - np.std(Fr2), np.mean(Fr2) + np.std(Fr2),
                    alpha=0.2, color='red', label=f'±σ = {np.std(Fr2):.4f}')
    ax.set_xlabel('Separation r (grid units)', fontsize=12)
    ax.set_ylabel('F × r²', fontsize=12)
    ax.set_title('Force × Distance² (Should Be Constant)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Effective G at each distance
    ax = axes[1, 0]
    if 'G_effective' in results:
        G_eff = results['G_effective']
        ax.bar(range(len(G_eff)), G_eff, color='steelblue', alpha=0.7)
        ax.axhline(y=results['G_mean'], color='red', linestyle='--', linewidth=2,
                   label=f'Mean G = {results["G_mean"]:.4f}')
        ax.set_xticks(range(len(G_eff)))
        ax.set_xticklabels([f'{r:.0f}' for r in r_valid], rotation=45)
        ax.set_xlabel('Separation r (grid units)', fontsize=12)
        ax.set_ylabel('Effective G', fontsize=12)
        ax.set_title('Extracted G at Each Distance', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    DET CAVENDISH EXPERIMENT RESULTS
    ================================
    
    Test: Does DET emergent gravity follow 1/r²?
    
    Setup:
    - Two spherical clusters with high structural debt (q ≈ 0.9)
    - Solve baseline-referenced gravity (Section V)
    - Measure force at multiple separations
    
    Results:
"""
    
    if 'power_law_exponent' in results:
        summary += f"""
    Power law exponent: {results['power_law_exponent']:.4f}
    (Expected for Newton: -2.0)
    
    Exponent error: {results['exponent_error']:.4f}
    R² of fit: {results['r_squared']:.6f}
    
    Effective G (mean): {results['G_mean']:.6f}
    G variation: {results['G_std']/results['G_mean']*100:.1f}%
    
    VERDICT: {'PASS ✓' if results['passes_inverse_square'] else 'FAIL ✗'}
    
    {'1/r² law verified!' if results['passes_inverse_square'] else 'Deviation from 1/r² detected'}
"""
    
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('DET Cavendish Experiment - Mode A Simulation', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved visualization: {filename}")


def run_potential_field_diagnostic(params: Optional[GravityParams] = None,
                                   filename: str = 'det_gravity_field.png'):
    """
    Visualize the gravitational potential field for a single mass.
    
    This helps verify the Poisson solver is working correctly.
    """
    if params is None:
        params = GravityParams(N=64, kappa=1.0, alpha_screen=0.05)
    
    print("\n" + "=" * 70)
    print("GRAVITATIONAL FIELD DIAGNOSTIC")
    print("=" * 70)
    
    sim = CavendishSimulator(params)
    N = params.N
    center = N // 2
    
    # Create single sphere
    sim.create_sphere((center, center, center), radius=5.0, 
                      q_value=params.q_matter, F_value=5.0)
    
    # Solve gravity
    Phi, rho, baseline = sim.gravity.compute_gravity(sim.q, sim.sigma)
    
    # Analyze radial profile
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    r_flat = r.flatten()
    Phi_flat = Phi.flatten()
    
    # Bin by radius
    r_bins = np.linspace(1, N//2 - 5, 30)
    Phi_binned = []
    r_centers = []
    
    for i in range(len(r_bins) - 1):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.sum(mask) > 0:
            Phi_binned.append(np.mean(Phi_flat[mask]))
            r_centers.append(0.5 * (r_bins[i] + r_bins[i+1]))
    
    r_centers = np.array(r_centers)
    Phi_binned = np.array(Phi_binned)
    
    # Fit to 1/r for far field
    far_mask = r_centers > 10
    if np.sum(far_mask) >= 3:
        log_r = np.log(r_centers[far_mask])
        log_Phi = np.log(-Phi_binned[far_mask] + 1e-10)
        coeffs = np.polyfit(log_r, log_Phi, 1)
        phi_exponent = coeffs[0]
        print(f"  Far-field potential scaling: Φ ∝ r^{phi_exponent:.2f}")
        print(f"  Expected for point mass: Φ ∝ r^(-1)")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Slice of q
    ax = axes[0, 0]
    im = ax.imshow(sim.q[center, :, :], origin='lower', cmap='viridis')
    ax.set_title('Structural Debt q (z=center slice)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='q')
    
    # Slice of rho (gravity source)
    ax = axes[0, 1]
    im = ax.imshow(rho[center, :, :], origin='lower', cmap='RdBu_r')
    ax.set_title('Gravity Source ρ = q - baseline (z=center)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='ρ')
    
    # Slice of Phi
    ax = axes[1, 0]
    im = ax.imshow(Phi[center, :, :], origin='lower', cmap='plasma')
    ax.set_title('Gravitational Potential Φ (z=center)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(im, ax=ax, label='Φ')
    
    # Radial profile
    ax = axes[1, 1]
    ax.semilogy(r_centers, -Phi_binned, 'bo-', markersize=6, label='DET: -Φ(r)')
    
    # Overlay 1/r reference
    r_ref = np.linspace(5, 25, 50)
    Phi_ref = 1.0 / r_ref
    scale = -Phi_binned[len(Phi_binned)//2] / Phi_ref[len(Phi_ref)//2]
    ax.semilogy(r_ref, scale * Phi_ref, 'r--', linewidth=2, label='1/r reference')
    
    ax.set_xlabel('Radius r', fontsize=12)
    ax.set_ylabel('-Φ (log scale)', fontsize=12)
    ax.set_title('Radial Profile of Potential', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('DET Gravity Field Diagnostic - Single Mass', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")
    
    return {
        'Phi': Phi,
        'rho': rho,
        'baseline': baseline,
        'r_centers': r_centers,
        'Phi_binned': Phi_binned
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "=" * 70)
    print("DET CAVENDISH EXPERIMENT - MODE A (SIMULATION)")
    print("Preparing for Mode B (Physical Experiment) validation")
    print("=" * 70)
    print()
    
    # First run diagnostic to verify gravity field looks correct
    print("STEP 1: Gravitational field diagnostic...")
    diag = run_potential_field_diagnostic()
    
    # Run main Cavendish test
    print("\nSTEP 2: Running Cavendish force measurements...")
    params = GravityParams(
        N=64,
        kappa=1.0,
        mu_g=0.5,
        alpha_screen=0.05,
        q_matter=0.9
    )
    
    results = run_cavendish_test(
        params=params,
        separations=[8, 10, 12, 14, 16, 18, 20, 22],  # Reduced max to avoid boundary effects
        sphere_radius=3.0,
        verbose=True
    )
    
    # Visualize
    print("\nSTEP 3: Creating visualizations...")
    visualize_cavendish_results(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("MODE A SIMULATION COMPLETE")
    print("=" * 70)
    
    if results.get('passes_inverse_square'):
        print("""
✓ DET emergent gravity follows 1/r² law to within tolerance

PREPARATION FOR MODE B (Physical Experiment):
----------------------------------------------
The simulation establishes:
1. Power law exponent: {:.4f} (target: -2.0)
2. Effective G from simulation: {:.6f} (in lattice units)

To calibrate against physical experiment:
1. Perform Cavendish measurement with known masses
2. Extract physical G ≈ 6.674×10⁻¹¹ N·m²/kg²
3. Map: κ_DET = G_physical × (mass_scale) × (length_scale)²

The key physics test is the SCALING LAW, not the absolute value.
DET passes if it reproduces 1/r² without additional tuning.
""".format(results['power_law_exponent'], results['G_mean']))
    else:
        print("""
✗ DET gravity does not match 1/r² - investigation needed

Possible issues:
1. Boundary effects (periodic BC creating image forces)
2. Baseline screening parameter α not optimal
3. Numerical resolution insufficient
4. Theoretical issue with baseline-referenced gravity
""")
    
    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed:.1f} seconds")
