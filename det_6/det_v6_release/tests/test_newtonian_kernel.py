"""
DET v6 Test 3.1: Newtonian Kernel (1/r Potential)
==================================================

This test verifies that the baseline-referenced gravity module produces
a Newtonian 1/r potential in the far-field limit.

Setup:
- 3D cubic lattice with N^3 nodes
- Initialize a compact region of high structural debt q (Gaussian blob)
- Compute baseline b via Helmholtz equation: (L_σ b)_i - α b_i = -α q_i
- Compute gravity source ρ = q - b
- Solve Poisson equation: (L_σ Φ)_i = -κ ρ_i
- Measure Φ(r) as function of distance from mass center

Success Criterion:
- Far-field (> a few lattice spacings) fits Φ(r) ≈ -A/r + B
- Monopole moment A > 0 and scales with total Σρ
- Violation of 1/r → gravity falsified (G.8.2)

Reference: DET Theory Card v6.0, Section V
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, cg
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import time


@dataclass
class GravityTestParams:
    """Parameters for the Newtonian kernel test."""
    N: int = 64                     # Grid size (N^3 nodes)
    
    # Structural debt (q) initialization
    q_center: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # Normalized center
    q_amplitude: float = 0.8        # Peak q value
    q_width: float = 3.0            # Gaussian width in lattice units
    
    # Helmholtz baseline parameters
    alpha: float = 0.1              # Screening parameter for baseline
    
    # Poisson gravity parameters
    kappa: float = 1.0              # Gravity coupling constant
    
    # Conductivity
    sigma: float = 1.0              # Uniform conductivity
    
    # Analysis
    r_min_fit: float = 8.0          # Minimum radius for 1/r fit (lattice units)
    r_max_fit: float = 25.0         # Maximum radius for 1/r fit


def build_3d_laplacian(N: int, sigma: float = 1.0) -> sparse.csr_matrix:
    """
    Build the weighted graph Laplacian L_σ for a 3D cubic lattice
    with periodic boundary conditions.
    
    L_σ[i,j] = -σ_{ij} for neighbors
    L_σ[i,i] = Σ_j σ_{ij} (sum of neighbor conductivities)
    
    For uniform σ, this is just σ times the standard Laplacian.
    """
    n_nodes = N**3
    
    # For a 3D cubic lattice with 6 neighbors per node
    # Each node has degree 6 (periodic BCs)
    diag = np.ones(n_nodes) * 6 * sigma
    
    # Build sparse matrix using COO format then convert
    row = []
    col = []
    data = []
    
    def idx(x, y, z):
        return ((x % N) * N + (y % N)) * N + (z % N)
    
    for x in range(N):
        for y in range(N):
            for z in range(N):
                i = idx(x, y, z)
                # Diagonal
                row.append(i)
                col.append(i)
                data.append(6 * sigma)
                
                # Six neighbors (periodic)
                neighbors = [
                    idx(x+1, y, z), idx(x-1, y, z),
                    idx(x, y+1, z), idx(x, y-1, z),
                    idx(x, y, z+1), idx(x, y, z-1)
                ]
                for j in neighbors:
                    row.append(i)
                    col.append(j)
                    data.append(-sigma)
    
    L = sparse.csr_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    return L


def initialize_structural_debt(N: int, params: GravityTestParams) -> np.ndarray:
    """
    Initialize structural debt q as a Gaussian blob centered at params.q_center.
    """
    q = np.zeros((N, N, N))
    
    cx = int(params.q_center[0] * N)
    cy = int(params.q_center[1] * N)
    cz = int(params.q_center[2] * N)
    
    for x in range(N):
        for y in range(N):
            for z in range(N):
                # Distance with periodic boundary handling
                dx = min(abs(x - cx), N - abs(x - cx))
                dy = min(abs(y - cy), N - abs(y - cy))
                dz = min(abs(z - cz), N - abs(z - cz))
                r2 = dx**2 + dy**2 + dz**2
                
                q[x, y, z] = params.q_amplitude * np.exp(-0.5 * r2 / params.q_width**2)
    
    return q


def solve_helmholtz_baseline(L: sparse.csr_matrix, q: np.ndarray, 
                              alpha: float) -> np.ndarray:
    """
    Solve the Helmholtz equation for the baseline field b:
    (L_σ - α I) b = -α q
    
    Rearranged: (L_σ - α I) b = -α q
    """
    n_nodes = L.shape[0]
    N = int(round(n_nodes ** (1/3)))
    
    q_flat = q.flatten()
    
    # Build (L_σ - α I)
    A = L - alpha * sparse.eye(n_nodes)
    
    # RHS = -α q
    rhs = -alpha * q_flat
    
    # Solve using conjugate gradient (symmetric positive definite after shift)
    # Note: L_σ - α I may not be positive definite, use direct solver
    b_flat, info = cg(A, rhs, maxiter=1000, atol=1e-10)
    
    if info != 0:
        print(f"  Warning: Helmholtz solver did not converge (info={info})")
        # Fall back to direct solver
        b_flat = spsolve(A, rhs)
    
    return b_flat.reshape((N, N, N))


def solve_poisson_potential(L: sparse.csr_matrix, rho: np.ndarray, 
                            kappa: float) -> np.ndarray:
    """
    Solve the Poisson equation for the gravitational potential Φ:
    L_σ Φ = -κ ρ
    
    The Laplacian is singular (constant functions are in the null space),
    so we fix Φ at one point (e.g., set mean to zero).
    """
    n_nodes = L.shape[0]
    N = int(round(n_nodes ** (1/3)))
    
    rho_flat = rho.flatten()
    
    # RHS = -κ ρ
    rhs = -kappa * rho_flat
    
    # Make RHS sum to zero (compatibility condition for Neumann BCs)
    rhs = rhs - np.mean(rhs)
    
    # Add regularization to make L invertible (fix gauge)
    # We add a small term to pin the mean of Φ
    eps = 1e-6
    A = L + eps * sparse.eye(n_nodes)
    
    # Solve
    Phi_flat, info = cg(A, rhs, maxiter=2000, atol=1e-10)
    
    if info != 0:
        print(f"  Warning: Poisson solver did not converge (info={info})")
        Phi_flat = spsolve(A, rhs)
    
    # Remove mean to set gauge
    Phi_flat = Phi_flat - np.mean(Phi_flat)
    
    return Phi_flat.reshape((N, N, N))


def compute_radial_profile(field: np.ndarray, center: Tuple[int, int, int],
                           n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the radial profile of a 3D field from a given center.
    
    Returns:
        r_bins: Bin centers (radii)
        mean_vals: Mean field value in each radial bin
        std_vals: Standard deviation in each radial bin
    """
    N = field.shape[0]
    cx, cy, cz = center
    
    # Compute distances
    x = np.arange(N)
    y = np.arange(N)
    z = np.arange(N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Handle periodic boundaries
    DX = np.minimum(np.abs(X - cx), N - np.abs(X - cx))
    DY = np.minimum(np.abs(Y - cy), N - np.abs(Y - cy))
    DZ = np.minimum(np.abs(Z - cz), N - np.abs(Z - cz))
    
    R = np.sqrt(DX**2 + DY**2 + DZ**2)
    
    # Bin by radius
    r_max = N / 2
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    mean_vals = np.zeros(n_bins)
    std_vals = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (R >= r_edges[i]) & (R < r_edges[i+1])
        if np.sum(mask) > 0:
            vals = field[mask]
            mean_vals[i] = np.mean(vals)
            std_vals[i] = np.std(vals)
            counts[i] = np.sum(mask)
    
    return r_bins, mean_vals, std_vals


def fit_newtonian_potential(r: np.ndarray, Phi: np.ndarray, 
                            r_min: float, r_max: float) -> Dict:
    """
    Fit the potential to Φ(r) = -A/r + B in the specified range.
    
    Returns:
        Dictionary with fit parameters and quality metrics
    """
    # Select fitting range
    mask = (r >= r_min) & (r <= r_max) & (r > 0)
    r_fit = r[mask]
    Phi_fit = Phi[mask]
    
    if len(r_fit) < 3:
        return {'success': False, 'error': 'Insufficient data points'}
    
    # Define model: Φ(r) = -A/r + B
    def model(r, A, B):
        return -A / r + B
    
    try:
        popt, pcov = curve_fit(model, r_fit, Phi_fit, p0=[1.0, 0.0], maxfev=5000)
        A, B = popt
        A_err, B_err = np.sqrt(np.diag(pcov))
        
        # Compute R² (coefficient of determination)
        Phi_pred = model(r_fit, A, B)
        ss_res = np.sum((Phi_fit - Phi_pred)**2)
        ss_tot = np.sum((Phi_fit - np.mean(Phi_fit))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Compute residuals
        residuals = Phi_fit - Phi_pred
        rms_residual = np.sqrt(np.mean(residuals**2))
        
        return {
            'success': True,
            'A': A,
            'A_err': A_err,
            'B': B,
            'B_err': B_err,
            'r_squared': r_squared,
            'rms_residual': rms_residual,
            'n_points': len(r_fit),
            'r_range': (r_min, r_max)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_newtonian_kernel_test(params: Optional[GravityTestParams] = None,
                               verbose: bool = True) -> Dict:
    """
    Run the complete Newtonian kernel test.
    
    Returns:
        Dictionary with all test results and diagnostics
    """
    if params is None:
        params = GravityTestParams()
    
    N = params.N
    results = {'params': params, 'success': False}
    
    if verbose:
        print("="*70)
        print("DET v6 TEST 3.1: NEWTONIAN KERNEL (1/r POTENTIAL)")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Grid size: {N}³ = {N**3} nodes")
        print(f"  q amplitude: {params.q_amplitude}")
        print(f"  q width: {params.q_width} lattice units")
        print(f"  Helmholtz α: {params.alpha}")
        print(f"  Gravity κ: {params.kappa}")
    
    # Step 1: Build Laplacian
    if verbose:
        print(f"\n[1/6] Building 3D Laplacian...")
    t0 = time.time()
    L = build_3d_laplacian(N, params.sigma)
    results['laplacian_time'] = time.time() - t0
    if verbose:
        print(f"      Done in {results['laplacian_time']:.2f}s")
    
    # Step 2: Initialize structural debt q
    if verbose:
        print(f"[2/6] Initializing structural debt q...")
    q = initialize_structural_debt(N, params)
    results['q'] = q
    results['total_q'] = np.sum(q)
    if verbose:
        print(f"      Total q: {results['total_q']:.4f}")
        print(f"      Max q: {np.max(q):.4f}")
    
    # Step 3: Solve Helmholtz for baseline b
    if verbose:
        print(f"[3/6] Solving Helmholtz equation for baseline b...")
    t0 = time.time()
    b = solve_helmholtz_baseline(L, q, params.alpha)
    results['helmholtz_time'] = time.time() - t0
    results['b'] = b
    results['total_b'] = np.sum(b)
    if verbose:
        print(f"      Done in {results['helmholtz_time']:.2f}s")
        print(f"      Total b: {results['total_b']:.4f}")
    
    # Step 4: Compute gravity source ρ = q - b
    if verbose:
        print(f"[4/6] Computing gravity source ρ = q - b...")
    rho = q - b
    results['rho'] = rho
    results['total_rho'] = np.sum(rho)
    if verbose:
        print(f"      Total ρ: {results['total_rho']:.4f}")
        print(f"      Max ρ: {np.max(rho):.4f}")
        print(f"      Min ρ: {np.min(rho):.4f}")
    
    # Step 5: Solve Poisson for potential Φ
    if verbose:
        print(f"[5/6] Solving Poisson equation for potential Φ...")
    t0 = time.time()
    Phi = solve_poisson_potential(L, rho, params.kappa)
    results['poisson_time'] = time.time() - t0
    results['Phi'] = Phi
    if verbose:
        print(f"      Done in {results['poisson_time']:.2f}s")
        print(f"      Φ range: [{np.min(Phi):.4f}, {np.max(Phi):.4f}]")
    
    # Step 6: Compute radial profile and fit
    if verbose:
        print(f"[6/6] Computing radial profile and fitting 1/r...")
    
    center = (int(params.q_center[0] * N),
              int(params.q_center[1] * N),
              int(params.q_center[2] * N))
    
    r_bins, Phi_mean, Phi_std = compute_radial_profile(Phi, center, n_bins=60)
    results['r_bins'] = r_bins
    results['Phi_mean'] = Phi_mean
    results['Phi_std'] = Phi_std
    
    # Also compute radial profiles of q, b, rho
    _, q_mean, _ = compute_radial_profile(q, center, n_bins=60)
    _, b_mean, _ = compute_radial_profile(b, center, n_bins=60)
    _, rho_mean, _ = compute_radial_profile(rho, center, n_bins=60)
    results['q_mean'] = q_mean
    results['b_mean'] = b_mean
    results['rho_mean'] = rho_mean
    
    # Fit 1/r in far-field
    fit_result = fit_newtonian_potential(r_bins, Phi_mean, 
                                         params.r_min_fit, params.r_max_fit)
    results['fit'] = fit_result
    
    if verbose:
        print(f"\n" + "="*70)
        print("FIT RESULTS")
        print("="*70)
        if fit_result['success']:
            print(f"  Fit range: r ∈ [{fit_result['r_range'][0]:.1f}, {fit_result['r_range'][1]:.1f}]")
            print(f"  Data points: {fit_result['n_points']}")
            print(f"  Model: Φ(r) = -A/r + B")
            print(f"  A = {fit_result['A']:.6f} ± {fit_result['A_err']:.6f}")
            print(f"  B = {fit_result['B']:.6f} ± {fit_result['B_err']:.6f}")
            print(f"  R² = {fit_result['r_squared']:.6f}")
            print(f"  RMS residual = {fit_result['rms_residual']:.6e}")
        else:
            print(f"  Fit FAILED: {fit_result.get('error', 'Unknown error')}")
    
    # Evaluate success criteria
    if fit_result['success']:
        # Success if:
        # 1. A > 0 (attractive potential)
        # 2. R² > 0.95 (good fit quality)
        # 3. A scales with total ρ (check sign consistency)
        
        A_positive = fit_result['A'] > 0
        good_fit = fit_result['r_squared'] > 0.95
        rho_positive = results['total_rho'] > 0
        sign_consistent = (A_positive == rho_positive) or abs(results['total_rho']) < 1e-6
        
        results['success'] = A_positive and good_fit
        results['A_positive'] = A_positive
        results['good_fit'] = good_fit
        results['sign_consistent'] = sign_consistent
        
        if verbose:
            print(f"\n" + "="*70)
            print("SUCCESS CRITERIA")
            print("="*70)
            print(f"  A > 0 (monopole positive): {'PASS' if A_positive else 'FAIL'} (A = {fit_result['A']:.6f})")
            print(f"  R² > 0.95 (good 1/r fit): {'PASS' if good_fit else 'FAIL'} (R² = {fit_result['r_squared']:.6f})")
            print(f"  Sign consistency (A ↔ Σρ): {'PASS' if sign_consistent else 'FAIL'}")
            print(f"\n  OVERALL: {'PASS - Newtonian 1/r kernel verified' if results['success'] else 'FAIL - Gravity falsified (G.8.2)'}")
    
    return results


def plot_results(results: Dict, save_path: Optional[str] = None):
    """Generate visualization of the test results."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Radial profiles of q, b, ρ
    ax1 = axes[0, 0]
    r = results['r_bins']
    ax1.plot(r, results['q_mean'], 'b-', linewidth=2, label='q (structural debt)')
    ax1.plot(r, results['b_mean'], 'g--', linewidth=2, label='b (baseline)')
    ax1.plot(r, results['rho_mean'], 'r-', linewidth=2, label='ρ = q - b (source)')
    ax1.set_xlabel('Radius r (lattice units)', fontsize=12)
    ax1.set_ylabel('Field value', fontsize=12)
    ax1.set_title('Radial Profiles: q, b, and ρ', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(r))
    
    # Plot 2: Potential Φ(r) with 1/r fit
    ax2 = axes[0, 1]
    Phi_mean = results['Phi_mean']
    Phi_std = results['Phi_std']
    
    # Plot data with error bars
    ax2.errorbar(r, Phi_mean, yerr=Phi_std, fmt='ko', markersize=4, 
                 capsize=2, alpha=0.6, label='Φ(r) data')
    
    # Plot fit if successful
    fit = results['fit']
    if fit['success']:
        r_fit = np.linspace(fit['r_range'][0], fit['r_range'][1], 100)
        Phi_fit = -fit['A'] / r_fit + fit['B']
        ax2.plot(r_fit, Phi_fit, 'r-', linewidth=2, 
                 label=f"Fit: Φ = -{fit['A']:.4f}/r + {fit['B']:.4f}")
        
        # Mark fit range
        ax2.axvline(fit['r_range'][0], color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(fit['r_range'][1], color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Radius r (lattice units)', fontsize=12)
    ax2.set_ylabel('Potential Φ', fontsize=12)
    ax2.set_title('Gravitational Potential Φ(r)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(r))
    
    # Plot 3: Φ(r) × r (should be constant in far-field for 1/r)
    ax3 = axes[1, 0]
    # Avoid division by zero
    valid = r > 1
    Phi_times_r = Phi_mean[valid] * r[valid]
    ax3.plot(r[valid], Phi_times_r, 'ko-', markersize=4, alpha=0.7)
    
    if fit['success']:
        # Expected: Φ × r = -A + B × r
        r_theory = np.linspace(fit['r_range'][0], fit['r_range'][1], 100)
        expected = -fit['A'] + fit['B'] * r_theory
        ax3.plot(r_theory, expected, 'r-', linewidth=2, label=f'Expected: -A + Br')
        ax3.axhline(-fit['A'], color='blue', linestyle='--', alpha=0.5, 
                    label=f'-A = {-fit["A"]:.4f}')
    
    ax3.set_xlabel('Radius r (lattice units)', fontsize=12)
    ax3.set_ylabel('Φ(r) × r', fontsize=12)
    ax3.set_title('Φ × r (constant → 1/r verified)', fontsize=14)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, max(r))
    
    # Plot 4: Log-log plot to verify power law
    ax4 = axes[1, 1]
    
    # For 1/r potential, log|Φ - B| vs log(r) should have slope -1
    if fit['success']:
        B = fit['B']
        valid_log = (r > 3) & (np.abs(Phi_mean - B) > 1e-10)
        r_log = r[valid_log]
        Phi_shifted = np.abs(Phi_mean[valid_log] - B)
        
        ax4.loglog(r_log, Phi_shifted, 'ko', markersize=5, alpha=0.7, label='|Φ - B|')
        
        # Reference line with slope -1
        r_ref = np.logspace(np.log10(5), np.log10(25), 50)
        ref_line = fit['A'] / r_ref
        ax4.loglog(r_ref, ref_line, 'r-', linewidth=2, label='1/r reference')
        
        ax4.set_xlabel('Radius r (lattice units)', fontsize=12)
        ax4.set_ylabel('|Φ - B|', fontsize=12)
        ax4.set_title('Log-Log Plot (slope = -1 for 1/r)', fontsize=14)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # Add overall title with result
    status = "PASS" if results['success'] else "FAIL"
    fig.suptitle(f'DET Test 3.1: Newtonian Kernel - {status}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.close()
    return fig


def run_scaling_test(verbose: bool = True) -> Dict:
    """
    Test that the monopole moment A scales with total source Σρ.
    """
    if verbose:
        print("\n" + "="*70)
        print("SCALING TEST: A vs Σρ")
        print("="*70)
    
    amplitudes = [0.2, 0.4, 0.6, 0.8, 1.0]
    results_list = []
    
    for amp in amplitudes:
        params = GravityTestParams(N=48, q_amplitude=amp, r_min_fit=6, r_max_fit=18)
        result = run_newtonian_kernel_test(params, verbose=False)
        
        if result['fit']['success']:
            results_list.append({
                'amplitude': amp,
                'total_rho': result['total_rho'],
                'A': result['fit']['A'],
                'R_squared': result['fit']['r_squared']
            })
            if verbose:
                print(f"  q_amp={amp:.1f}: Σρ={result['total_rho']:.2f}, A={result['fit']['A']:.4f}, R²={result['fit']['r_squared']:.4f}")
    
    # Check linear scaling
    if len(results_list) >= 3:
        rhos = np.array([r['total_rho'] for r in results_list])
        As = np.array([r['A'] for r in results_list])
        
        # Linear fit: A = m * Σρ + c
        coeffs = np.polyfit(rhos, As, 1)
        m, c = coeffs
        
        # Compute R² for linear fit
        A_pred = m * rhos + c
        ss_res = np.sum((As - A_pred)**2)
        ss_tot = np.sum((As - np.mean(As))**2)
        r_squared_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        if verbose:
            print(f"\n  Linear fit: A = {m:.4f} × Σρ + {c:.4f}")
            print(f"  R² (linear) = {r_squared_linear:.4f}")
            print(f"  Scaling {'VERIFIED' if r_squared_linear > 0.95 else 'NOT VERIFIED'}")
        
        return {
            'results': results_list,
            'slope': m,
            'intercept': c,
            'r_squared': r_squared_linear,
            'scaling_verified': r_squared_linear > 0.95
        }
    
    return {'results': results_list, 'scaling_verified': False}


if __name__ == "__main__":
    # Run main test
    params = GravityTestParams(N=64)
    results = run_newtonian_kernel_test(params, verbose=True)
    
    # Generate plots
    plot_results(results, save_path='/home/ubuntu/det_v6_release/results/newtonian_kernel_test.png')
    
    # Run scaling test
    scaling = run_scaling_test(verbose=True)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  1/r potential test: {'PASS' if results['success'] else 'FAIL'}")
    print(f"  A scaling test: {'PASS' if scaling.get('scaling_verified', False) else 'FAIL'}")
