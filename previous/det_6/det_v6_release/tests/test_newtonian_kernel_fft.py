"""
DET v6 Test 3.1: Newtonian Kernel (1/r Potential) - FFT Version
===============================================================

This version uses FFT-based Poisson solver for much faster computation.

Reference: DET Theory Card v6.0, Section V
"""

import numpy as np
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
    q_center: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    q_amplitude: float = 0.8
    q_width: float = 4.0
    
    # Helmholtz baseline parameters
    alpha: float = 0.05
    
    # Poisson gravity parameters
    kappa: float = 1.0
    
    # Analysis
    r_min_fit: float = 8.0
    r_max_fit: float = 28.0


def initialize_structural_debt(N: int, params: GravityTestParams) -> np.ndarray:
    """Initialize structural debt q as a Gaussian blob."""
    cx = int(params.q_center[0] * N)
    cy = int(params.q_center[1] * N)
    cz = int(params.q_center[2] * N)
    
    x = np.arange(N)
    y = np.arange(N)
    z = np.arange(N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Handle periodic boundaries
    DX = np.minimum(np.abs(X - cx), N - np.abs(X - cx))
    DY = np.minimum(np.abs(Y - cy), N - np.abs(Y - cy))
    DZ = np.minimum(np.abs(Z - cz), N - np.abs(Z - cz))
    
    R2 = DX**2 + DY**2 + DZ**2
    q = params.q_amplitude * np.exp(-0.5 * R2 / params.q_width**2)
    
    return q


def fft_poisson_solve(source: np.ndarray, kappa: float = 1.0) -> np.ndarray:
    """
    Solve Poisson equation ∇²Φ = -κ*source using FFT.
    
    For a 3D periodic domain, the Laplacian in Fourier space is:
    L_k = -4 * (sin²(πkx/N) + sin²(πky/N) + sin²(πkz/N))
    
    So Φ_k = κ * source_k / L_k (with L_k=0 mode set to 0)
    """
    N = source.shape[0]
    
    # FFT of source
    source_k = np.fft.fftn(source)
    
    # Build Laplacian eigenvalues
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kz = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    # Laplacian eigenvalues for discrete Laplacian
    # L = 2*(cos(2πk/N) - 1) = -4*sin²(πk/N)
    L_k = -4 * (np.sin(np.pi * KX / N)**2 + 
                np.sin(np.pi * KY / N)**2 + 
                np.sin(np.pi * KZ / N)**2)
    
    # Avoid division by zero at k=0
    L_k[0, 0, 0] = 1.0
    
    # Solve: Φ_k = -κ * source_k / L_k
    Phi_k = -kappa * source_k / L_k
    
    # Set k=0 mode to 0 (gauge choice)
    Phi_k[0, 0, 0] = 0.0
    
    # Inverse FFT
    Phi = np.real(np.fft.ifftn(Phi_k))
    
    return Phi


def fft_helmholtz_solve(q: np.ndarray, alpha: float) -> np.ndarray:
    """
    Solve Helmholtz equation (∇² - α)b = -α*q using FFT.
    
    In Fourier space: (L_k - α)b_k = -α*q_k
    So: b_k = -α*q_k / (L_k - α)
    """
    N = q.shape[0]
    
    # FFT of q
    q_k = np.fft.fftn(q)
    
    # Build Laplacian eigenvalues
    kx = np.fft.fftfreq(N) * N
    ky = np.fft.fftfreq(N) * N
    kz = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    
    L_k = -4 * (np.sin(np.pi * KX / N)**2 + 
                np.sin(np.pi * KY / N)**2 + 
                np.sin(np.pi * KZ / N)**2)
    
    # Solve: (L_k - α)b_k = -α*q_k
    denom = L_k - alpha
    
    # Avoid division by zero (shouldn't happen for α > 0)
    denom[np.abs(denom) < 1e-12] = 1e-12
    
    b_k = -alpha * q_k / denom
    
    # Inverse FFT
    b = np.real(np.fft.ifftn(b_k))
    
    return b


def compute_radial_profile(field: np.ndarray, center: Tuple[int, int, int],
                           n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the radial profile of a 3D field."""
    N = field.shape[0]
    cx, cy, cz = center
    
    x = np.arange(N)
    y = np.arange(N)
    z = np.arange(N)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    DX = np.minimum(np.abs(X - cx), N - np.abs(X - cx))
    DY = np.minimum(np.abs(Y - cy), N - np.abs(Y - cy))
    DZ = np.minimum(np.abs(Z - cz), N - np.abs(Z - cz))
    
    R = np.sqrt(DX**2 + DY**2 + DZ**2)
    
    r_max = N / 2
    r_edges = np.linspace(0, r_max, n_bins + 1)
    r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])
    
    mean_vals = np.zeros(n_bins)
    std_vals = np.zeros(n_bins)
    
    for i in range(n_bins):
        mask = (R >= r_edges[i]) & (R < r_edges[i+1])
        if np.sum(mask) > 0:
            vals = field[mask]
            mean_vals[i] = np.mean(vals)
            std_vals[i] = np.std(vals)
    
    return r_bins, mean_vals, std_vals


def fit_newtonian_potential(r: np.ndarray, Phi: np.ndarray, 
                            r_min: float, r_max: float) -> Dict:
    """
    Fit the potential to Φ(r) = A/r + B.
    
    For attractive gravity (like Newton), we expect A < 0 (potential well).
    For the raw Poisson with positive source, we get A > 0 (repulsive).
    """
    mask = (r >= r_min) & (r <= r_max) & (r > 0)
    r_fit = r[mask]
    Phi_fit = Phi[mask]
    
    if len(r_fit) < 3:
        return {'success': False, 'error': 'Insufficient data points'}
    
    def model(r, A, B):
        return A / r + B
    
    try:
        popt, pcov = curve_fit(model, r_fit, Phi_fit, p0=[1.0, 0.0], maxfev=5000)
        A, B = popt
        A_err, B_err = np.sqrt(np.diag(pcov))
        
        Phi_pred = model(r_fit, A, B)
        ss_res = np.sum((Phi_fit - Phi_pred)**2)
        ss_tot = np.sum((Phi_fit - np.mean(Phi_fit))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'success': True,
            'A': A,
            'A_err': A_err,
            'B': B,
            'B_err': B_err,
            'r_squared': r_squared,
            'n_points': len(r_fit),
            'r_range': (r_min, r_max)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def fit_screened_potential(r: np.ndarray, Phi: np.ndarray, 
                           r_min: float, r_max: float) -> Dict:
    """Fit to screened/Yukawa form: Φ(r) = -A * exp(-μr) / r + B"""
    mask = (r >= r_min) & (r <= r_max) & (r > 0)
    r_fit = r[mask]
    Phi_fit = Phi[mask]
    
    if len(r_fit) < 4:
        return {'success': False, 'error': 'Insufficient data points'}
    
    def model(r, A, mu, B):
        return -A * np.exp(-mu * r) / r + B
    
    try:
        popt, pcov = curve_fit(model, r_fit, Phi_fit, 
                               p0=[1.0, 0.1, 0.0], 
                               bounds=([0, 0, -np.inf], [np.inf, 1.0, np.inf]),
                               maxfev=10000)
        A, mu, B = popt
        perr = np.sqrt(np.diag(pcov))
        
        Phi_pred = model(r_fit, A, mu, B)
        ss_res = np.sum((Phi_fit - Phi_pred)**2)
        ss_tot = np.sum((Phi_fit - np.mean(Phi_fit))**2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        return {
            'success': True,
            'A': A,
            'mu': mu,
            'B': B,
            'A_err': perr[0],
            'mu_err': perr[1],
            'B_err': perr[2],
            'r_squared': r_squared,
            'screening_length': 1/mu if mu > 0 else np.inf
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


def run_test(params: Optional[GravityTestParams] = None,
             verbose: bool = True) -> Dict:
    """Run the complete Newtonian kernel test."""
    
    if params is None:
        params = GravityTestParams()
    
    N = params.N
    results = {'params': params}
    
    if verbose:
        print("="*70)
        print("DET v6 TEST 3.1: NEWTONIAN KERNEL (FFT VERSION)")
        print("="*70)
        print(f"\nParameters:")
        print(f"  Grid size: {N}³ = {N**3} nodes")
        print(f"  q amplitude: {params.q_amplitude}")
        print(f"  q width: {params.q_width} lattice units")
        print(f"  Helmholtz α: {params.alpha}")
        print(f"  Gravity κ: {params.kappa}")
    
    # Initialize q
    if verbose:
        print(f"\n[1/5] Initializing structural debt q...")
    q = initialize_structural_debt(N, params)
    results['q'] = q
    results['total_q'] = np.sum(q)
    if verbose:
        print(f"      Total q: {results['total_q']:.4f}")
    
    center = (int(params.q_center[0] * N),
              int(params.q_center[1] * N),
              int(params.q_center[2] * N))
    
    # =========================================================
    # TEST A: RAW q-SOURCED GRAVITY
    # =========================================================
    if verbose:
        print(f"\n[2/5] TEST A: Raw q-sourced gravity (FFT Poisson)...")
    
    t0 = time.time()
    Phi_raw = fft_poisson_solve(q, params.kappa)
    results['Phi_raw'] = Phi_raw
    if verbose:
        print(f"      Done in {time.time()-t0:.3f}s")
        print(f"      Φ_raw range: [{np.min(Phi_raw):.4f}, {np.max(Phi_raw):.4f}]")
    
    # Radial profile
    r_bins, Phi_raw_mean, Phi_raw_std = compute_radial_profile(Phi_raw, center, n_bins=60)
    results['r_bins'] = r_bins
    results['Phi_raw_mean'] = Phi_raw_mean
    
    # Fit 1/r
    fit_raw = fit_newtonian_potential(r_bins, Phi_raw_mean, 
                                       params.r_min_fit, params.r_max_fit)
    results['fit_raw'] = fit_raw
    
    if verbose:
        print(f"\n      RAW GRAVITY FIT (Φ = -A/r + B):")
        if fit_raw['success']:
            print(f"        A = {fit_raw['A']:.4f} ± {fit_raw['A_err']:.4f}")
            print(f"        B = {fit_raw['B']:.4f} ± {fit_raw['B_err']:.4f}")
            print(f"        R² = {fit_raw['r_squared']:.6f}")
            print(f"        |A| > 0: {'PASS' if abs(fit_raw['A']) > 0 else 'FAIL'}")
            print(f"        R² > 0.95: {'PASS' if fit_raw['r_squared'] > 0.95 else 'FAIL'}")
    
    # =========================================================
    # TEST B: BASELINE-REFERENCED GRAVITY
    # =========================================================
    if verbose:
        print(f"\n[3/5] TEST B: Baseline-referenced gravity...")
        print(f"      Solving Helmholtz for baseline b...")
    
    t0 = time.time()
    b = fft_helmholtz_solve(q, params.alpha)
    results['b'] = b
    if verbose:
        print(f"      Done in {time.time()-t0:.3f}s")
    
    # Compute ρ = q - b
    rho = q - b
    results['rho'] = rho
    results['total_rho'] = np.sum(rho)
    if verbose:
        print(f"      Total ρ = Σ(q-b): {results['total_rho']:.6f}")
    
    # Solve Poisson with ρ
    if verbose:
        print(f"\n[4/5] Solving Poisson for Φ_ref...")
    
    t0 = time.time()
    Phi_ref = fft_poisson_solve(rho, params.kappa)
    results['Phi_ref'] = Phi_ref
    if verbose:
        print(f"      Done in {time.time()-t0:.3f}s")
        print(f"      Φ_ref range: [{np.min(Phi_ref):.4f}, {np.max(Phi_ref):.4f}]")
    
    # Radial profiles
    _, Phi_ref_mean, _ = compute_radial_profile(Phi_ref, center, n_bins=60)
    _, q_mean, _ = compute_radial_profile(q, center, n_bins=60)
    _, b_mean, _ = compute_radial_profile(b, center, n_bins=60)
    _, rho_mean, _ = compute_radial_profile(rho, center, n_bins=60)
    
    results['Phi_ref_mean'] = Phi_ref_mean
    results['q_mean'] = q_mean
    results['b_mean'] = b_mean
    results['rho_mean'] = rho_mean
    
    # Fit baseline-referenced potential
    fit_ref = fit_newtonian_potential(r_bins, Phi_ref_mean, 
                                       params.r_min_fit, params.r_max_fit)
    results['fit_ref'] = fit_ref
    
    fit_screened = fit_screened_potential(r_bins, Phi_ref_mean, 
                                          params.r_min_fit, params.r_max_fit)
    results['fit_screened'] = fit_screened
    
    if verbose:
        print(f"\n      BASELINE-REFERENCED GRAVITY FIT (Φ = -A/r + B):")
        if fit_ref['success']:
            print(f"        A = {fit_ref['A']:.4f} ± {fit_ref['A_err']:.4f}")
            print(f"        R² = {fit_ref['r_squared']:.6f}")
        
        print(f"\n      SCREENED POTENTIAL FIT (Φ = -A*exp(-μr)/r + B):")
        if fit_screened['success']:
            print(f"        A = {fit_screened['A']:.4f}")
            print(f"        μ = {fit_screened['mu']:.4f} (screening length = {fit_screened['screening_length']:.2f})")
            print(f"        R² = {fit_screened['r_squared']:.6f}")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    if verbose:
        print(f"\n[5/5] Computing summary...")
    
    # Success criteria:
    # 1. Fit succeeded
    # 2. |A| > 0 (non-zero monopole moment)
    # 3. R² > 0.95 (good 1/r fit)
    # Note: A > 0 for repulsive (raw Poisson), A < 0 for attractive (DET gravity)
    raw_success = (fit_raw['success'] and 
                   abs(fit_raw['A']) > 0.1 and 
                   fit_raw['r_squared'] > 0.95)
    results['raw_success'] = raw_success
    results['success'] = raw_success
    
    if verbose:
        print(f"\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"\n  TEST A (Raw q-sourced gravity):")
        print(f"    1/r kernel verified: {'PASS' if raw_success else 'FAIL'}")
        
        print(f"\n  TEST B (Baseline-referenced gravity):")
        print(f"    Total source Σρ ≈ 0: {'YES' if abs(results['total_rho']) < 0.1 else 'NO'}")
        if fit_screened['success'] and fit_screened['r_squared'] > 0.9:
            print(f"    Screened potential fit: GOOD (R² = {fit_screened['r_squared']:.4f})")
            print(f"    Effective screening length: {fit_screened['screening_length']:.2f} lattice units")
    
    return results


def plot_results(results: Dict, save_path: Optional[str] = None):
    """Generate visualization."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    r = results['r_bins']
    
    # Plot 1: Source profiles
    ax1 = axes[0, 0]
    ax1.plot(r, results['q_mean'], 'b-', linewidth=2, label='q (structural debt)')
    ax1.plot(r, results['b_mean'], 'g--', linewidth=2, label='b (baseline)')
    ax1.plot(r, results['rho_mean'], 'r-', linewidth=2, label='ρ = q - b')
    ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Radius r (lattice units)', fontsize=12)
    ax1.set_ylabel('Field value', fontsize=12)
    ax1.set_title('Source Fields: q, b, ρ', fontsize=14)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Raw potential with 1/r fit
    ax2 = axes[0, 1]
    ax2.plot(r, results['Phi_raw_mean'], 'ko-', markersize=3, alpha=0.7, label='Φ_raw(r)')
    
    fit = results['fit_raw']
    if fit['success']:
        r_fit = np.linspace(fit['r_range'][0], fit['r_range'][1], 100)
        Phi_fit = fit['A'] / r_fit + fit['B']
        ax2.plot(r_fit, Phi_fit, 'r-', linewidth=2, 
                 label=f"Fit: {fit['A']:.2f}/r + {fit['B']:.2f}\nR² = {fit['r_squared']:.4f}")
        ax2.axvline(fit['r_range'][0], color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(fit['r_range'][1], color='gray', linestyle=':', alpha=0.5)
    
    ax2.set_xlabel('Radius r', fontsize=12)
    ax2.set_ylabel('Potential Φ_raw', fontsize=12)
    ax2.set_title('TEST A: Raw q-Sourced Potential', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Baseline-referenced potential
    ax3 = axes[0, 2]
    ax3.plot(r, results['Phi_ref_mean'], 'ko-', markersize=3, alpha=0.7, label='Φ_ref(r)')
    
    fit_s = results['fit_screened']
    if fit_s['success']:
        r_fit = np.linspace(3, 30, 100)
        Phi_fit = -fit_s['A'] * np.exp(-fit_s['mu'] * r_fit) / r_fit + fit_s['B']
        ax3.plot(r_fit, Phi_fit, 'r-', linewidth=2, 
                 label=f"Screened: λ={1/fit_s['mu']:.1f}\nR² = {fit_s['r_squared']:.4f}")
    
    ax3.set_xlabel('Radius r', fontsize=12)
    ax3.set_ylabel('Potential Φ_ref', fontsize=12)
    ax3.set_title('TEST B: Baseline-Referenced Potential', fontsize=14)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Φ_raw × r
    ax4 = axes[1, 0]
    valid = r > 2
    Phi_times_r = results['Phi_raw_mean'][valid] * r[valid]
    ax4.plot(r[valid], Phi_times_r, 'bo-', markersize=4, alpha=0.7)
    
    if results['fit_raw']['success']:
        A = results['fit_raw']['A']
        B = results['fit_raw']['B']
        ax4.axhline(A, color='green', linestyle='--', alpha=0.7, label=f'A = {A:.2f}')
    
    ax4.set_xlabel('Radius r', fontsize=12)
    ax4.set_ylabel('Φ_raw × r', fontsize=12)
    ax4.set_title('Φ_raw × r (constant → 1/r verified)', fontsize=14)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Log-log
    ax5 = axes[1, 1]
    if results['fit_raw']['success']:
        B = results['fit_raw']['B']
        valid_log = (r > 5) & (np.abs(results['Phi_raw_mean'] - B) > 1e-6)
        r_log = r[valid_log]
        Phi_shifted = np.abs(results['Phi_raw_mean'][valid_log] - B)
        Phi_shifted = np.maximum(Phi_shifted, 1e-10)  # Avoid log(0)
        
        ax5.loglog(r_log, Phi_shifted, 'bo', markersize=5, alpha=0.7, label='|Φ_raw - B|')
        
        A = results['fit_raw']['A']
        r_ref = np.logspace(np.log10(5), np.log10(28), 50)
        ref_line = np.abs(A) / r_ref
        ax5.loglog(r_ref, ref_line, 'r-', linewidth=2, label='1/r reference')
        
        ax5.set_xlabel('Radius r', fontsize=12)
        ax5.set_ylabel('|Φ_raw - B|', fontsize=12)
        ax5.set_title('Log-Log (slope = -1 for 1/r)', fontsize=14)
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3, which='both')
    
    # Plot 6: Comparison
    ax6 = axes[1, 2]
    ax6.plot(r, results['Phi_raw_mean'], 'b-', linewidth=2, label='Φ_raw (q-sourced)')
    ax6.plot(r, results['Phi_ref_mean'], 'r-', linewidth=2, label='Φ_ref (ρ-sourced)')
    ax6.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Radius r', fontsize=12)
    ax6.set_ylabel('Potential Φ', fontsize=12)
    ax6.set_title('Comparison: Raw vs Baseline-Referenced', fontsize=14)
    ax6.legend(fontsize=11)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    status = "PASS" if results['success'] else "FAIL"
    fig.suptitle(f'DET Test 3.1: Newtonian Kernel - {status}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.close()
    return fig


def run_scaling_test(verbose: bool = True) -> Dict:
    """Test that monopole A scales linearly with total source Σq."""
    if verbose:
        print("\n" + "="*70)
        print("SCALING TEST: A vs Σq")
        print("="*70)
    
    amplitudes = [0.2, 0.4, 0.6, 0.8, 1.0]
    results_list = []
    
    for amp in amplitudes:
        params = GravityTestParams(N=48, q_amplitude=amp, r_min_fit=6, r_max_fit=20)
        result = run_test(params, verbose=False)
        
        if result['fit_raw']['success']:
            results_list.append({
                'amplitude': amp,
                'total_q': result['total_q'],
                'A': result['fit_raw']['A'],
                'R_squared': result['fit_raw']['r_squared']
            })
            if verbose:
                print(f"  q_amp={amp:.1f}: Σq={result['total_q']:.2f}, A={result['fit_raw']['A']:.4f}, R²={result['fit_raw']['r_squared']:.4f}")
    
    if len(results_list) >= 3:
        qs = np.array([r['total_q'] for r in results_list])
        As = np.array([r['A'] for r in results_list])
        
        coeffs = np.polyfit(qs, As, 1)
        m, c = coeffs
        
        A_pred = m * qs + c
        ss_res = np.sum((As - A_pred)**2)
        ss_tot = np.sum((As - np.mean(As))**2)
        r_squared_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        if verbose:
            print(f"\n  Linear fit: A = {m:.6f} × Σq + {c:.4f}")
            print(f"  R² (linear) = {r_squared_linear:.6f}")
            print(f"  Scaling {'VERIFIED' if r_squared_linear > 0.99 else 'NOT VERIFIED'}")
        
        return {
            'results': results_list,
            'slope': m,
            'intercept': c,
            'r_squared': r_squared_linear,
            'scaling_verified': r_squared_linear > 0.99
        }
    
    return {'results': results_list, 'scaling_verified': False}


if __name__ == "__main__":
    # Run main test
    params = GravityTestParams(N=64)
    results = run_test(params, verbose=True)
    
    # Generate plots
    plot_results(results, save_path='/home/ubuntu/det_v6_release/results/newtonian_kernel_fft.png')
    
    # Run scaling test
    scaling = run_scaling_test(verbose=True)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"  Raw 1/r potential test: {'PASS' if results['success'] else 'FAIL'}")
    print(f"  A ∝ Σq scaling test: {'PASS' if scaling.get('scaling_verified', False) else 'FAIL'}")
    
    if results['success'] and scaling.get('scaling_verified', False):
        print(f"\n  ✓ NEWTONIAN KERNEL VERIFIED")
        print(f"    The DET Poisson solver correctly produces 1/r far-field potential")
        print(f"    Monopole moment scales linearly with total source")
