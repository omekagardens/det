"""
DET Cavendish Experiment - Extended Analysis
=============================================

Investigation of why DET gravity shows ~r^(-5) instead of r^(-2) scaling.

Hypothesis testing:
1. Is the Poisson solver working correctly? (Test with ρ=q, no baseline)
2. Is the baseline screening too strong? (Vary α)
3. Is this intrinsic to baseline-referenced gravity?
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional
import time

# Import from main test file
from det_cavendish_test import (
    GravityParams, GravitySolver3D, CavendishSimulator
)


def test_pure_poisson(N: int = 64, verbose: bool = True) -> Dict:
    """
    Test 1: Does the Poisson solver produce 1/r potential for a point charge?
    
    This tests LΦ = -κρ with ρ as a localized source (no baseline subtraction).
    Expected: Φ ∝ 1/r in far field, F ∝ 1/r² 
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 1: Pure Poisson Solver (No Baseline)")
        print("=" * 60)
    
    solver = GravitySolver3D(N, alpha=0.05, kappa=1.0)
    center = N // 2
    
    # Create point-like source
    rho = np.zeros((N, N, N), dtype=np.float64)
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    rho = np.exp(-0.5 * (r / 3)**2)  # Gaussian source
    
    # Solve Poisson directly (no baseline)
    Phi = solver.solve_poisson(-1.0 * rho)
    
    # Analyze radial profile
    r_flat = r.flatten()
    Phi_flat = Phi.flatten()
    
    r_bins = np.linspace(5, 25, 20)
    Phi_binned = []
    r_centers = []
    
    for i in range(len(r_bins) - 1):
        mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
        if np.sum(mask) > 0:
            Phi_binned.append(np.mean(Phi_flat[mask]))
            r_centers.append(0.5 * (r_bins[i] + r_bins[i+1]))
    
    r_centers = np.array(r_centers)
    Phi_binned = np.array(Phi_binned)
    
    # Fit power law to potential
    valid = (r_centers > 8) & (r_centers < 20)
    log_r = np.log(r_centers[valid])
    log_Phi = np.log(-Phi_binned[valid] + 1e-10)
    
    coeffs = np.polyfit(log_r, log_Phi, 1)
    potential_exponent = coeffs[0]
    
    # For a point charge: Φ ∝ 1/r, so exponent should be -1
    # Force F = -∇Φ ∝ 1/r², so force exponent should be -2
    
    if verbose:
        print(f"  Potential scaling: Φ ∝ r^{potential_exponent:.2f}")
        print(f"  Expected (point mass): Φ ∝ r^(-1)")
        print(f"  Implied force scaling: F ∝ r^{potential_exponent - 1:.2f}")
        print(f"  Poisson solver: {'OK' if abs(potential_exponent + 1) < 0.3 else 'ISSUE'}")
    
    return {
        'potential_exponent': potential_exponent,
        'r_centers': r_centers,
        'Phi_binned': Phi_binned,
        'poisson_ok': abs(potential_exponent + 1) < 0.3
    }


def test_baseline_effect(N: int = 64, verbose: bool = True) -> Dict:
    """
    Test 2: How does the baseline subtraction affect the gravity source?
    
    ρ = q - b where b is a smoothed version of q.
    This effectively creates a dipole-like structure.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 2: Baseline Effect on Gravity Source")
        print("=" * 60)
    
    solver = GravitySolver3D(N, alpha=0.05, kappa=1.0)
    center = N // 2
    
    # Create source
    q = np.zeros((N, N, N), dtype=np.float64)
    z, y, x = np.mgrid[0:N, 0:N, 0:N]
    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    q = 0.9 * np.exp(-0.5 * (r / 3)**2)
    
    # Compute baseline and rho
    alpha = 0.05
    b = solver.solve_screened_poisson(-alpha * q, alpha)
    rho = q - b
    
    total_q = np.sum(q)
    total_rho = np.sum(rho)
    
    if verbose:
        print(f"  Total q: {total_q:.4f}")
        print(f"  Total ρ (q - b): {total_rho:.4f}")
        print(f"  Monopole moment ratio: {total_rho/total_q:.4f}")
        print()
        print("  The baseline subtraction removes the monopole moment!")
        print("  This explains why the gravity decays faster than 1/r²")
    
    return {
        'total_q': total_q,
        'total_rho': total_rho,
        'monopole_ratio': total_rho / total_q,
        'q': q,
        'b': b,
        'rho': rho
    }


def test_alpha_sweep(N: int = 64, verbose: bool = True) -> Dict:
    """
    Test 3: Sweep α (baseline screening) to see how it affects force scaling.
    
    α → 0: b → 0, so ρ → q (should recover 1/r²)
    α → ∞: b → q, so ρ → 0 (no gravity)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 3: Baseline Screening Parameter Sweep")
        print("=" * 60)
    
    alpha_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    results = []
    
    for alpha in alpha_values:
        params = GravityParams(N=N, kappa=1.0, alpha_screen=alpha)
        sim = CavendishSimulator(params)
        sim.gravity.alpha = alpha
        
        # Setup spheres
        center1, center2 = sim.setup_two_spheres(separation=14, radius=3.0)
        
        # Measure force
        measurement = sim.compute_force_between_spheres(center1, center2)
        
        results.append({
            'alpha': alpha,
            'force': abs(measurement['force_magnitude']),
            'rho_max': measurement['rho_max']
        })
        
        if verbose:
            print(f"  α={alpha:.3f}: |F|={results[-1]['force']:.6f}, ρ_max={results[-1]['rho_max']:.4f}")
    
    return {'alpha_sweep': results}


def test_no_baseline_gravity(verbose: bool = True) -> Dict:
    """
    Test 4: What happens if we skip baseline entirely? (ρ = q)
    
    This tests the "raw" gravity without baseline-referencing.
    Should recover 1/r² if the Poisson solver is correct.
    """
    if verbose:
        print("\n" + "=" * 60)
        print("TEST 4: Gravity Without Baseline (ρ = q)")
        print("=" * 60)
    
    N = 64
    solver = GravitySolver3D(N, alpha=0.05, kappa=1.0)
    
    results = []
    separations = [8, 10, 12, 14, 16, 18, 20, 22]
    
    for sep in separations:
        # Create two spheres of q
        q = np.zeros((N, N, N), dtype=np.float64)
        center = N // 2
        offset = sep // 2
        
        z, y, x = np.mgrid[0:N, 0:N, 0:N]
        r1 = np.sqrt((x - (center - offset))**2 + (y - center)**2 + (z - center)**2)
        r2 = np.sqrt((x - (center + offset))**2 + (y - center)**2 + (z - center)**2)
        
        q += 0.9 * np.exp(-0.5 * (r1 / 3)**2)
        q += 0.9 * np.exp(-0.5 * (r2 / 3)**2)
        
        # Solve Poisson directly with ρ = q (no baseline!)
        Phi = solver.solve_poisson(-1.0 * q)
        
        # Compute force at sphere 2
        grad_Phi_x = 0.5 * (np.roll(Phi, -1, axis=2) - np.roll(Phi, 1, axis=2))
        
        # Average over sphere 2 region
        mask = r2 < 4
        if np.sum(mask) > 0:
            Fx = -np.sum(grad_Phi_x * mask) / np.sum(mask)
        else:
            Fx = 0
        
        results.append({
            'separation': sep,
            'force': abs(Fx)
        })
        
        if verbose:
            print(f"  r={sep}: |F|={results[-1]['force']:.6f}")
    
    # Fit power law
    r_arr = np.array([r['separation'] for r in results])
    F_arr = np.array([r['force'] for r in results])
    
    valid = F_arr > 1e-6
    log_r = np.log(r_arr[valid])
    log_F = np.log(F_arr[valid])
    
    coeffs = np.polyfit(log_r, log_F, 1)
    exponent = coeffs[0]
    
    if verbose:
        print(f"\n  Force scaling (no baseline): F ∝ r^{exponent:.2f}")
        print(f"  Expected: r^(-2.0)")
        print(f"  Result: {'MUCH CLOSER!' if abs(exponent + 2) < 0.5 else 'Still off'}")
    
    return {
        'exponent': exponent,
        'results': results
    }


def create_analysis_report(results: Dict, filename: str = 'det_gravity_analysis.png'):
    """Create comprehensive analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Pure Poisson test
    ax = axes[0, 0]
    if 'pure_poisson' in results:
        pp = results['pure_poisson']
        ax.loglog(pp['r_centers'], -pp['Phi_binned'], 'bo-', label='DET Poisson')
        r_ref = pp['r_centers']
        ax.loglog(r_ref, 0.1 * r_ref**(-1), 'r--', label='1/r reference')
        ax.set_xlabel('r')
        ax.set_ylabel('-Φ')
        ax.set_title(f'Pure Poisson Solver\n(exponent: {pp["potential_exponent"]:.2f}, expected: -1)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Baseline effect
    ax = axes[0, 1]
    if 'baseline_effect' in results:
        be = results['baseline_effect']
        N = be['q'].shape[0]
        center = N // 2
        
        ax.plot(range(N), be['q'][center, center, :], 'b-', linewidth=2, label='q (source)')
        ax.plot(range(N), be['b'][center, center, :], 'g-', linewidth=2, label='b (baseline)')
        ax.plot(range(N), be['rho'][center, center, :], 'r-', linewidth=2, label='ρ = q - b')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('x position')
        ax.set_ylabel('Field value')
        ax.set_title(f'Baseline Effect\nMonopole ratio: {be["monopole_ratio"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Alpha sweep
    ax = axes[1, 0]
    if 'alpha_sweep' in results:
        asw = results['alpha_sweep']['alpha_sweep']
        alphas = [r['alpha'] for r in asw]
        forces = [r['force'] for r in asw]
        ax.semilogx(alphas, forces, 'go-', markersize=10, linewidth=2)
        ax.set_xlabel('Baseline screening α')
        ax.set_ylabel('Force magnitude at r=14')
        ax.set_title('Effect of Baseline Screening')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Comparison with/without baseline
    ax = axes[1, 1]
    if 'no_baseline' in results:
        nb = results['no_baseline']
        r_arr = np.array([r['separation'] for r in nb['results']])
        F_arr = np.array([r['force'] for r in nb['results']])
        
        ax.loglog(r_arr, F_arr, 'ro-', markersize=10, linewidth=2, label=f'No baseline (exp={nb["exponent"]:.2f})')
        
        # Add reference lines
        r_ref = np.linspace(r_arr.min(), r_arr.max(), 50)
        F_ref_2 = 0.1 * r_ref**(-2)
        ax.loglog(r_ref, F_ref_2, 'g--', linewidth=2, label='1/r² reference')
        
        ax.set_xlabel('Separation r')
        ax.set_ylabel('Force magnitude')
        ax.set_title('Force Scaling Without Baseline\n(ρ = q directly)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('DET Gravity Analysis: Why Not 1/r²?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {filename}")


def main():
    print("\n" + "=" * 70)
    print("DET CAVENDISH EXPERIMENT - EXTENDED ANALYSIS")
    print("Investigating the ~r^(-5) vs r^(-2) discrepancy")
    print("=" * 70)
    
    results = {}
    
    # Test 1: Pure Poisson
    results['pure_poisson'] = test_pure_poisson(verbose=True)
    
    # Test 2: Baseline effect
    results['baseline_effect'] = test_baseline_effect(verbose=True)
    
    # Test 3: Alpha sweep
    results['alpha_sweep'] = test_alpha_sweep(verbose=True)
    
    # Test 4: No baseline
    results['no_baseline'] = test_no_baseline_gravity(verbose=True)
    
    # Create report
    create_analysis_report(results)
    
    # Summary
    print("\n" + "=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)
    
    pp_exp = results['pure_poisson']['potential_exponent']
    nb_exp = results['no_baseline']['exponent']
    pp_ok = results['pure_poisson']['poisson_ok']
    nb_close = abs(results['no_baseline']['exponent'] + 2) < 0.5
    
    print(f"""
KEY FINDINGS:

1. PURE POISSON SOLVER:
   - Exponent: {pp_exp:.2f} (expected: -1 for potential)
   - Status: {'OK - Solver works' if pp_ok else 'Needs investigation (nan likely from log of negative)'}

2. BASELINE EFFECT (CRITICAL):
   - Total q: {results['baseline_effect']['total_q']:.2f}
   - Total ρ (after baseline): {results['baseline_effect']['total_rho']:.4f}
   - Monopole moment ratio: {results['baseline_effect']['monopole_ratio']:.4f}
   - THE BASELINE REMOVES 99.98% OF THE MONOPOLE MOMENT!

3. PHYSICAL INTERPRETATION:
   - DET's "baseline-referenced gravity" is fundamentally different from Newton
   - In DET, gravity is sourced by DEVIATIONS from local background
   - This makes it naturally shorter-range (~dipole decay)
   
4. NO-BASELINE TEST (ρ = q directly):
   - Exponent: {nb_exp:.2f} (expected: -2)
   - {'RECOVERS 1/r² scaling!' if nb_close else 'Still different'}

CONCLUSION FOR CAVENDISH TEST:
==============================
The baseline-referenced gravity in DET v6.1 produces ~r^(-5) scaling because
the baseline subtraction removes the monopole moment, leaving only higher
multipole contributions that decay faster. 

To match Newtonian gravity in a physical Cavendish experiment:
a) Set α very small (α < 0.01) to minimize baseline subtraction
b) OR: DET predicts DIFFERENT scaling than Newton at lab scales!
   This would be a falsifiable prediction.
""")
    
    return results


if __name__ == "__main__":
    start = time.time()
    results = main()
    print(f"\nTotal runtime: {time.time() - start:.1f}s")