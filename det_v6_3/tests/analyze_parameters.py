"""
DET Parameter Analysis: Symmetries, Patterns, and Unification
==============================================================

Analyze the DET parameter space for:
1. Equal or nearly-equal values
2. Simple ratios (1/2, 2, golden ratio, etc.)
3. Derived parameters (one depends on another)
4. Candidates for unification
"""

import numpy as np
from collections import defaultdict

# Current DET parameters (from det_v6_3_3d_collider.py)
params = {
    # Momentum
    'alpha_pi': 0.12,
    'lambda_pi': 0.008,
    'mu_pi': 0.35,
    'pi_max': 3.0,

    # Angular Momentum
    'alpha_L': 0.06,
    'lambda_L': 0.005,
    'mu_L': 0.18,
    'L_max': 5.0,

    # Floor
    'eta_floor': 0.12,
    'F_core': 5.0,
    'floor_power': 2.0,

    # Structure/Agency
    'alpha_q': 0.012,
    'lambda_a': 30.0,
    'beta_a': 0.2,
    'gamma_a_max': 0.15,
    'gamma_a_power': 2.0,

    # Coherence
    'alpha_C': 0.04,
    'lambda_C': 0.002,
    'C_init': 0.15,

    # Gravity
    'alpha_grav': 0.02,
    'kappa_grav': 5.0,
    'mu_grav': 2.0,
    'beta_g': 10.0,

    # Boundary
    'F_MIN_grace': 0.05,
    'eta_heal': 0.03,

    # Lattice
    'eta_lattice': 0.965,

    # Numerical
    'DT': 0.02,
    'outflow_limit': 0.2,
}

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618

def analyze_ratios():
    """Find simple ratios between parameters."""
    print("="*70)
    print("RATIO ANALYSIS")
    print("="*70)

    # Check for simple ratios
    simple_ratios = [0.5, 1.0, 2.0, PHI, 1/PHI, 5.0, 10.0, 0.1]
    ratio_names = ['1/2', '1', '2', 'φ', '1/φ', '5', '10', '1/10']

    found_ratios = []

    names = list(params.keys())
    for i, name1 in enumerate(names):
        for name2 in names[i+1:]:
            v1, v2 = params[name1], params[name2]
            if v1 == 0 or v2 == 0:
                continue

            ratio = v1 / v2
            inv_ratio = v2 / v1

            for r, rname in zip(simple_ratios, ratio_names):
                if abs(ratio - r) < 0.01:
                    found_ratios.append((name1, name2, ratio, rname))
                elif abs(inv_ratio - r) < 0.01:
                    found_ratios.append((name2, name1, inv_ratio, rname))

    print("\nSimple Ratios Found:")
    print("-"*60)
    for n1, n2, ratio, rname in sorted(found_ratios, key=lambda x: x[3]):
        print(f"  {n1} / {n2} = {ratio:.4f} ≈ {rname}")


def analyze_equal_values():
    """Find equal or nearly-equal parameter values."""
    print("\n" + "="*70)
    print("EQUAL VALUE ANALYSIS")
    print("="*70)

    # Group by value
    by_value = defaultdict(list)
    for name, val in params.items():
        # Round to avoid floating point issues
        key = round(val, 4)
        by_value[key].append(name)

    print("\nParameters with identical values:")
    print("-"*60)
    for val, names in sorted(by_value.items()):
        if len(names) > 1:
            print(f"  {val}: {', '.join(names)}")

    # Find nearly equal (within 10%)
    print("\nParameters within 10% of each other:")
    print("-"*60)
    names = list(params.keys())
    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            v1, v2 = params[n1], params[n2]
            if v1 == 0 or v2 == 0:
                continue
            ratio = max(v1, v2) / min(v1, v2)
            if ratio < 1.1 and v1 != v2:
                print(f"  {n1}={v1} ≈ {n2}={v2} (ratio: {ratio:.3f})")


def analyze_derived_parameters():
    """Identify parameters that appear to be derived from others."""
    print("\n" + "="*70)
    print("DERIVED PARAMETER ANALYSIS")
    print("="*70)

    print("\nPotential derivations:")
    print("-"*60)

    # Check explicit relationships
    derivations = [
        ('beta_g', 'mu_grav', params['beta_g'] / params['mu_grav'], 'beta_g = 5 * mu_grav'),
        ('alpha_L', 'alpha_pi', params['alpha_L'] / params['alpha_pi'], 'alpha_L = alpha_pi / 2'),
        ('eta_floor', 'alpha_pi', params['eta_floor'] / params['alpha_pi'], 'eta_floor = alpha_pi'),
        ('gamma_a_max', 'C_init', params['gamma_a_max'] / params['C_init'], 'gamma_a_max = C_init'),
        ('beta_a', 'alpha_grav', params['beta_a'] / params['alpha_grav'], 'beta_a = 10 * alpha_grav'),
        ('alpha_q', 'alpha_pi', params['alpha_q'] / params['alpha_pi'], 'alpha_q = alpha_pi / 10'),
    ]

    for p1, p2, ratio, desc in derivations:
        print(f"  {desc}")
        print(f"    Actual: {p1}={params[p1]}, {p2}={params[p2]}, ratio={ratio:.4f}")


def analyze_module_symmetry():
    """Check for symmetry between momentum and angular momentum modules."""
    print("\n" + "="*70)
    print("MODULE SYMMETRY ANALYSIS")
    print("="*70)

    print("\nMomentum vs Angular Momentum:")
    print("-"*60)

    pairs = [
        ('alpha_pi', 'alpha_L'),
        ('lambda_pi', 'lambda_L'),
        ('mu_pi', 'mu_L'),
        ('pi_max', 'L_max'),
    ]

    for p_mom, p_ang in pairs:
        v_mom = params[p_mom]
        v_ang = params[p_ang]
        ratio = v_mom / v_ang if v_ang != 0 else float('inf')
        print(f"  {p_mom}/{p_ang} = {v_mom}/{v_ang} = {ratio:.4f}")

    print("\nCharging/Decay Ratios:")
    print("-"*60)
    modules = [
        ('Momentum', 'alpha_pi', 'lambda_pi'),
        ('Angular', 'alpha_L', 'lambda_L'),
        ('Coherence', 'alpha_C', 'lambda_C'),
    ]
    for name, alpha, lam in modules:
        ratio = params[alpha] / params[lam]
        print(f"  {name}: {alpha}/{lam} = {params[alpha]}/{params[lam]} = {ratio:.1f}")


def propose_unification():
    """Propose parameter unification based on analysis."""
    print("\n" + "="*70)
    print("PROPOSED UNIFICATION")
    print("="*70)

    print("""
Based on the analysis, here are proposed unifications:

1. MOMENTUM/ANGULAR MOMENTUM SYMMETRY
   Current: 6 independent parameters (α_π, λ_π, μ_π, α_L, λ_L, μ_L)
   Proposal: 3 base parameters + 1 scaling factor

   α_L = α_π / 2
   λ_L = λ_π * 0.625  (or λ_π / φ ?)
   μ_L = μ_π / 2

   New params: α_π, λ_π, μ_π, ρ_L (angular scaling ≈ 0.5)
   Reduction: 6 → 4

2. FLOOR-MOMENTUM COUPLING
   Current: η_floor = 0.12 = α_π (coincidence?)
   Proposal: η_floor = α_π (derived)
   Reduction: 1 parameter

3. COHERENCE-AGENCY COUPLING
   Current: γ_a_max = 0.15 = C_init (coincidence?)
   Proposal: γ_a_max = C_init (natural: max drive at initial coherence)
   Reduction: 1 parameter

4. GRAVITY-AGENCY COUPLING
   Current: β_a = 0.2 = 10 * α_grav (0.02)
   Proposal: β_a = 10 * α_grav (agency rate tied to gravity screening)
   Reduction: 1 parameter (if relationship is fundamental)

5. CHARGING/DECAY UNIVERSALITY
   Current: α/λ ratios are 15, 12, 20 for π, L, C
   Proposal: Universal τ_eq = α/λ ≈ 15
   New: α_C = 15 * λ_C, etc.

TOTAL POTENTIAL REDUCTION: 6+ parameters

CURRENT COUNT: ~25 physical parameters
PROPOSED COUNT: ~19 physical parameters
""")


def check_golden_ratio():
    """Check for golden ratio relationships."""
    print("\n" + "="*70)
    print("GOLDEN RATIO (φ ≈ 1.618) ANALYSIS")
    print("="*70)

    print(f"\nφ = {PHI:.6f}, 1/φ = {1/PHI:.6f}, φ² = {PHI**2:.6f}")
    print()

    names = list(params.keys())
    golden_found = []

    for i, n1 in enumerate(names):
        for n2 in names[i+1:]:
            v1, v2 = params[n1], params[n2]
            if v1 == 0 or v2 == 0:
                continue
            ratio = max(v1, v2) / min(v1, v2)
            if abs(ratio - PHI) < 0.05:
                golden_found.append((n1, n2, ratio))
            elif abs(ratio - PHI**2) < 0.1:
                golden_found.append((n1, n2, ratio, 'φ²'))

    if golden_found:
        print("Golden ratio relationships found:")
        for item in golden_found:
            if len(item) == 3:
                n1, n2, ratio = item
                print(f"  {n1}/{n2} = {ratio:.4f} ≈ φ")
            else:
                n1, n2, ratio, note = item
                print(f"  {n1}/{n2} = {ratio:.4f} ≈ {note}")
    else:
        print("No obvious golden ratio relationships found.")

    # Check specific ratios
    print("\nSpecific ratio checks:")
    print(f"  lambda_pi / lambda_L = {params['lambda_pi']/params['lambda_L']:.4f} (φ = {PHI:.4f})")
    print(f"  mu_pi / mu_L = {params['mu_pi']/params['mu_L']:.4f}")
    print(f"  L_max / pi_max = {params['L_max']/params['pi_max']:.4f} ≈ φ?")


if __name__ == "__main__":
    analyze_ratios()
    analyze_equal_values()
    analyze_derived_parameters()
    analyze_module_symmetry()
    check_golden_ratio()
    propose_unification()
