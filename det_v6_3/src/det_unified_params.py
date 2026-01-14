"""
DET Unified Parameter Schema v1.0
==================================

Reduces ~25 physical parameters to ~12 base parameters by recognizing:
1. Equal value clusters (coincidences → constraints)
2. Golden ratio (φ) in momentum/angular ratios
3. Powers of 2, 5, 10 linking modules

Base Parameters (12):
- τ_base: Time/screening scale
- σ_base: Charging rate scale
- λ_base: Decay rate scale
- μ_base: Mobility scale
- κ_base: Coupling scale
- C_0: Coherence scale
- φ_L: Angular/momentum ratio (default: 1/2)
- λ_a: Agency ceiling coupling
- τ_eq: Equilibration time ratio (α/λ)
- p_floor: Floor exponent
- p_agency: Agency coherence exponent
- π_max: Momentum cap

Derived Parameters (computed automatically):
- α_π, λ_π, μ_π from base scales
- α_L, λ_L, μ_L from momentum × φ_L
- η_floor = σ_base
- γ_a_max = C_0
- β_a = 10 * τ_base
- α_q = σ_base / 10
- β_g = 5 * μ_base
- etc.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2  # ≈ 1.618


@dataclass
class DETUnifiedParams:
    """
    Unified DET parameter schema with automatic derivation.

    Only specify base parameters; derived parameters are computed automatically.
    """

    # === GRID ===
    N: int = 64
    DT: float = 0.02

    # === BASE SCALES ===
    # These are the fundamental independent parameters

    # Time/screening scale (τ_base)
    tau_base: float = 0.02  # = α_grav = DT

    # Charging rate scale (σ_base)
    sigma_base: float = 0.12  # = α_π = η_floor

    # Decay rate scale (λ_base)
    lambda_base: float = 0.008  # = λ_π

    # Mobility scale (μ_base)
    mu_base: float = 2.0  # = μ_grav = floor_power = γ_a_power

    # Coupling scale (κ_base)
    kappa_base: float = 5.0  # = κ_grav = F_core = L_max

    # Coherence scale
    C_0: float = 0.15  # = C_init = γ_a_max

    # === RATIO PARAMETERS ===

    # Angular/momentum ratio (applies to α, λ, μ)
    phi_L: float = 0.5  # Angular params = momentum params × φ_L

    # Golden ratio for decay (λ_L = λ_π / φ)
    use_golden_decay: bool = False  # If True, λ_L = λ_π / φ instead of λ_π * φ_L

    # Lambda_L factor (used when not using golden decay)
    lambda_L_factor: float = 0.625  # λ_L = λ_π × 0.625 = 0.005

    # Mu_L factor (allows slight tuning from φ_L)
    mu_L_factor: float = 0.514  # μ_L = μ_π × 0.514 ≈ 0.18

    # Momentum mobility factor (μ_π relative to μ_base)
    mu_pi_factor: float = 0.175  # μ_π = mu_base × mu_pi_factor = 2.0 × 0.175 = 0.35

    # === MODULE-SPECIFIC TUNING ===

    # Agency
    lambda_a: float = 30.0  # Structural ceiling coupling (independent)

    # Momentum caps
    pi_max: float = 3.0
    L_max_factor: float = 1.0  # L_max = κ_base × L_max_factor = 5.0

    # Coherence dynamics
    tau_eq_C: float = 20.0  # α_C / λ_C ratio

    # Floor
    F_core_factor: float = 1.0  # F_core = κ_base × F_core_factor

    # Lattice correction
    eta_lattice: float = 0.965

    # Numerical
    outflow_limit: float = 0.2

    # === FEATURE TOGGLES ===
    diff_enabled: bool = True
    momentum_enabled: bool = True
    angular_momentum_enabled: bool = True
    floor_enabled: bool = True
    q_enabled: bool = True
    agency_dynamic: bool = True
    sigma_dynamic: bool = True
    coherence_dynamic: bool = True
    gravity_enabled: bool = True
    boundary_enabled: bool = True
    grace_enabled: bool = True
    healing_enabled: bool = False
    coherence_weighted_H: bool = False

    # === DERIVED PARAMETERS (computed) ===

    def __post_init__(self):
        """Compute all derived parameters."""
        self._compute_derived()

    def _compute_derived(self):
        """Compute derived parameters from base scales."""

        # === MOMENTUM ===
        self.alpha_pi = self.sigma_base  # 0.12
        self.lambda_pi = self.lambda_base  # 0.008
        self.mu_pi = self.mu_base * self.mu_pi_factor  # 2.0 * 0.175 = 0.35

        # === ANGULAR MOMENTUM ===
        self.alpha_L = self.alpha_pi * self.phi_L  # 0.12 * 0.5 = 0.06

        if self.use_golden_decay:
            self.lambda_L = self.lambda_pi / PHI  # 0.008 / 1.618 ≈ 0.00494
        else:
            self.lambda_L = self.lambda_pi * self.lambda_L_factor  # 0.008 * 0.625 = 0.005

        self.mu_L = self.mu_pi * self.mu_L_factor  # 0.35 * 0.514 ≈ 0.18
        self.L_max = self.kappa_base * self.L_max_factor  # 5.0

        # === FLOOR ===
        self.eta_floor = self.sigma_base  # 0.12 (= α_π)
        self.F_core = self.kappa_base * self.F_core_factor  # 5.0
        self.floor_power = self.mu_base  # 2.0

        # === STRUCTURE ===
        self.alpha_q = self.sigma_base / 10.0  # 0.012

        # === AGENCY ===
        self.beta_a = 10.0 * self.tau_base  # 10 * 0.02 = 0.2
        self.gamma_a_max = self.C_0  # 0.15
        self.gamma_a_power = self.mu_base  # 2.0

        # === COHERENCE ===
        self.C_init = self.C_0  # 0.15
        self.lambda_C = self.tau_base / 10.0  # 0.002
        self.alpha_C = self.lambda_C * self.tau_eq_C  # 0.002 * 20 = 0.04

        # === GRAVITY ===
        self.alpha_grav = self.tau_base  # 0.02
        self.kappa_grav = self.kappa_base  # 5.0
        self.mu_grav = self.mu_base  # 2.0
        self.beta_g = 5.0 * self.mu_grav  # 10.0

        # === BOUNDARY ===
        self.F_VAC = 0.01
        self.F_MIN = 0.0
        self.F_MIN_grace = 0.05
        self.eta_heal = 0.03
        self.R_boundary = 2

    def to_legacy_params(self):
        """Convert to legacy DETParams3D format."""
        from det_v6_3_3d_collider import DETParams3D

        return DETParams3D(
            N=self.N,
            DT=self.DT,
            F_VAC=self.F_VAC,
            F_MIN=self.F_MIN,
            C_init=self.C_init,

            diff_enabled=self.diff_enabled,

            momentum_enabled=self.momentum_enabled,
            alpha_pi=self.alpha_pi,
            lambda_pi=self.lambda_pi,
            mu_pi=self.mu_pi,
            pi_max=self.pi_max,

            angular_momentum_enabled=self.angular_momentum_enabled,
            alpha_L=self.alpha_L,
            lambda_L=self.lambda_L,
            mu_L=self.mu_L,
            L_max=self.L_max,

            floor_enabled=self.floor_enabled,
            eta_floor=self.eta_floor,
            F_core=self.F_core,
            floor_power=self.floor_power,

            q_enabled=self.q_enabled,
            alpha_q=self.alpha_q,

            agency_dynamic=self.agency_dynamic,
            lambda_a=self.lambda_a,
            beta_a=self.beta_a,
            gamma_a_max=self.gamma_a_max,
            gamma_a_power=self.gamma_a_power,

            sigma_dynamic=self.sigma_dynamic,

            coherence_dynamic=self.coherence_dynamic,
            alpha_C=self.alpha_C,
            lambda_C=self.lambda_C,

            gravity_enabled=self.gravity_enabled,
            alpha_grav=self.alpha_grav,
            kappa_grav=self.kappa_grav,
            mu_grav=self.mu_grav,
            beta_g=self.beta_g,

            boundary_enabled=self.boundary_enabled,
            grace_enabled=self.grace_enabled,
            F_MIN_grace=self.F_MIN_grace,
            healing_enabled=self.healing_enabled,
            eta_heal=self.eta_heal,
            R_boundary=self.R_boundary,

            eta_lattice=self.eta_lattice,
            coherence_weighted_H=self.coherence_weighted_H,
            outflow_limit=self.outflow_limit,
        )

    def print_summary(self):
        """Print parameter summary showing base → derived relationships."""
        print("="*70)
        print("DET UNIFIED PARAMETER SCHEMA")
        print("="*70)

        print("\n--- BASE PARAMETERS (12) ---")
        print(f"  τ_base (time scale)     = {self.tau_base}")
        print(f"  σ_base (charging scale) = {self.sigma_base}")
        print(f"  λ_base (decay scale)    = {self.lambda_base}")
        print(f"  μ_base (mobility scale) = {self.mu_base}")
        print(f"  κ_base (coupling scale) = {self.kappa_base}")
        print(f"  C_0 (coherence scale)   = {self.C_0}")
        print(f"  φ_L (angular ratio)     = {self.phi_L}")
        print(f"  λ_a (agency coupling)   = {self.lambda_a}")
        print(f"  τ_eq_C (C eq. ratio)    = {self.tau_eq_C}")
        print(f"  π_max                   = {self.pi_max}")
        print(f"  μ_π factor              = {self.mu_pi_factor}")
        print(f"  use_golden_decay        = {self.use_golden_decay}")

        print("\n--- DERIVED PARAMETERS ---")
        print("\nMomentum (from σ_base, λ_base, μ_base):")
        print(f"  α_π = σ_base = {self.alpha_pi}")
        print(f"  λ_π = λ_base = {self.lambda_pi}")
        print(f"  μ_π = μ_base × {self.mu_pi_factor} = {self.mu_pi}")

        print("\nAngular Momentum (from momentum × φ_L):")
        print(f"  α_L = α_π × φ_L = {self.alpha_L}")
        if self.use_golden_decay:
            print(f"  λ_L = λ_π / φ = {self.lambda_L:.6f} (golden)")
        else:
            print(f"  λ_L = λ_π × φ_L = {self.lambda_L}")
        print(f"  μ_L = μ_π × φ_L = {self.mu_L}")
        print(f"  L_max = κ_base = {self.L_max}")

        print("\nFloor (from σ_base, κ_base, μ_base):")
        print(f"  η_floor = σ_base = {self.eta_floor}")
        print(f"  F_core = κ_base = {self.F_core}")
        print(f"  floor_power = μ_base = {self.floor_power}")

        print("\nStructure (from σ_base):")
        print(f"  α_q = σ_base / 10 = {self.alpha_q}")

        print("\nAgency (from τ_base, C_0, μ_base):")
        print(f"  β_a = 10 × τ_base = {self.beta_a}")
        print(f"  γ_a_max = C_0 = {self.gamma_a_max}")
        print(f"  γ_a_power = μ_base = {self.gamma_a_power}")

        print("\nCoherence (from τ_base, τ_eq_C, C_0):")
        print(f"  C_init = C_0 = {self.C_init}")
        print(f"  λ_C = τ_base / 10 = {self.lambda_C}")
        print(f"  α_C = λ_C × τ_eq_C = {self.alpha_C}")

        print("\nGravity (from τ_base, κ_base, μ_base):")
        print(f"  α_grav = τ_base = {self.alpha_grav}")
        print(f"  κ_grav = κ_base = {self.kappa_grav}")
        print(f"  μ_grav = μ_base = {self.mu_grav}")
        print(f"  β_g = 5 × μ_grav = {self.beta_g}")

        print("\n" + "="*70)
        print("PARAMETER COUNT: 12 base → 25+ derived")
        print("="*70)


def compare_to_legacy():
    """Compare unified params to legacy defaults."""
    print("\n" + "="*70)
    print("COMPARISON: UNIFIED vs LEGACY DEFAULTS")
    print("="*70)

    unified = DETUnifiedParams()

    # Legacy values from det_v6_3_3d_collider.py
    legacy = {
        'alpha_pi': 0.12,
        'lambda_pi': 0.008,
        'mu_pi': 0.35,
        'alpha_L': 0.06,
        'lambda_L': 0.005,
        'mu_L': 0.18,
        'eta_floor': 0.12,
        'F_core': 5.0,
        'floor_power': 2.0,
        'alpha_q': 0.012,
        'beta_a': 0.2,
        'gamma_a_max': 0.15,
        'gamma_a_power': 2.0,
        'alpha_C': 0.04,
        'lambda_C': 0.002,
        'C_init': 0.15,
        'alpha_grav': 0.02,
        'kappa_grav': 5.0,
        'mu_grav': 2.0,
        'beta_g': 10.0,
    }

    print(f"\n{'Parameter':<15} {'Legacy':>10} {'Unified':>10} {'Match':>8}")
    print("-"*50)

    all_match = True
    for name, leg_val in legacy.items():
        uni_val = getattr(unified, name)
        match = abs(leg_val - uni_val) < 0.001
        all_match = all_match and match
        match_str = "✓" if match else f"Δ={uni_val-leg_val:.4f}"
        print(f"{name:<15} {leg_val:>10.4f} {uni_val:>10.4f} {match_str:>8}")

    print("-"*50)
    if all_match:
        print("✓ All parameters match legacy defaults!")
    else:
        print("⚠ Some parameters differ (see Δ values)")


if __name__ == "__main__":
    params = DETUnifiedParams()
    params.print_summary()
    compare_to_legacy()
