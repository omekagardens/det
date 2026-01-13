"""
DET v6.3 1D Collider - Unified Implementation
=============================================

Complete implementation with all DET modules:
- Gravity module (Section V): Helmholtz baseline, Poisson potential, gravitational flux
- Boundary operators (Section VI): Grace injection, bond healing
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics with gravity coupling (IV.4)
- Floor repulsion (IV.6)

Reference: DET Theory Card v6.3

Changelog from v6.2:
- Added beta_g parameter for gravity-momentum coupling
- Added lattice correction factor eta
- Updated version references
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.fft import fft, ifft


def compute_lattice_correction_1d(N: int) -> float:
    """Compute lattice correction factor eta for 1D grid."""
    if N <= 64:
        return 0.92
    elif N <= 128:
        return 0.96
    elif N <= 256:
        return 0.98
    else:
        return 0.99


@dataclass
class DETParams1D:
    """DET v6.3 1D simulation parameters - complete."""
    N: int = 200
    DT: float = 0.02
    F_VAC: float = 0.01
    F_MIN: float = 0.0

    # Coherence
    C_init: float = 0.3

    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.10
    lambda_pi: float = 0.02
    mu_pi: float = 0.30
    pi_max: float = 3.0

    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015

    # Agency dynamics (VI.2B)
    a_coupling: float = 30.0
    a_rate: float = 0.2

    # Floor repulsion (IV.6)
    floor_enabled: bool = True
    eta_floor: float = 0.15
    F_core: float = 5.0
    floor_power: float = 2.0

    # Gravity module (V.1-V.3)
    gravity_enabled: bool = True
    alpha_grav: float = 0.05
    kappa_grav: float = 2.0
    mu_grav: float = 1.0
    beta_g: float = 5.0  # v6.3: gravity-momentum coupling

    # Boundary operators (VI)
    boundary_enabled: bool = True
    grace_enabled: bool = True
    F_MIN_grace: float = 0.05
    healing_enabled: bool = False
    eta_heal: float = 0.03
    R_boundary: int = 3

    # v6.3: Lattice correction factor
    eta_lattice: float = 0.92

    # Numerical stability
    outflow_limit: float = 0.25


def periodic_local_sum_1d(x: np.ndarray, radius: int) -> np.ndarray:
    """Compute local sum within radius (periodic BCs)."""
    result = np.zeros_like(x)
    for d in range(-radius, radius + 1):
        result += np.roll(x, d)
    return result


class DETCollider1D:
    """
    DET v6.3 1D Collider - Unified with Gravity, Boundary Operators, and Lattice Correction

    Key v6.3 features:
    - beta_g gravity-momentum coupling
    - Lattice correction factor eta
    - Enhanced time dilation tracking
    """

    def __init__(self, params: Optional[DETParams1D] = None):
        self.p = params or DETParams1D()
        N = self.p.N

        # Update lattice correction based on grid size
        self.p.eta_lattice = compute_lattice_correction_1d(N)

        # Per-node state
        self.F = np.ones(N) * self.p.F_VAC
        self.q = np.zeros(N)
        self.a = np.ones(N)

        # Per-bond state
        self.pi_R = np.zeros(N)
        self.C_R = np.ones(N) * self.p.C_init
        self.sigma = np.ones(N)

        # Gravity fields
        self.b = np.zeros(N)
        self.Phi = np.zeros(N)
        self.g = np.zeros(N)

        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones(N)
        self.Delta_tau = np.ones(N) * self.p.DT

        # Time dilation tracking (v6.3)
        self.accumulated_proper_time = np.zeros(N)

        # Boundary operator diagnostics
        self.last_grace_injection = np.zeros(N)
        self.last_healing = np.zeros(N)
        self.total_grace_injected = 0.0

        self._setup_fft_solvers()

    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers."""
        N = self.p.N
        k = np.fft.fftfreq(N) * N
        self.L_k = -4 * np.sin(np.pi * k / N)**2
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0] = 1.0

    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        source_k = fft(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifft(b_k))

    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        source_k = fft(source)
        source_k[0] = 0
        # Apply lattice correction (v6.3)
        Phi_k = -self.p.kappa_grav * self.p.eta_lattice * source_k / self.L_k_poisson
        Phi_k[0] = 0
        return np.real(ifft(Phi_k))

    def _compute_gravity(self):
        """Compute gravitational fields from q."""
        if not self.p.gravity_enabled:
            self.g = np.zeros(self.p.N)
            return

        self.b = self._solve_helmholtz(self.q)
        rho = self.q - self.b
        self.Phi = self._solve_poisson(rho)

        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        self.g = -0.5 * (R(self.Phi) - L(self.Phi))

    def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
        """Grace Injection per DET VI.5"""
        p = self.p
        n = np.maximum(0, p.F_MIN_grace - self.F)
        w = self.a * n
        w_sum = periodic_local_sum_1d(w, p.R_boundary) + 1e-12
        I_g = D * w / w_sum
        return I_g

    def compute_bond_healing(self, D: np.ndarray) -> np.ndarray:
        """Bond Healing Operator (Agency-Gated)"""
        p = self.p
        R = lambda x: np.roll(x, -1)
        g_R = np.sqrt(self.a * R(self.a))
        room = 1.0 - self.C_R
        D_avg_R = 0.5 * (D + R(D))
        Delta_tau_R = 0.5 * (self.Delta_tau + R(self.Delta_tau))
        dC_heal = p.eta_heal * g_R * room * D_avg_R * Delta_tau_R
        return dC_heal

    def add_packet(self, center: int, mass: float = 5.0,
                   width: float = 5.0, momentum: float = 0, initial_q: float = 0.0):
        """Add a Gaussian resource packet."""
        N = self.p.N
        x = np.arange(N)
        r2 = (x - center)**2
        envelope = np.exp(-0.5 * r2 / width**2)

        self.F += mass * envelope
        self.C_R = np.clip(self.C_R + 0.5 * envelope, self.p.C_init, 1.0)

        if momentum != 0:
            mom_env = np.exp(-0.5 * r2 / (width * 2)**2)
            self.pi_R += momentum * mom_env

        if initial_q > 0:
            self.q += initial_q * envelope

        self._clip()

    def _clip(self):
        self.F = np.clip(self.F, self.p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_R = np.clip(self.pi_R, -self.p.pi_max, self.p.pi_max)

    def step(self):
        """Execute one canonical DET update step per Theory Card v6.3."""
        p = self.p
        N = p.N
        dk = p.DT

        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)

        # STEP 0: Gravity
        self._compute_gravity()

        # STEP 1: Presence (III.1)
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk

        # Track accumulated proper time (v6.3)
        self.accumulated_proper_time += self.Delta_tau

        Delta_tau_R = 0.5 * (self.Delta_tau + R(self.Delta_tau))

        # STEP 2: Flow computation
        classical_R = self.F - R(self.F)
        classical_L = self.F - L(self.F)

        sqrt_C_R = np.sqrt(self.C_R)
        sqrt_C_L = np.sqrt(L(self.C_R))

        drive_R = (1 - sqrt_C_R) * classical_R
        drive_L = (1 - sqrt_C_L) * classical_L

        g_R = np.sqrt(self.a * R(self.a))
        g_L = np.sqrt(self.a * L(self.a))

        cond_R = self.sigma * (self.C_R + 1e-4)
        cond_L = self.sigma * (L(self.C_R) + 1e-4)

        J_diff_R = g_R * cond_R * drive_R
        J_diff_L = g_L * cond_L * drive_L

        # Momentum flux
        if p.momentum_enabled:
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            J_mom_R = p.mu_pi * self.sigma * self.pi_R * F_avg_R
            J_mom_L = -p.mu_pi * self.sigma * L(self.pi_R) * F_avg_L
        else:
            J_mom_R = J_mom_L = 0

        # Floor flux
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_R = p.eta_floor * self.sigma * (s + R(s)) * classical_R
            J_floor_L = p.eta_floor * self.sigma * (s + L(s)) * classical_L
        else:
            J_floor_R = J_floor_L = 0

        # Gravitational flux
        if p.gravity_enabled:
            g_bond_R = 0.5 * (self.g + R(self.g))
            g_bond_L = 0.5 * (self.g + L(self.g))
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            J_grav_R = p.mu_grav * self.sigma * g_bond_R * F_avg_R
            J_grav_L = p.mu_grav * self.sigma * g_bond_L * F_avg_L
        else:
            J_grav_R = J_grav_L = 0

        # Total flux
        J_R = J_diff_R + J_mom_R + J_floor_R + J_grav_R
        J_L = J_diff_L + J_mom_L + J_floor_L + J_grav_L

        # STEP 3: Limiter
        total_outflow = np.maximum(0, J_R) + np.maximum(0, J_L)
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))

        J_R_lim = np.where(J_R > 0, J_R * scale, J_R)
        J_L_lim = np.where(J_L > 0, J_L * scale, J_L)
        J_diff_R_scaled = np.where(J_diff_R > 0, J_diff_R * scale, J_diff_R)

        # Dissipation
        D = (np.abs(J_R_lim) + np.abs(J_L_lim)) * self.Delta_tau

        # STEP 4: Resource update
        transfer_R = J_R_lim * self.Delta_tau
        transfer_L = J_L_lim * self.Delta_tau
        outflow = transfer_R + transfer_L
        inflow = L(transfer_R) + R(transfer_L)
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)

        # STEP 5: Grace Injection (VI.5)
        if p.boundary_enabled and p.grace_enabled:
            I_g = self.compute_grace_injection(D)
            self.F = self.F + I_g
            self.last_grace_injection = I_g.copy()
            self.total_grace_injected += np.sum(I_g)
        else:
            self.last_grace_injection = np.zeros(N)

        # STEP 6: Momentum update with gravity coupling (v6.3: beta_g)
        if p.momentum_enabled:
            decay_R = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_R)
            dpi_diff = p.alpha_pi * J_diff_R_scaled * Delta_tau_R
            if p.gravity_enabled:
                g_bond_R = 0.5 * (self.g + R(self.g))
                dpi_grav = p.beta_g * g_bond_R * Delta_tau_R
            else:
                dpi_grav = 0
            self.pi_R = decay_R * self.pi_R + dpi_diff + dpi_grav

        # STEP 7: Structure update
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)

        # STEP 8: Bond Healing
        if p.boundary_enabled and p.healing_enabled:
            dC_heal = self.compute_bond_healing(D)
            self.C_R = np.clip(self.C_R + dC_heal, p.C_init, 1.0)
            self.last_healing = dC_heal.copy()
        else:
            self.last_healing = np.zeros(N)

        # STEP 9: Agency update
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)

        self._clip()
        self.time += dk
        self.step_count += 1

    def total_mass(self) -> float:
        return float(np.sum(self.F))

    def center_of_mass(self) -> float:
        x = np.arange(self.p.N)
        total = np.sum(self.F) + 1e-9
        return float(np.sum(x * self.F) / total)

    def total_q(self) -> float:
        return float(np.sum(self.q))

    def potential_energy(self) -> float:
        return float(np.sum(self.F * self.Phi))

    def separation(self) -> float:
        from scipy.ndimage import label
        threshold = self.p.F_VAC * 10
        above = self.F > threshold
        labeled, num = label(above)

        if num < 2:
            return 0.0

        N = self.p.N
        x = np.arange(N)

        coms = []
        for i in range(1, num + 1):
            mask = labeled == i
            if np.sum(mask) == 0:
                continue
            weights = self.F[mask]
            total_mass = np.sum(weights)
            if total_mass < 0.1:
                continue
            com = np.sum(x[mask] * weights) / total_mass
            coms.append({'x': com, 'mass': total_mass})

        coms.sort(key=lambda c: -c['mass'])

        if len(coms) < 2:
            return 0.0

        dx = coms[1]['x'] - coms[0]['x']
        if dx > N/2: dx -= N
        if dx < -N/2: dx += N

        return abs(dx)


# ============================================================
# TEST SUITE
# ============================================================

def test_v6_3_gravity_vacuum(verbose: bool = True) -> bool:
    """Gravity has no effect in vacuum (q=0)"""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)

    params = DETParams1D(gravity_enabled=True, q_enabled=False, boundary_enabled=False)
    sim = DETCollider1D(params)

    for _ in range(500):
        sim.step()

    max_g = np.max(np.abs(sim.g))
    max_Phi = np.max(np.abs(sim.Phi))

    passed = max_g < 1e-10 and max_Phi < 1e-10

    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Phi|: {max_Phi:.2e}")
        print(f"  Lattice correction eta: {sim.p.eta_lattice:.3f}")
        print(f"  Vacuum gravity {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_binding(verbose: bool = True) -> bool:
    """Test gravitational binding with beta_g coupling."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravitational Binding (v6.3)")
        print("="*60)

    params = DETParams1D(
        N=200, DT=0.02, F_VAC=0.001, F_MIN=0.0,
        C_init=0.5,
        momentum_enabled=True, alpha_pi=0.2, lambda_pi=0.002, mu_pi=1.0,
        q_enabled=True, alpha_q=0.02,
        a_coupling=3.0, a_rate=0.05,
        floor_enabled=False,
        gravity_enabled=True, alpha_grav=0.01, kappa_grav=10.0, mu_grav=5.0,
        beta_g=25.0,  # v6.3 gravity-momentum coupling
        boundary_enabled=True, grace_enabled=True
    )

    sim = DETCollider1D(params)

    initial_sep = 60
    center = params.N // 2
    sim.add_packet(center - initial_sep//2, mass=8.0, width=5.0, momentum=0.1, initial_q=0.3)
    sim.add_packet(center + initial_sep//2, mass=8.0, width=5.0, momentum=-0.1, initial_q=0.3)

    min_sep = initial_sep

    for t in range(2000):
        sep = sim.separation()
        min_sep = min(min_sep, sep)

        if verbose and t % 400 == 0:
            print(f"  t={t}: sep={sep:.1f}, PE={sim.potential_energy():.3f}")

        sim.step()

    passed = min_sep < initial_sep * 0.5

    if verbose:
        print(f"\n  Initial sep: {initial_sep:.1f}")
        print(f"  Min sep: {min_sep:.1f}")
        print(f"  beta_g: {params.beta_g}")
        print(f"  Binding {'PASSED' if passed else 'FAILED'}")

    return passed


def test_v6_3_time_dilation(verbose: bool = True) -> bool:
    """Test time dilation P = a*sigma/(1+F)/(1+H)."""
    if verbose:
        print("\n" + "="*60)
        print("TEST: Time Dilation (v6.3)")
        print("="*60)

    params = DETParams1D(
        N=200,
        gravity_enabled=True, q_enabled=True,
        boundary_enabled=False
    )
    sim = DETCollider1D(params)

    center = params.N // 2
    sim.add_packet(center, mass=20.0, width=5.0, initial_q=0.5)

    for _ in range(200):
        sim.step()

    F_center = sim.F[center]
    F_edge = sim.F[center + 50]
    P_center = sim.P[center]
    P_edge = sim.P[center + 50]

    time_dilated = P_center < P_edge

    # Check formula
    a_center = sim.a[center]
    sigma_center = sim.sigma[center]
    H_center = sigma_center
    predicted_P = a_center * sigma_center / (1 + F_center) / (1 + H_center)
    formula_error = abs(P_center - predicted_P) / (predicted_P + 1e-10)

    passed = time_dilated and formula_error < 0.01

    if verbose:
        print(f"  F at center: {F_center:.4f}")
        print(f"  F at edge: {F_edge:.4f}")
        print(f"  P at center: {P_center:.6f}")
        print(f"  P at edge: {P_edge:.6f}")
        print(f"  Time dilation (P_center < P_edge): {time_dilated}")
        print(f"  Formula error: {formula_error*100:.4f}%")
        print(f"  Time dilation {'PASSED' if passed else 'FAILED'}")

    return passed


def run_v6_3_test_suite():
    """Run v6.3 1D test suite."""
    print("="*70)
    print("DET v6.3 1D COLLIDER - TEST SUITE")
    print("="*70)

    results = {}

    results['vacuum_gravity'] = test_v6_3_gravity_vacuum(verbose=True)
    results['binding'] = test_v6_3_binding(verbose=True)
    results['time_dilation'] = test_v6_3_time_dilation(verbose=True)

    print("\n" + "="*70)
    print("v6.3 1D TEST SUMMARY")
    print("="*70)

    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n  OVERALL: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return results


if __name__ == "__main__":
    run_v6_3_test_suite()
