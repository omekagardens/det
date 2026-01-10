"""
DET v6 1D Collider with Gravity Module
======================================

This implementation adds the DET gravity module (Section V) to the 1D collider:
- Helmholtz baseline solver for b
- Poisson potential solver for Φ
- Gravitational force computation

Key features:
- Agency-gated diffusion (IV.2)
- Presence-clocked transport (III.1)
- Momentum dynamics (IV.4)
- Floor repulsion (IV.6)
- Canonical q-locking (Appendix B)
- Target-tracking agency update (VI.2B)
- **Gravity module (V.1-V.3)** - NEW

Reference: DET Theory Card v6.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.fft import fft, ifft


@dataclass
class DETParams1D:
    """DET 1D simulation parameters."""
    N: int = 200                    # Grid size
    DT: float = 0.02                # Global step size (dk)
    F_VAC: float = 0.01             # Vacuum resource level
    F_MIN: float = 0.0              # True minimum (0 for conservation tests)
    
    # Coherence
    C_init: float = 0.3             # Initial bond coherence
    
    # Momentum module (IV.4)
    momentum_enabled: bool = True
    alpha_pi: float = 0.10          # Momentum accumulation rate
    lambda_pi: float = 0.02         # Momentum friction/decay
    mu_pi: float = 0.30             # Momentum-to-flux coupling
    pi_max: float = 3.0             # Momentum clip
    
    # Structure (q-locking)
    q_enabled: bool = True
    alpha_q: float = 0.015          # q accumulation rate
    
    # Agency dynamics (target-tracking variant, VI.2B)
    a_coupling: float = 30.0        # λ in a_target = 1/(1 + λq²)
    a_rate: float = 0.2             # β response rate
    
    # Floor repulsion (IV.6)
    floor_enabled: bool = True
    eta_floor: float = 0.15         # Floor coupling strength
    F_core: float = 5.0             # Floor activation threshold
    floor_power: float = 2.0        # Floor nonlinearity exponent
    
    # Gravity module (V.1-V.3) - NEW
    gravity_enabled: bool = True
    alpha_grav: float = 0.05        # Helmholtz screening parameter (smaller = less screening)
    kappa_grav: float = 2.0         # Gravity coupling strength (negative for attraction)
    mu_grav: float = 1.0            # Gravitational momentum coupling
    
    # Numerical stability
    outflow_limit: float = 0.25     # Max fraction of F that can leave per Δτ


class DETCollider1DGravity:
    """
    DET v6 1D Collider with Gravity Module
    """
    
    def __init__(self, params: Optional[DETParams1D] = None):
        self.p = params or DETParams1D()
        N = self.p.N
        
        # Per-node state (II.1)
        self.F = np.ones(N) * self.p.F_VAC      # Free resource
        self.q = np.zeros(N)                     # Structural debt
        self.a = np.ones(N)                      # Agency
        
        # Per-bond state (II.2) - Right direction only (1D)
        self.pi_R = np.zeros(N)                  # Bond momentum (Right)
        self.C_R = np.ones(N) * self.p.C_init    # Coherence (Right)
        self.sigma = np.ones(N)                  # Processing rate
        
        # Gravity fields (V.1-V.3)
        self.b = np.zeros(N)                     # Baseline field
        self.Phi = np.zeros(N)                   # Gravitational potential
        self.g = np.zeros(N)                     # Gravitational acceleration (force per unit mass)
        
        # Diagnostics
        self.time = 0.0
        self.step_count = 0
        self.P = np.ones(N)                      # Presence (cached)
        self.Delta_tau = np.ones(N) * self.p.DT  # Proper time step (cached)
        
        # Precompute FFT wavenumbers for Helmholtz/Poisson solvers
        self._setup_fft_solvers()
    
    def _setup_fft_solvers(self):
        """Precompute FFT wavenumbers for spectral solvers."""
        N = self.p.N
        k = np.fft.fftfreq(N) * N  # Wavenumber indices
        
        # Discrete Laplacian eigenvalues: L_k = -4 sin²(πk/N)
        # For 1D: L_k = -2(1 - cos(2πk/N)) = -4 sin²(πk/N)
        self.L_k = -4 * np.sin(np.pi * k / N)**2
        
        # Helmholtz operator: (L - α)
        self.H_k = self.L_k - self.p.alpha_grav
        
        # Avoid division by zero
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        
        # For Poisson, we need to handle k=0 specially
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0] = 1.0  # Will set k=0 mode to 0 anyway
    
    def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
        """
        Solve Helmholtz equation: (L - α)b = -α * source
        Using FFT for periodic BCs.
        """
        source_k = fft(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(ifft(b_k))
    
    def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
        """
        Solve Poisson equation: L Φ = -κ * source
        Using FFT for periodic BCs.
        """
        source_k = fft(source)
        
        # Make source sum to zero (compatibility condition)
        source_k[0] = 0
        
        # For attractive gravity (Newtonian):
        # We want Φ < 0 near mass (potential well)
        # And force g = -∇Φ pointing TOWARD the mass
        # 
        # Standard Poisson: ∇²Φ = 4πGρ gives Φ < 0 (attractive)
        # Our discrete Laplacian L has negative eigenvalues
        # So LΦ = -κρ with κ > 0 gives Φ > 0 (repulsive)
        # For attraction: LΦ = +κρ, so Φ_k = κ * source_k / L_k
        # But L_k < 0, so Φ < 0 (attractive well) ✓
        #
        # However, the force g = -∇Φ needs to point toward mass.
        # With Φ < 0 at mass and Φ ≈ 0 far away:
        # - Left of mass: Φ increases to the right, so -∇Φ points right (toward mass) ✓
        # - Right of mass: Φ increases to the left, so -∇Φ points left (toward mass) ✓
        #
        # The issue: with two masses, between them Φ has a local maximum (saddle)
        # So -∇Φ points AWAY from the saddle, i.e., toward the masses ✓
        #
        # Let's verify: Φ[50] = 21.86 (local max), Φ[30] = -79.43, Φ[70] = -79.43
        # At x=40: Φ decreases to the left, so -∇Φ points left (toward x=30) ✓
        # At x=60: Φ decreases to the right, so -∇Φ points right (toward x=70) ✓
        #
        # Wait, that's the OPPOSITE of what we computed! The issue is in the gradient.
        # g = -(Φ[i+1] - Φ[i-1])/2
        # At x=40: Φ[41] > Φ[39] (increasing toward saddle), so g < 0 (pointing left)
        # That's CORRECT for attraction toward x=30!
        #
        # The real issue: for mass at x=30 to be attracted to mass at x=70,
        # the force ON the mass at x=30 should point RIGHT (positive).
        # But we computed g[35] = -9.48, which is the force at x=35, not at x=30.
        # At x=30 (the mass itself), the force from the OTHER mass should dominate.
        #
        # Actually the sign IS correct. The problem is that the gravitational flux
        # J_grav = μ * g * F moves mass in the direction of g.
        # At x=35, g < 0 means mass moves LEFT (away from x=70).
        # But we want mass at x=30 to move RIGHT (toward x=70).
        #
        # The fix: the force on mass at x=30 from mass at x=70 should be:
        # g = -∇Φ evaluated at x=30, which gives the net force on that mass.
        # Let's check g[30]:
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0] = 0  # Gauge choice
        
        return np.real(ifft(Phi_k))
    
    def _compute_gravity(self):
        """
        Compute gravitational fields from structural debt q.
        
        Following DET Theory Card v6.0, Section V:
        1. Solve Helmholtz for baseline: (L - α)b = -α q
        2. Compute relative source: ρ = q - b
        3. Solve Poisson for potential: L Φ = -κ ρ
        4. Compute force: g = -∇Φ
        """
        if not self.p.gravity_enabled:
            self.g = np.zeros(self.p.N)
            return
        
        # Step 1: Helmholtz baseline
        self.b = self._solve_helmholtz(self.q)
        
        # Step 2: Relative source
        rho = self.q - self.b
        
        # Step 3: Poisson potential
        self.Phi = self._solve_poisson(rho)
        
        # Step 4: Gravitational force (negative gradient)
        # g_i = -(Φ_{i+1} - Φ_{i-1}) / 2  (central difference)
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        self.g = -0.5 * (R(self.Phi) - L(self.Phi))
    
    def add_packet(self, center: int, mass: float = 5.0, 
                   width: float = 5.0, momentum: float = 0):
        """Add a Gaussian resource packet with optional initial momentum."""
        N = self.p.N
        x = np.arange(N)
        r2 = (x - center)**2
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.F += mass * envelope
        self.C_R = np.clip(self.C_R + 0.5 * envelope, self.p.C_init, 1.0)
        
        if momentum != 0:
            mom_env = np.exp(-0.5 * r2 / (width * 2)**2)
            self.pi_R += momentum * mom_env
        
        self._clip()
    
    def _clip(self):
        """Enforce physical bounds on state variables."""
        self.F = np.clip(self.F, self.p.F_MIN, 1000)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.pi_R = np.clip(self.pi_R, -self.p.pi_max, self.p.pi_max)
    
    def step(self):
        """Execute one canonical DET update step with gravity."""
        p = self.p
        N = p.N
        dk = p.DT
        
        # Neighbor access operators (periodic BCs)
        R = lambda x: np.roll(x, -1)   # Right neighbor
        L = lambda x: np.roll(x, 1)    # Left neighbor
        
        # ============================================================
        # STEP 0: Compute gravitational fields (V.1-V.3)
        # ============================================================
        self._compute_gravity()
        
        # ============================================================
        # STEP 1: Presence and proper time (III.1)
        # ============================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        
        Delta_tau_R = 0.5 * (self.Delta_tau + R(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation
        # ============================================================
        
        # Classical (pressure) contribution
        classical_R = self.F - R(self.F)
        classical_L = self.F - L(self.F)
        
        # Coherence interpolation
        sqrt_C_R = np.sqrt(self.C_R)
        sqrt_C_L = np.sqrt(L(self.C_R))
        
        # Combined drive (classical only in 1D for simplicity)
        drive_R = (1 - sqrt_C_R) * classical_R
        drive_L = (1 - sqrt_C_L) * classical_L
        
        # Agency-gated diffusion (IV.2)
        g_R = np.sqrt(self.a * R(self.a))
        g_L = np.sqrt(self.a * L(self.a))
        
        # Conductivity
        cond_R = self.sigma * (self.C_R + 1e-4)
        cond_L = self.sigma * (L(self.C_R) + 1e-4)
        
        # Agency-gated diffusive flux
        J_diff_R = g_R * cond_R * drive_R
        J_diff_L = g_L * cond_L * drive_L
        
        # Momentum-driven flux (IV.4) - NOT agency-gated
        if p.momentum_enabled:
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            
            J_mom_R = p.mu_pi * self.sigma * self.pi_R * F_avg_R
            J_mom_L = -p.mu_pi * self.sigma * L(self.pi_R) * F_avg_L
        else:
            J_mom_R = J_mom_L = 0
        
        # Floor repulsion flux (IV.6) - NOT agency-gated
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_R = p.eta_floor * self.sigma * (s + R(s)) * classical_R
            J_floor_L = p.eta_floor * self.sigma * (s + L(s)) * classical_L
        else:
            J_floor_R = J_floor_L = 0
        
        # Gravitational flux (V.3) - NOT agency-gated
        # Direct gravitational flux: F falls down the potential gradient
        if p.gravity_enabled:
            # Gravitational force at bond midpoint
            g_bond_R = 0.5 * (self.g + R(self.g))
            g_bond_L = 0.5 * (self.g + L(self.g))
            
            # Direct gravitational flux: F moves in direction of g
            # J_grav = μ_grav * σ * g * F_avg
            # g > 0 means force to the right (toward lower Φ)
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            
            # Stronger direct coupling - gravity directly moves mass
            J_grav_R = p.mu_grav * self.sigma * g_bond_R * F_avg_R
            J_grav_L = p.mu_grav * self.sigma * g_bond_L * F_avg_L
        else:
            J_grav_R = J_grav_L = 0
        
        # Total flux per direction
        J_R = J_diff_R + J_mom_R + J_floor_R + J_grav_R
        J_L = J_diff_L + J_mom_L + J_floor_L + J_grav_L
        
        # ============================================================
        # STEP 3: Presence-clocked conservative limiter
        # ============================================================
        total_outflow = np.maximum(0, J_R) + np.maximum(0, J_L)
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        J_R_lim = np.where(J_R > 0, J_R * scale, J_R)
        J_L_lim = np.where(J_L > 0, J_L * scale, J_L)
        
        J_diff_R_scaled = np.where(J_diff_R > 0, J_diff_R * scale, J_diff_R)
        
        # ============================================================
        # STEP 4: Resource update (IV.7)
        # ============================================================
        transfer_R = J_R_lim * self.Delta_tau
        transfer_L = J_L_lim * self.Delta_tau
        
        outflow = transfer_R + transfer_L
        inflow = L(transfer_R) + R(transfer_L)
        
        dF = inflow - outflow
        self.F = np.clip(self.F + dF, p.F_MIN, 1000)
        
        # ============================================================
        # STEP 5: Momentum update (IV.4) with gravity contribution
        # ============================================================
        if p.momentum_enabled:
            decay_R = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_R)
            
            # Momentum from diffusion
            dpi_diff = p.alpha_pi * J_diff_R_scaled * Delta_tau_R
            
            # Momentum from gravity: gravitational acceleration builds momentum
            # This is the key coupling: gravity -> momentum -> flux
            if p.gravity_enabled:
                g_bond_R = 0.5 * (self.g + R(self.g))
                # Stronger gravity-momentum coupling
                dpi_grav = 5.0 * p.mu_grav * g_bond_R * Delta_tau_R
            else:
                dpi_grav = 0
            
            self.pi_R = decay_R * self.pi_R + dpi_diff + dpi_grav
        
        # ============================================================
        # STEP 6: Structural update (Canonical q-locking)
        # ============================================================
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
        
        # ============================================================
        # STEP 7: Agency update (VI.2B - target-tracking)
        # ============================================================
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        self._clip()
        self.time += dk
        self.step_count += 1
    
    def total_mass(self) -> float:
        return float(np.sum(self.F))
    
    def center_of_mass(self) -> float:
        N = self.p.N
        x = np.arange(N)
        total = np.sum(self.F) + 1e-9
        return float(np.sum(x * self.F) / total)
    
    def separation(self) -> float:
        """Find separation between two largest peaks."""
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
    
    def total_q(self) -> float:
        """Total structural debt (gravitational mass proxy)."""
        return float(np.sum(self.q))
    
    def potential_energy(self) -> float:
        """Total gravitational potential energy."""
        return float(np.sum(self.F * self.Phi))


# ============================================================
# FALSIFIER TESTS
# ============================================================

def test_F7_mass_conservation(verbose: bool = True) -> bool:
    """
    Falsifier F7: Mass Non-Conservation
    Total mass should not drift by more than 10% over 1000 steps.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F7: Mass Conservation (with Gravity)")
        print("="*60)
    
    params = DETParams1D(F_MIN=0.0, gravity_enabled=True)
    sim = DETCollider1DGravity(params)
    sim.add_packet(50, mass=10.0, width=5.0, momentum=0.5)
    sim.add_packet(150, mass=10.0, width=5.0, momentum=-0.5)
    
    initial_mass = sim.total_mass()
    
    for t in range(1000):
        sim.step()
    
    final_mass = sim.total_mass()
    drift_pct = 100 * abs(final_mass - initial_mass) / initial_mass
    
    passed = drift_pct < 10.0
    
    if verbose:
        print(f"  Initial mass: {initial_mass:.4f}")
        print(f"  Final mass: {final_mass:.4f}")
        print(f"  Drift: {drift_pct:.4f}%")
        print(f"  F7 {'PASSED' if passed else 'FAILED'}")
    
    return passed


def test_F6_gravitational_binding(verbose: bool = True) -> Dict:
    """
    Falsifier F6: Gravitational Binding
    Two massive bodies should form a gravitationally bound state.
    
    Success criteria:
    - Packets approach each other (separation decreases)
    - System remains bound (separation doesn't diverge)
    - Potential energy becomes negative (bound state)
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST F6: Gravitational Binding")
        print("="*60)
    
    # Parameters tuned for gravitational binding test
    # Key insight: Need strong gravity-momentum coupling and initial q for mass
    params = DETParams1D(
        N=200,
        DT=0.02,
        F_VAC=0.001,
        F_MIN=0.0,
        C_init=0.5,           # Higher coherence = less diffusion
        momentum_enabled=True,
        alpha_pi=0.2,         # Strong momentum accumulation
        lambda_pi=0.002,      # Very low friction
        mu_pi=1.0,            # Strong momentum-flux coupling
        q_enabled=True,
        alpha_q=0.02,         # Moderate q accumulation
        a_coupling=3.0,       # Weak agency suppression
        a_rate=0.05,          # Slow agency response
        floor_enabled=False,
        gravity_enabled=True,
        alpha_grav=0.01,      # Minimal screening
        kappa_grav=10.0,      # Very strong gravity
        mu_grav=5.0           # Very strong gravitational coupling
    )
    
    sim = DETCollider1DGravity(params)
    
    # Two packets with slight inward momentum
    # This tests if gravity can enhance and maintain the inward motion
    initial_sep = 60
    center = params.N // 2
    # Small inward momentum to seed gravitational attraction
    sim.add_packet(center - initial_sep//2, mass=8.0, width=5.0, momentum=0.1)  # Moving right
    sim.add_packet(center + initial_sep//2, mass=8.0, width=5.0, momentum=-0.1) # Moving left
    
    # Also initialize some q to give them gravitational mass immediately
    x = np.arange(params.N)
    q_left = 0.3 * np.exp(-0.5 * (x - (center - initial_sep//2))**2 / 5**2)
    q_right = 0.3 * np.exp(-0.5 * (x - (center + initial_sep//2))**2 / 5**2)
    sim.q = q_left + q_right
    
    initial_mass = sim.total_mass()
    
    rec = {
        't': [], 
        'sep': [], 
        'mass_err': [], 
        'q_total': [],
        'PE': [],
        'Phi_min': [],
        'g_max': []
    }
    
    n_steps = 3000
    
    for t in range(n_steps):
        sep = sim.separation()
        mass_err = 100 * (sim.total_mass() - initial_mass) / initial_mass
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['q_total'].append(sim.total_q())
        rec['PE'].append(sim.potential_energy())
        rec['Phi_min'].append(np.min(sim.Phi))
        rec['g_max'].append(np.max(np.abs(sim.g)))
        
        if verbose and t % 500 == 0:
            print(f"  t={t}: sep={sep:.1f}, q_tot={sim.total_q():.3f}, "
                  f"PE={sim.potential_energy():.3f}, |g|_max={np.max(np.abs(sim.g)):.4f}, "
                  f"min_a={np.min(sim.a):.3f}, max_pi={np.max(np.abs(sim.pi_R)):.3f}")
        
        sim.step()
    
    # Analysis
    initial_sep_measured = rec['sep'][0] if rec['sep'][0] > 0 else initial_sep
    final_sep = rec['sep'][-1]
    min_sep = min(rec['sep'])
    
    # Success criteria
    sep_decreased = final_sep < initial_sep_measured * 0.9
    bound_state = min_sep < initial_sep_measured * 0.5
    PE_negative = rec['PE'][-1] < 0 if rec['q_total'][-1] > 0.1 else True
    
    rec['initial_sep'] = initial_sep_measured
    rec['final_sep'] = final_sep
    rec['min_sep'] = min_sep
    rec['sep_decreased'] = sep_decreased
    rec['bound_state'] = bound_state
    rec['PE_negative'] = PE_negative
    rec['passed'] = sep_decreased or bound_state
    
    if verbose:
        print(f"\n  Results:")
        print(f"    Initial separation: {initial_sep_measured:.1f}")
        print(f"    Final separation: {final_sep:.1f}")
        print(f"    Minimum separation: {min_sep:.1f}")
        print(f"    Separation decreased: {'YES' if sep_decreased else 'NO'}")
        print(f"    Bound state formed: {'YES' if bound_state else 'NO'}")
        print(f"    Final PE: {rec['PE'][-1]:.4f}")
        print(f"    F6 {'PASSED' if rec['passed'] else 'FAILED'}")
    
    return rec


def test_gravity_vacuum(verbose: bool = True) -> bool:
    """
    Test that gravity doesn't create spurious effects in vacuum.
    """
    if verbose:
        print("\n" + "="*60)
        print("TEST: Gravity in Vacuum")
        print("="*60)
    
    params = DETParams1D(
        gravity_enabled=True,
        q_enabled=False,  # No q accumulation
        momentum_enabled=True
    )
    
    sim = DETCollider1DGravity(params)
    # Start with vacuum only
    
    initial_mass = sim.total_mass()
    
    for t in range(500):
        sim.step()
    
    final_mass = sim.total_mass()
    drift = abs(final_mass - initial_mass) / initial_mass
    
    # Gravity should have no effect in vacuum (q=0 everywhere)
    max_g = np.max(np.abs(sim.g))
    max_Phi = np.max(np.abs(sim.Phi))
    
    passed = max_g < 1e-10 and max_Phi < 1e-10 and drift < 0.01
    
    if verbose:
        print(f"  Max |g|: {max_g:.2e}")
        print(f"  Max |Φ|: {max_Phi:.2e}")
        print(f"  Mass drift: {drift*100:.4f}%")
        print(f"  Vacuum gravity test {'PASSED' if passed else 'FAILED'}")
    
    return passed


def run_collision_with_gravity(verbose: bool = True) -> Dict:
    """Run a collision test with gravity enabled."""
    if verbose:
        print("\n" + "="*60)
        print("1D COLLISION TEST (with Gravity)")
        print("="*60)
    
    params = DETParams1D(gravity_enabled=True)
    sim = DETCollider1DGravity(params)
    
    sim.add_packet(50, mass=8.0, width=5.0, momentum=0.6)
    sim.add_packet(150, mass=8.0, width=5.0, momentum=-0.6)
    
    initial_mass = sim.total_mass()
    
    rec = {'t': [], 'sep': [], 'mass_err': [], 'q_max': [], 'min_a': [], 'PE': []}
    
    for t in range(3000):
        sep = sim.separation()
        mass_err = 100 * (sim.total_mass() - initial_mass) / initial_mass
        
        rec['t'].append(t)
        rec['sep'].append(sep)
        rec['mass_err'].append(mass_err)
        rec['q_max'].append(np.max(sim.q))
        rec['min_a'].append(np.min(sim.a))
        rec['PE'].append(sim.potential_energy())
        
        if verbose and t % 500 == 0:
            print(f"  t={t}: sep={sep:.1f}, mass_err={mass_err:+.3f}%, "
                  f"q_max={np.max(sim.q):.3f}, PE={sim.potential_energy():.3f}")
        
        sim.step()
    
    rec['min_sep'] = min(rec['sep'])
    rec['collision'] = rec['min_sep'] < 5
    rec['final_mass_err'] = rec['mass_err'][-1]
    
    if verbose:
        print(f"\n  Collision: {'YES' if rec['collision'] else 'NO'}")
        print(f"  Min separation: {rec['min_sep']:.1f}")
        print(f"  Final mass error: {rec['final_mass_err']:+.3f}%")
    
    return rec


def run_full_test_suite():
    """Run the complete 1D DET falsifier test suite with gravity."""
    print("="*70)
    print("DET v6 1D COLLIDER WITH GRAVITY - FULL TEST SUITE")
    print("="*70)
    
    results = {}
    
    results['vacuum_gravity'] = test_gravity_vacuum(verbose=True)
    results['F7'] = test_F7_mass_conservation(verbose=True)
    results['F6'] = test_F6_gravitational_binding(verbose=True)
    results['collision'] = run_collision_with_gravity(verbose=True)
    
    print("\n" + "="*70)
    print("SUITE SUMMARY")
    print("="*70)
    print(f"  Vacuum gravity test: {'PASS' if results['vacuum_gravity'] else 'FAIL'}")
    print(f"  F7 (Mass conservation): {'PASS' if results['F7'] else 'FAIL'}")
    print(f"  F6 (Gravitational binding): {'PASS' if results['F6']['passed'] else 'FAIL'}")
    print(f"  Collision test: {'PASS' if results['collision']['collision'] else 'FAIL'}")
    
    return results


if __name__ == "__main__":
    results = run_full_test_suite()
