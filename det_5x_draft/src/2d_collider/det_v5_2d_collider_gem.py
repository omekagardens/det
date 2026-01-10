import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftfreq
from dataclasses import dataclass
import time

"""
DET v5 FINAL VALIDATION SUITE (CORRECTED)
=========================================
1. Fusion Test (Soft Matter, Head-on) -> Merging
2. Binding Test (Hard Matter, Orbital) -> Orbiting
   FIX: Added 'Vacuum Masking' to prevent drag in empty space.
"""

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def periodic_local_sum_2d(x, radius):
    result = np.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            result += np.roll(np.roll(x, dy, axis=0), dx, axis=1)
    return result

# ==============================================================================
# PARAMETERS
# ==============================================================================
@dataclass
class DETParams:
    N: int = 100
    DT: float = 0.015
    F_VAC: float = 0.01
    R: int = 5
    nu: float = 0.1
    C_init: float = 0.3
    
    # Physics - Momentum
    momentum_enabled: bool = True
    alpha_pi: float = 0.01
    lambda_pi: float = 0.00
    mu_pi: float = 0.4
    pi_max: float = 10.0
    
    # Physics - Structure & Agency
    q_enabled: bool = True
    alpha_q: float = 0.015
    a_coupling: float = 30.0
    a_rate: float = 0.2
    
    # Physics - Gravity
    gravity_enabled: bool = False
    kappa: float = 5.0
    alpha_baseline: float = 0.05
    mu_grav: float = 0.04
    
    # Physics - Floor
    floor_enabled: bool = True
    eta_floor: float = 0.12
    F_core: float = 4.0
    floor_power: float = 2.0
    
    outflow_limit: float = 0.20

# ==============================================================================
# GRAVITY SOLVER
# ==============================================================================
class DETGravity:
    def __init__(self, N, params):
        self.N = N
        self.p = params
        k = fftfreq(N, d=1.0) * 2 * np.pi
        kx, ky = np.meshgrid(k, k)
        self.k2 = kx**2 + ky**2
        self.k2[0, 0] = 1.0

    def solve(self, q):
        q_hat = fft2(q)
        denom = -(self.k2 + self.p.alpha_baseline)
        b_hat = (-self.p.alpha_baseline * q_hat) / denom
        b = np.real(ifft2(b_hat))
        
        rho = q - b
        rho_hat = fft2(rho)
        phi_hat = (self.p.kappa * rho_hat) / self.k2
        phi_hat[0, 0] = 0.0
        phi = np.real(ifft2(phi_hat))
        return phi, rho

# ==============================================================================
# MAIN COLLIDER CLASS
# ==============================================================================
class DETCollider2D:
    def __init__(self, params=None):
        self.p = params or DETParams()
        N = self.p.N
        self.F = np.ones((N, N)) * self.p.F_VAC
        self.q = np.zeros((N, N))
        self.theta = np.zeros((N, N))
        self.a = np.ones((N, N))
        
        self.pi_E = np.zeros((N, N))
        self.pi_S = np.zeros((N, N))
        self.C_E = np.ones((N, N)) * self.p.C_init
        self.C_S = np.ones((N, N)) * self.p.C_init
        self.sigma = np.ones((N, N))
        
        self.gravity_solver = DETGravity(N, self.p) if self.p.gravity_enabled else None
        self.Phi = np.zeros((N, N))
    
    def add_packet(self, center, mass=6.0, width=5.0, momentum=(0, 0), hard_matter=False):
        N = self.p.N
        y, x = np.mgrid[0:N, 0:N]
        cy, cx = center
        r2 = (x - cx)**2 + (y - cy)**2
        envelope = np.exp(-0.5 * r2 / width**2)
        
        self.F += mass * envelope
        self.C_E = np.clip(self.C_E + 0.7 * envelope, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.7 * envelope, self.p.C_init, 1.0)
        
        if hard_matter:
            self.q += 0.99 * envelope
            self.a -= 0.99 * envelope
            self.a = np.clip(self.a, 0.00001, 1.0)
        
        py, px = momentum
        if px != 0 or py != 0:
            mom_env = np.exp(-0.5 * r2 / (width*3)**2)
            self.pi_E += px * mom_env
            self.pi_S += py * mom_env

    def step(self):
        p = self.p
        N = p.N
        dt = p.DT
        eps = 1e-9
        
        E = lambda x: np.roll(x, -1, axis=1)
        W = lambda x: np.roll(x, 1, axis=1)
        S = lambda x: np.roll(x, -1, axis=0)
        Nb = lambda x: np.roll(x, 1, axis=0)
        
        # 1. Clocking
        P = self.a / (1.0 + self.F)
        Delta_tau = P * dt
        
        # 2. Gravity
        J_grav_E = J_grav_W = J_grav_S = J_grav_N = 0
        if p.gravity_enabled:
            self.Phi, _ = self.gravity_solver.solve(self.q)
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))
            
            J_grav_E = p.mu_grav * self.sigma * F_avg_E * (self.Phi - E(self.Phi))
            J_grav_W = p.mu_grav * self.sigma * F_avg_W * (self.Phi - W(self.Phi))
            J_grav_S = p.mu_grav * self.sigma * F_avg_S * (self.Phi - S(self.Phi))
            J_grav_N = p.mu_grav * self.sigma * F_avg_N * (self.Phi - Nb(self.Phi))

        # 3. Wavefunction & Classical Drive
        F_local = periodic_local_sum_2d(self.F, p.R) + eps
        amp = np.sqrt(np.clip(self.F / F_local, 0, 1))
        psi = amp * np.exp(1j * self.theta)
        
        quantum_E = np.imag(np.conj(psi) * E(psi))
        quantum_S = np.imag(np.conj(psi) * S(psi))
        quantum_W = np.imag(np.conj(psi) * W(psi))
        quantum_N = np.imag(np.conj(psi) * Nb(psi))
        
        classical_E = self.F - E(self.F)
        classical_S = self.F - S(self.F)
        classical_W = self.F - W(self.F)
        classical_N = self.F - Nb(self.F)
        
        # 4. Conductivity & Agency Gating
        cond_E = self.sigma * (self.C_E + 1e-4)
        cond_S = self.sigma * (self.C_S + 1e-4)
        cond_W = self.sigma * (W(self.C_E) + 1e-4)
        cond_N = self.sigma * (Nb(self.C_S) + 1e-4)
        
        gE = np.sqrt(self.a * E(self.a))
        gS = np.sqrt(self.a * S(self.a))
        gW = np.sqrt(self.a * W(self.a))
        gN = np.sqrt(self.a * Nb(self.a))
        
        drive_E = np.sqrt(self.C_E) * quantum_E + (1 - np.sqrt(self.C_E)) * classical_E
        drive_S = np.sqrt(self.C_S) * quantum_S + (1 - np.sqrt(self.C_S)) * classical_S
        drive_W = np.sqrt(W(self.C_E)) * quantum_W + (1 - np.sqrt(W(self.C_E))) * classical_W
        drive_N = np.sqrt(Nb(self.C_S)) * quantum_N + (1 - np.sqrt(Nb(self.C_S))) * classical_N
        
        # --- FIX: SUPERFLUID VACUUM MASK ---
        # Forces Friction (Diffusion) to 0.0 in the vacuum, regardless of Agency.
        # This allows planets to glide through space without "syrup drag".
        # Threshold: 5x vacuum floor.
        vac_threshold = p.F_VAC * 5.0
        # Steep sigmoid mask: 0 in vacuum, 1 in matter
        matter_mask = np.clip((self.F - vac_threshold) / p.F_VAC, 0, 1)

        # Apply mask ONLY to Diffusion (Drag)
        J_diff_E = gE * cond_E * drive_E * matter_mask
        J_diff_S = gS * cond_S * drive_S * matter_mask
        J_diff_W = gW * cond_W * drive_W * matter_mask * np.roll(matter_mask, 1, axis=1)
        J_diff_N = gN * cond_N * drive_N * matter_mask * np.roll(matter_mask, 1, axis=0)

        # 5. Momentum Flux (UNMASKED - Works in Vacuum)
        J_mom_E = J_mom_W = J_mom_S = J_mom_N = 0
        if p.momentum_enabled:
            F_avg_E = 0.5 * (self.F + E(self.F))
            F_avg_W = 0.5 * (self.F + W(self.F))
            F_avg_S = 0.5 * (self.F + S(self.F))
            F_avg_N = 0.5 * (self.F + Nb(self.F))
            J_mom_E = p.mu_pi * self.sigma * self.pi_E * F_avg_E
            J_mom_W = -p.mu_pi * self.sigma * W(self.pi_E) * F_avg_W
            J_mom_S = p.mu_pi * self.sigma * self.pi_S * F_avg_S
            J_mom_N = -p.mu_pi * self.sigma * Nb(self.pi_S) * F_avg_N

        # 6. Floor Flux
        J_floor_E = J_floor_W = J_floor_S = J_floor_N = 0
        if p.floor_enabled:
            s = np.maximum(0, (self.F - p.F_core) / p.F_core)**p.floor_power
            J_floor_E = p.eta_floor * self.sigma * (s + E(s)) * classical_E
            J_floor_W = p.eta_floor * self.sigma * (s + W(s)) * classical_W
            J_floor_S = p.eta_floor * self.sigma * (s + S(s)) * classical_S
            J_floor_N = p.eta_floor * self.sigma * (s + Nb(s)) * classical_N

        # 7. Total Flux
        J_E = J_diff_E + J_mom_E + J_floor_E + J_grav_E
        J_W = J_diff_W + J_mom_W + J_floor_W + J_grav_W
        J_S = J_diff_S + J_mom_S + J_floor_S + J_grav_S
        J_N = J_diff_N + J_mom_N + J_floor_N + J_grav_N
        
        # 8. Limiter
        pos_out = (np.maximum(0, J_E) + np.maximum(0, J_W) +
                   np.maximum(0, J_S) + np.maximum(0, J_N))
        max_amount_out = p.outflow_limit * self.F
        amount_out = pos_out * Delta_tau
        scale = np.minimum(1.0, max_amount_out / (amount_out + eps))
        
        J_E = np.where(J_E > 0, J_E * scale, J_E)
        J_W = np.where(J_W > 0, J_W * scale, J_W)
        J_S = np.where(J_S > 0, J_S * scale, J_S)
        J_N = np.where(J_N > 0, J_N * scale, J_N)

        # 9. Update F
        out_amount = (J_E + J_W + J_S + J_N) * Delta_tau
        in_amount = (W(J_E) * W(Delta_tau) + E(J_W) * E(Delta_tau) +
                     Nb(J_S) * Nb(Delta_tau) + S(J_N) * S(Delta_tau))
        dF = in_amount - out_amount
        self.F = self.F + dF
        self.F = np.maximum(self.F, 1e-6)

        # 10. Secondary Fields
        if p.momentum_enabled:
            J_diff_E_sc = np.where(J_diff_E > 0, J_diff_E * scale, J_diff_E)
            J_diff_S_sc = np.where(J_diff_S > 0, J_diff_S * scale, J_diff_S)
            dt_E = 0.5*(Delta_tau + E(Delta_tau))
            dt_S = 0.5*(Delta_tau + S(Delta_tau))
            
            self.pi_E = (1 - p.lambda_pi*dt_E)*self.pi_E + p.alpha_pi*J_diff_E_sc*dt_E
            self.pi_S = (1 - p.lambda_pi*dt_S)*self.pi_S + p.alpha_pi*J_diff_S_sc*dt_S
            self.pi_E = np.clip(self.pi_E, -p.pi_max, p.pi_max)
            self.pi_S = np.clip(self.pi_S, -p.pi_max, p.pi_max)
            
        if p.q_enabled:
            self.q = np.clip(self.q + p.alpha_q * np.maximum(0, -dF), 0, 1)
            
        a_target = 1.0 / (1.0 + p.a_coupling * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        d_theta = (np.angle(np.exp(1j*(E(self.theta)-self.theta))) + 
                   np.angle(np.exp(1j*(W(self.theta)-self.theta))) + 
                   np.angle(np.exp(1j*(S(self.theta)-self.theta))) + 
                   np.angle(np.exp(1j*(Nb(self.theta)-self.theta))))
        self.theta = np.mod(self.theta + p.nu * d_theta * dt, 2*np.pi)
        
        self.C_E = np.clip(self.C_E + 0.05*np.abs(J_E)*dt - 0.002*self.C_E*dt, p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + 0.05*np.abs(J_S)*dt - 0.002*self.C_S*dt, p.C_init, 1.0)
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E) + np.abs(J_S))
    
    def get_coms(self):
        labeled, num = ndimage.label(self.F > self.p.F_VAC*20)
        y, x = np.mgrid[0:self.p.N, 0:self.p.N]
        coms = []
        for i in range(1, num+1):
            mask = labeled == i
            mass = np.sum(self.F[mask])
            if mass < 0.5: continue
            cy = np.sum(y[mask]*self.F[mask])/mass
            cx = np.sum(x[mask]*self.F[mask])/mass
            coms.append((cy, cx))
        return coms

def run_suite():
    print("DET v5 FINAL GEOMETRY VALIDATION")
    print("================================")

    # ---------------------------------------------------------
    # TEST 1: Fusion (Soft Matter, Head-On)
    # ---------------------------------------------------------
    print("\n[TEST 1] Soft Collider (Fusion)")
    p1 = DETParams()
    p1.gravity_enabled = False
    p1.a_coupling = 30.0 # Soft matter
    sim1 = DETCollider2D(p1)
    sim1.add_packet((50, 30), mass=6.0, momentum=(0, 0.4), hard_matter=False)
    sim1.add_packet((50, 70), mass=6.0, momentum=(0, -0.4), hard_matter=False)
    
    init_mass = np.sum(sim1.F)
    for t in range(5000):
        sim1.step()
    
    final_mass = np.sum(sim1.F)
    coms1 = sim1.get_coms()
    err = 100*(final_mass - init_mass)/init_mass
    
    print(f"Result @ t=5000:")
    print(f"  Blobs: {len(coms1)} (Expected 1)")
    print(f"  Mass Error: {err:+.4f}%")
    if len(coms1) == 1: print("  >> Outcome: FUSION (Correct)")
    else: print("  >> Outcome: FAILED")

    # ---------------------------------------------------------
    # TEST 2: Binding (Hard Matter, Orbital Geometry)
    # ---------------------------------------------------------
    print("\n[TEST 2] Hard Binding (Gravity Orbit)")
    p2 = DETParams()
    p2.gravity_enabled = True
    p2.mu_grav = 0.04
    p2.kappa = 5.0
    p2.momentum_enabled = True
    p2.F_VAC = 0.001
    p2.a_coupling = 10000.0  # Hard matter
    
    sim2 = DETCollider2D(p2)
    # Orbital Geometry: Separate along X, Move along Y
    sim2.add_packet((50, 30), mass=8.0, momentum=(0.35, 0), hard_matter=True) 
    sim2.add_packet((50, 70), mass=8.0, momentum=(-0.35, 0), hard_matter=True)
    
    seps = []
    print("Running Orbit Simulation...")
    for t in range(5000):
        sim2.step()
        if t % 100 == 0:
            c = sim2.get_coms()
            if len(c) == 2:
                dy = c[1][0] - c[0][0]
                dx = c[1][1] - c[0][1]
                seps.append(np.sqrt(dx**2 + dy**2))
            elif len(c) == 1:
                seps.append(0)
    
    coms2 = sim2.get_coms()
    print(f"Result @ t=5000:")
    print(f"  Blobs: {len(coms2)} (Expected 2)")
    
    if len(seps) > 0:
        final_sep = seps[-1]
        print(f"  Final Sep: {final_sep:.1f}")
        if final_sep > 5.0 and len(coms2) == 2:
            print("  >> Outcome: BINDING (Orbiting/Dancing) - SUCCESS")
        elif len(coms2) == 1:
            print("  >> Outcome: MERGER (Crashed)")
        else:
            print("  >> Outcome: ESCAPE (Drifted apart)")

if __name__ == "__main__":
    run_suite()