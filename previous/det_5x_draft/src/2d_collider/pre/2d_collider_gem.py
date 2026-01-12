"""
DET v5 2D Collider Suite (v2 Update)
====================================
Ported and adapted from the 1D React/Python v2 logic to 2D.

Fixes Applied (2D Adaptation):
1. 2D Periodic Local Sum (using convolution with wrap mode)
2. 2D Phase Gradient Initialization (Wave vector k)
3. Decay factor clamped (prevent sign-flip on both axes)
4. Generalized 2D Conservative Limiter (4-neighbor outflow check)
5. 2D Robust Separation (Torus-aware Center of Mass)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import convolve, label
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import time

# =============================================================================
# MATH UTILITIES (2D TORUS)
# =============================================================================

def periodic_dist_2d(pos1: np.ndarray, pos2: np.ndarray, shape: Tuple[int, int]) -> float:
    """Calculate Euclidean distance on a torus."""
    dx = np.abs(pos1[0] - pos2[0])
    dy = np.abs(pos1[1] - pos2[1])
    
    if dx > shape[0] / 2:
        dx = shape[0] - dx
    if dy > shape[1] / 2:
        dy = shape[1] - dy
        
    return np.sqrt(dx**2 + dy**2)

def periodic_com_2d(mask: np.ndarray, weights: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Compute Center of Mass on a 2D torus using circular mean for X and Y separately.
    """
    if np.sum(weights) < 1e-9:
        return np.array([shape[0]/2, shape[1]/2])

    indices = np.argwhere(mask)
    masses = weights[mask]
    total_mass = np.sum(masses)
    
    # Process Y (0th axis)
    y_angles = 2 * np.pi * indices[:, 0] / shape[0]
    y_cos = np.sum(masses * np.cos(y_angles)) / total_mass
    y_sin = np.sum(masses * np.sin(y_angles)) / total_mass
    y_angle = np.arctan2(y_sin, y_cos)
    if y_angle < 0: y_angle += 2 * np.pi
    com_y = y_angle * shape[0] / (2 * np.pi)

    # Process X (1st axis)
    x_angles = 2 * np.pi * indices[:, 1] / shape[1]
    x_cos = np.sum(masses * np.cos(x_angles)) / total_mass
    x_sin = np.sum(masses * np.sin(x_angles)) / total_mass
    x_angle = np.arctan2(x_sin, x_cos)
    if x_angle < 0: x_angle += 2 * np.pi
    com_x = x_angle * shape[1] / (2 * np.pi)

    return np.array([com_y, com_x])

# =============================================================================
# PARAMETERS
# =============================================================================

@dataclass
class DETParams2D:
    # Grid
    Ny: int = 100
    Nx: int = 100
    DT: float = 0.02
    
    # Physics
    F_VAC: float = 0.01
    R: int = 4  # Local normalization radius
    
    # Phase
    nu: float = 0.1
    omega_0: float = 0.0
    
    # Modules
    mom_enabled: bool = True
    alpha_pi: float = 0.1
    lambda_pi: float = 0.01
    # mu_pi: float = 0.3   # [REMOVED: replaced below]
    pi_max: float = 3.0

    # Agency-gated diffusion (DET5 collider requirement)
    diff_gate_power: float = 2.0      # higher => stronger suppression when C is high
    diff_floor: float = 0.03          # minimum classical diffusion fraction

    # Phase-driven momentum (avoid "momentum = lagged diffusion")
    alpha_pi_phase: float = 0.6       # momentum drive from sin(dtheta)
    alpha_pi_diff: float = 0.05       # small remaining drive from diff flux (stability)

    # Momentum transport strength (advection)
    mu_pi: float = 0.8                # override previous default 0.3 for visible translation

    coh_alpha: float = 0.05
    coh_gamma: float = 2.0
    coh_lambda: float = 0.002
    c_min: float = 0.05
    
    q_enabled: bool = True
    alpha_q: float = 0.02
    
    # Numerics
    outflow_limit: float = 0.25

# =============================================================================
# ENGINE
# =============================================================================

class DETEngine2D:
    def __init__(self, params: DETParams2D):
        self.p = params
        self.shape = (params.Ny, params.Nx)
        
        # --- STATE ARRAYS ---
        # Node variables
        self.F = np.ones(self.shape) * params.F_VAC
        self.q = np.zeros(self.shape)
        self.theta = np.zeros(self.shape)
        self.a = np.ones(self.shape)
        
        # Bond variables (Approx: Store X-right and Y-down components on nodes)
        # pi_x[i,j] is momentum on bond (i,j) -> (i, j+1)
        # pi_y[i,j] is momentum on bond (i,j) -> (i+1, j)
        self.pi_x = np.zeros(self.shape)
        self.pi_y = np.zeros(self.shape)
        
        # Coherence (Directional)
        self.C_x = np.ones(self.shape) * params.c_min
        self.C_y = np.ones(self.shape) * params.c_min
        
        self.sigma = np.ones(self.shape)
        self.P = np.ones(self.shape)
        self.Delta_tau = np.ones(self.shape) * params.DT
        
        self.time = 0.0
        self.step_count = 0
        
        # Optimization: Kernel for local sum
        # A square kernel of 1s with size (2R+1)
        d = 2 * params.R + 1
        self.local_sum_kernel = np.ones((d, d))

    def add_gaussian(self, center_y: float, center_x: float, amp: float, width: float):
        """Add Gaussian blob to F."""
        y = np.arange(self.shape[0])
        x = np.arange(self.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Distance on torus
        dy = np.abs(Y - center_y)
        dy = np.where(dy > self.shape[0]/2, self.shape[0] - dy, dy)
        
        dx = np.abs(X - center_x)
        dx = np.where(dx > self.shape[1]/2, self.shape[1] - dx, dx)
        
        dist_sq = dx**2 + dy**2
        envelope = np.exp(-0.5 * dist_sq / (width**2))
        
        self.F += amp * envelope
        
        # Boost coherence locally
        self.C_x = np.minimum(1.0, self.C_x + 0.7 * envelope)
        self.C_y = np.minimum(1.0, self.C_y + 0.7 * envelope)

    def add_momentum_via_phase(self, center_y, center_x, vel_y, vel_x, width):
        """
        FIX #2: Initialize momentum by setting a phase gradient.
        theta ~ k_x * x + k_y * y
        """
        y = np.arange(self.shape[0])
        x = np.arange(self.shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Localize the gradient to the packet
        dy = np.abs(Y - center_y)
        dy = np.where(dy > self.shape[0]/2, self.shape[0] - dy, dy)
        dx = np.abs(X - center_x)
        dx = np.where(dx > self.shape[1]/2, self.shape[1] - dx, dx)
        dist_sq = dx**2 + dy**2
        envelope = np.exp(-0.5 * dist_sq / (width**2))
        
        # Linear phase gradient localized by envelope
        # Note: We must handle periodic coordinates for the phase ramp carefully.
        # Simple approximation: just add v*dist from center, respecting sign
        
        # Simpler: just iterate (slower but safer for prototype)
        # Or vectorised:
        dx_signed = X - center_x
        # Wrap signed distance
        dx_signed = np.where(dx_signed > self.shape[1]/2, dx_signed - self.shape[1], dx_signed)
        dx_signed = np.where(dx_signed < -self.shape[1]/2, dx_signed + self.shape[1], dx_signed)

        dy_signed = Y - center_y
        dy_signed = np.where(dy_signed > self.shape[0]/2, dy_signed - self.shape[0], dy_signed)
        dy_signed = np.where(dy_signed < -self.shape[0]/2, dy_signed + self.shape[0], dy_signed)

        self.theta += (vel_x * dx_signed + vel_y * dy_signed) * envelope
        self.theta = np.mod(self.theta, 2*np.pi)

    def step(self):
        p = self.p
        dt = p.DT
        
        # 1. Presence & Local Time
        self.P = self.a / (1.0 + self.F)
        self.Delta_tau = self.P * dt
        
        # 2. Wavefunction & Flow
        
        # FIX #1: Periodic Local Sum (2D Convolution with wrap)
        F_local = convolve(self.F, self.local_sum_kernel, mode='wrap') + 1e-9
        amp = np.sqrt(np.clip(self.F / F_local, 0, 1))
        
        # Rolls for neighbors
        # Right (x+1), Left (x-1), Down (y+1), Up (y-1)
        amp_E = np.roll(amp, -1, axis=1) # East (i, j+1)
        amp_W = np.roll(amp, 1, axis=1)  # West (i, j-1)
        amp_S = np.roll(amp, -1, axis=0) # South (i+1, j)
        amp_N = np.roll(amp, 1, axis=0)  # North (i-1, j)
        
        th = self.theta
        th_E = np.roll(th, -1, axis=1)
        th_W = np.roll(th, 1, axis=1)
        th_S = np.roll(th, -1, axis=0)
        th_N = np.roll(th, 1, axis=0)
        
        F_E = np.roll(self.F, -1, axis=1)
        F_W = np.roll(self.F, 1, axis=1)
        F_S = np.roll(self.F, -1, axis=0)
        F_N = np.roll(self.F, 1, axis=0)
        
        # --- X FLUXES ---
        # DET5: Decompose phase and classical transport, with coherence gating
        dth_E = np.angle(np.exp(1j * (th_E - th)))
        quant_E = amp * amp_E * np.sin(dth_E)
        class_E = self.F - F_E

        # DET5: Coherence favors phase-transport and suppresses classical equalization
        Cx = np.clip(self.C_x, p.c_min, 1.0)
        diff_frac_x = np.maximum(p.diff_floor, (1.0 - Cx) ** p.diff_gate_power)
        phase_frac_x = Cx

        J_phase_x = (self.sigma * phase_frac_x) * quant_E
        J_class_x = (self.sigma * diff_frac_x) * class_E
        J_diff_x = J_phase_x + J_class_x

        J_mom_x = np.zeros_like(self.F)
        if p.mom_enabled:
            # Conservative-ish advection: flow proportional to momentum on the bond and local carried mass
            F_avg_E = 0.5 * (self.F + F_E)
            J_mom_x = p.mu_pi * self.pi_x * F_avg_E

        J_x = J_diff_x + J_mom_x # Flow from (i,j) -> (i, j+1)

        # --- Y FLUXES ---
        dth_S = np.angle(np.exp(1j * (th_S - th)))
        quant_S = amp * amp_S * np.sin(dth_S)
        class_S = self.F - F_S

        Cy = np.clip(self.C_y, p.c_min, 1.0)
        diff_frac_y = np.maximum(p.diff_floor, (1.0 - Cy) ** p.diff_gate_power)
        phase_frac_y = Cy

        J_phase_y = (self.sigma * phase_frac_y) * quant_S
        J_class_y = (self.sigma * diff_frac_y) * class_S
        J_diff_y = J_phase_y + J_class_y

        J_mom_y = np.zeros_like(self.F)
        if p.mom_enabled:
            F_avg_S = 0.5 * (self.F + F_S)
            J_mom_y = p.mu_pi * self.pi_y * F_avg_S

        J_y = J_diff_y + J_mom_y # Flow from (i,j) -> (i+1, j)

        # --- FIX #5: Generalized 2D Conservative Limiter ---
        # Calculate total outflow from each cell
        # Outflows:
        # 1. To East: max(0, J_x[i,j])
        # 2. To West: max(0, -J_x[i,j-1]) -> need roll
        # 3. To South: max(0, J_y[i,j])
        # 4. To North: max(0, -J_y[i-1,j])
        
        J_x_prev = np.roll(J_x, 1, axis=1) # Flow coming in from West
        J_y_prev = np.roll(J_y, 1, axis=0) # Flow coming in from North
        
        out_E = np.maximum(0, J_x)
        out_W = np.maximum(0, -J_x_prev)
        out_S = np.maximum(0, J_y)
        out_N = np.maximum(0, -J_y_prev)
        
        total_out = out_E + out_W + out_S + out_N
        max_out = p.outflow_limit * self.F / dt
        
        # Avoid div by zero
        scale = np.minimum(1.0, max_out / (total_out + 1e-9))
        
        # Apply scaling to flows originating from this cell
        # Note: J_x[i,j] is flow OUT of i,j to East. J_x[i,j-1] is flow OUT of i,j-1 to i,j.
        # We need to scale J_x based on the source cell's scale factor.
        
        # J_x is flow (i,j)->(i,j+1). Source is (i,j). Scale by scale[i,j] if J_x>0.
        # If J_x<0, it's inflow to (i,j) from (i,j+1), so it's outflow from (i,j+1). Scale by scale[i,j+1].
        
        scale_E = np.roll(scale, -1, axis=1)
        scale_S = np.roll(scale, -1, axis=0)
        
        # Update J_x
        # Positive J_x means flow i,j -> i,j+1. Source i,j. Use scale[i,j].
        # Negative J_x means flow i,j+1 -> i,j. Source i,j+1. Use scale[i,j+1].
        mask_pos_x = J_x > 0
        J_x = np.where(mask_pos_x, J_x * scale, J_x * scale_E)
        if p.mom_enabled:
            # Scale diff components for momentum update consistency
            J_diff_x = np.where(mask_pos_x, J_diff_x * scale, J_diff_x * scale_E)
            
        # Update J_y
        mask_pos_y = J_y > 0
        J_y = np.where(mask_pos_y, J_y * scale, J_y * scale_S)
        if p.mom_enabled:
            J_diff_y = np.where(mask_pos_y, J_diff_y * scale, J_diff_y * scale_S)
            
        # Hard clamp
        J_x = np.clip(J_x, -20, 20)
        J_y = np.clip(J_y, -20, 20)
        J_diff_x = np.clip(J_diff_x, -20, 20)
        J_diff_y = np.clip(J_diff_y, -20, 20)

        # 3. Resource Update
        # Div(J) = (J_x - J_x_prev) + (J_y - J_y_prev)
        # But J_x is flow out of cell to right.
        # dF = Inflow - Outflow
        # Inflow = J_x[i, j-1] + J_y[i-1, j] (flows entering)
        # Outflow = J_x[i, j] + J_y[i, j] (flows leaving)
        
        J_x_in = np.roll(J_x, 1, axis=1)
        J_y_in = np.roll(J_y, 1, axis=0)
        
        dF = (J_x_in + J_y_in) - (J_x + J_y)
        F_new = np.clip(self.F + dF * dt, p.F_VAC, 1000)
        
        # [Inside step(), replacing the Flux Calculation section]

        # 1. Define Vacuum Mask (Where F is close to vacuum)
        # 0.0 in vacuum, 1.0 inside matter
        # Steep sigmoid to sharply cut off drag outside the body
        vac_threshold = p.F_VAC * 5.0
        matter_mask = np.clip((self.F - vac_threshold) / (p.F_VAC), 0, 1)

        # 2. Conductivity (Must be NON-ZERO in vacuum for Gravity/Momentum to work)
        # We allow a baseline conductivity even in vacuum
        self.sigma = 1.0 + 0.1 * np.log(1.0 + np.abs(J_E) + np.abs(J_S))
        
        # 3. Calculate Diffusive Flux (The "Drag")
        # Standard calculation...
        drive_E = np.sqrt(self.C_E) * quantum_E + (1 - np.sqrt(self.C_E)) * classical_E
        drive_S = np.sqrt(self.C_S) * quantum_S + (1 - np.sqrt(self.C_S)) * classical_S
        drive_W = np.sqrt(W(self.C_E)) * quantum_W + (1 - np.sqrt(W(self.C_E))) * classical_W
        drive_N = np.sqrt(Nb(self.C_S)) * quantum_N + (1 - np.sqrt(Nb(self.C_S))) * classical_N
        
        # APPLY MASK ONLY HERE: Friction turns off in vacuum
        J_diff_E = gE * cond_E * drive_E * matter_mask
        J_diff_S = gS * cond_S * drive_S * matter_mask
        J_diff_W = gW * cond_W * drive_W * matter_mask * np.roll(matter_mask, 1, axis=1)
        J_diff_N = gN * cond_N * drive_N * matter_mask * np.roll(matter_mask, 1, axis=0)

        # 4. Momentum & Gravity (UNMASKED)
        # They can travel through vacuum freely
        J_mom_E = J_mom_W = J_mom_S = J_mom_N = 0
        if p.momentum_enabled:
            # ... (Existing momentum code) ...
            # Ensure calculating logic matches previous provided snippet
            F_avg_E = 0.5 * (self.F + E(self.F))
            J_mom_E = p.mu_pi * self.sigma * self.pi_E * F_avg_E
            # ... etc for W, S, N

        J_grav_E = J_grav_W = J_grav_S = J_grav_N = 0
        if p.gravity_enabled:
            # ... (Existing gravity code) ...
            J_grav_E = p.mu_grav * self.sigma * F_avg_E * (self.Phi - E(self.Phi))
            # ... etc for W, S, N

        # 5. Floor Flux (Existing code)
        
        # 6. Total Sum
        J_E = J_diff_E + J_mom_E + J_floor_E + J_grav_E
        # ... etc

        # Prefer coherent "phase transport" over raw magnitude; penalize pure classical diffusion
        J_mag = np.sqrt(J_x**2 + J_y**2)
        phase_mag = np.sqrt((J_phase_x)**2 + (J_phase_y)**2)
        class_mag = np.sqrt((J_class_x)**2 + (J_class_y)**2)
        drive = alpha_eff * (phase_mag - 0.25 * class_mag)
        drive = np.clip(drive, -0.5, 5.0)

        self.C_x += (drive - p.coh_lambda * self.C_x) * dt
        self.C_y += (drive - p.coh_lambda * self.C_y) * dt
        self.C_x = np.clip(self.C_x, p.c_min, 1.0)
        self.C_y = np.clip(self.C_y, p.c_min, 1.0)
        
        # 8. Conductivity
        self.sigma = 1.0 + 0.1 * np.log(1.0 + J_mag)
        
        self.time += dt
        self.step_count += 1

    # --- ANALYSIS ---

    def get_blobs(self) -> List[Dict]:
        """FIX #4: Robust 2D Blob Detection."""
        threshold = self.p.F_VAC * 10
        mask = self.F > threshold
        
        # Handle periodic boundaries for labeling? 
        # Scipy doesn't support periodic labeling natively.
        # Pad array with 1 pixel border to check connections
        padded = np.pad(mask, 1, mode='wrap')
        labeled_padded, n_feats = label(padded)
        
        # This is complex to resolve fully in short code. 
        # Simpler approach: Standard labeling, then merge if edge blobs touch
        labeled, n_feats = label(mask)
        
        blobs = []
        for i in range(1, n_feats + 1):
            b_mask = (labeled == i)
            total_mass = np.sum(self.F[b_mask])
            com = periodic_com_2d(b_mask, self.F, self.shape)
            blobs.append({'mass': total_mass, 'com': com})
            
        blobs.sort(key=lambda x: x['mass'], reverse=True)
        return blobs

# =============================================================================
# SIMULATION RUNNER
# =============================================================================

def run_2d_collision():
    print("Initializing DET v5 2D Collider (v2)...")
    
    # Setup: Head-on collision in X
    params = DETParams2D(Ny=80, Nx=120)
    engine = DETEngine2D(params)
    
    cy, cx = 40, 60
    sep = 40
    
    # Left blob moving Right
    engine.add_gaussian(cy, cx - sep/2, 6.0, 5.0)
    engine.add_momentum_via_phase(cy, cx - sep/2, 0, 0.6, 15.0) # vy=0, vx=0.6
    
    # Right blob moving Left
    engine.add_gaussian(cy, cx + sep/2, 6.0, 5.0)
    engine.add_momentum_via_phase(cy, cx + sep/2, 0, -0.6, 15.0) # vy=0, vx=-0.6
    
    print("Starting simulation loop...")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Field Plot
    im_f = ax1.imshow(engine.F, cmap='magma', vmin=params.F_VAC, vmax=10, origin='lower')
    ax1.set_title("Resource (F)")
    
    # Momentum Quiver (self-normalized for visibility)
    # Subsample for clarity
    step = 4
    Y, X = np.mgrid[0:params.Ny:step, 0:params.Nx:step]
    U0 = engine.pi_x[::step, ::step]
    V0 = engine.pi_y[::step, ::step]
    mag0 = np.sqrt(U0**2 + V0**2)
    denom0 = np.percentile(mag0, 95) + 1e-12
    U0n = U0 / denom0
    V0n = V0 / denom0

    quiver = ax2.quiver(
        X, Y, U0n, V0n,
        angles='xy', scale_units='xy', scale=1.0,
        color='cyan', pivot='mid', minlength=0.1
    )
    ax2.set_facecolor('black')
    ax2.set_title("Momentum Flow")
    
    frame_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, color='white')

    def update(frame):
        # Multiple physics steps per frame
        for _ in range(5):
            engine.step()

        blobs = engine.get_blobs()
        sep_dist = 0.0
        if len(blobs) >= 2:
            sep_dist = periodic_dist_2d(blobs[0]['com'], blobs[1]['com'], engine.shape)

        im_f.set_data(engine.F)

        # Update Quiver (normalize by robust percentile so arrows remain visible)
        U = engine.pi_x[::step, ::step]
        V = engine.pi_y[::step, ::step]
        mag = np.sqrt(U**2 + V**2)
        denom = np.percentile(mag, 95) + 1e-12
        Un = U / denom
        Vn = V / denom
        quiver.set_UVC(Un, Vn)

        p95_pi = float(np.percentile(mag, 95)) if mag.size else 0.0
        frame_text.set_text(
            f"Step: {engine.step_count}\nSep: {sep_dist:.1f}\nMax F: {np.max(engine.F):.1f}\npi p95: {p95_pi:.3e}"
        )
        return im_f, quiver, frame_text

    ani = FuncAnimation(fig, update, frames=200, interval=50, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_2d_collision()