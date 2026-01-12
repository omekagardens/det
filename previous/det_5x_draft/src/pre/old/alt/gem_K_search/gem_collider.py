import numpy as np
import matplotlib.pyplot as plt

"""
DET v5 - DISSIPATIVE CAPTURE MODEL
==================================
Goal: Resolve the Neutron Anomaly by modeling capture as Kinetic Energy Dissipation.

The "Brake" Hypothesis:
- In classical collisions, particles scatter elastically. Capture is impossible without energy loss.
- EM Interaction (Phase Friction) provides a massive dissipation channel for P-P.
- Strong Force (Gamma Emission) provides a smaller channel for P-N.

Implementation:
1. Conservative Forces: EM (Repulsion) + Strong (Attraction).
2. Dissipative Forces: 
   - EM Drag: Scales with Phase Mismatch (High for P-P).
   - Nuclear Drag: Scales with Density (Baseline for all).
3. Fusion Condition: Total Energy (K + V) < 0.
"""

# Physics Parameters
MASS = 1.0
EM_STRENGTH = 2.0
STRONG_STRENGTH = 200.0
EM_RANGE = 10.0
STRONG_RANGE = 0.5

# Dissipation Parameters (The "Kappa" factors)
DISS_EM = 0.5       # Energy loss due to Phase Friction (P-P)
DISS_NUCLEAR = 0.05 # Energy loss due to Radiative Capture (Gamma analog) - P-N relies on this

DT = 0.005
STEPS = 40000

class Particle:
    def __init__(self, x, v, phase, is_charged=True):
        self.x = x
        self.v = v
        self.phase = phase
        self.is_charged = is_charged

def compute_potential(r, p1, p2):
    # EM Potential (Coulomb-like with phase factor)
    if p1.is_charged and p2.is_charged:
        phase_diff = p2.phase - p1.phase
        phase_factor = np.sin(phase_diff / 2) ** 2
        # Integral of 1/(r+0.5)^2ish approximation or just standard potential form
        # Using V ~ exp(-r)/r form to match force
        V_em = EM_STRENGTH * phase_factor * np.exp(-r / EM_RANGE) / (r + 0.5)
    else:
        V_em = 0.0
        
    # Strong Potential (Yukawa-like well)
    # Force was ~ exp(-r/R) / r. Potential is similar.
    # Potential is -V0 * exp(-r/R) / r
    V_strong = -STRONG_STRENGTH * np.exp(-r / STRONG_RANGE) / (r + 0.1)
    
    return V_em + V_strong

def run_dissipative_collision(p1_charged, p2_charged, energy, diss_nuclear_factor=1.0):
    separation = 50.0
    v_init = np.sqrt(2 * energy / MASS)
    
    p1 = Particle(x=-separation/2, v=v_init, phase=0.0, is_charged=p1_charged)
    p2 = Particle(x=separation/2, v=-v_init, phase=np.pi if p2_charged else 0.0, 
                  is_charged=p2_charged)
    
    fused = False
    min_dist = separation
    capture_counter = 0
    
    for _ in range(STEPS):
        r = abs(p2.x - p1.x)
        r = max(r, 0.1)
        direction = 1 if p1.x < p2.x else -1
        
        # 1. Calculate Conservative Forces
        if p1.is_charged and p2.is_charged:
            phase_diff = p2.phase - p1.phase
            phase_factor = np.sin(phase_diff / 2) ** 2
            F_em = EM_STRENGTH * phase_factor * np.exp(-r / EM_RANGE) / (r + 0.5)
        else:
            F_em = 0.0
            phase_factor = 0.0
            
        density = np.exp(-r / STRONG_RANGE)
        F_strong = -STRONG_STRENGTH * density / (r + 0.1)
        
        F_cons = (F_em + F_strong) * direction
        
        # 2. Calculate Dissipation (The "Brake")
        # Relative velocity
        v_rel = p2.v - p1.v
        
        # Dissipation Coefficient eta
        # Component A: EM/Phase Friction (Active only if charged)
        eta_em = DISS_EM * phase_factor * density # Dissipates inside interaction zone
        
        # Component B: Nuclear Friction (Gamma emission analog)
        # Active for everyone inside strong zone
        eta_nuc = DISS_NUCLEAR * diss_nuclear_factor * density
        
        eta_total = eta_em + eta_nuc
        
        # Dissipative Force: F_diss = -eta * v_rel
        # Acts to reduce relative velocity
        # F_drag = +eta * v_rel_vector
        F_drag = eta_total * v_rel # v_rel is (v2 - v1). If v2<v1 (closing), v_rel < 0. Drag < 0 (pushes p1 left). Correct.
        
        # Total Force
        F1 = F_cons + F_drag
        F2 = -F_cons - F_drag # Newton's 3rd
        
        # Update Kinematics
        a1 = F1 / MASS
        a2 = F2 / MASS
        
        p1.v += a1 * DT
        p2.v += a2 * DT
        p1.x += p1.v * DT
        p2.x += p2.v * DT
        
        min_dist = min(min_dist, r)
        
        # 3. Check Bound State Condition
        # Total Energy = Kinetic + Potential
        v_rel_now = p2.v - p1.v
        K_rel = 0.5 * (MASS/2) * v_rel_now**2 # Reduced mass = M/2
        V_pot = compute_potential(r, p1, p2)
        E_tot = K_rel + V_pot
        
        if E_tot < -0.1: # Bound state (negative energy)
            capture_counter += 1
            if capture_counter > 50: # Persisted for 50 steps
                fused = True
                break
        else:
            capture_counter = 0
            
        if r > separation * 1.2: # Escaped
            break
            
    return fused

def find_capture_window(p1_charged, p2_charged, nuclear_factor):
    """
    Find the energy range where fusion occurs.
    Returns (E_min, E_max).
    """
    # Sweep energies
    energies = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0]
    fused_energies = []
    
    for E in energies:
        if run_dissipative_collision(p1_charged, p2_charged, E, nuclear_factor):
            fused_energies.append(E)
            
    if not fused_energies:
        return 0.0, 0.0
    
    return min(fused_energies), max(fused_energies)

def run_dissipation_test():
    print(f"{'Collision':<15} | {'Nuclear Diss (K_nuc)':<20} | {'Fusion Window (E)':<20} | {'Result'}")
    print("-" * 75)
    
    # 1. Reference: Proton-Proton
    # Has Strong EM Dissipation + Standard Nuclear Dissipation
    emin_pp, emax_pp = find_capture_window(True, True, 1.0)
    print(f"{'P-P':<15} | {'1.0 (Standard)':<20} | {f'{emin_pp:.1f} - {emax_pp:.1f}':<20} | {'Baseline'}")
    
    # 2. Neutron Test: Proton-Neutron
    # Has NO EM Dissipation.
    # Case A: Standard Nuclear Dissipation (Is it enough?)
    emin_pn, emax_pn = find_capture_window(True, False, 1.0)
    print(f"{'P-N':<15} | {'1.0 (Standard)':<20} | {f'{emin_pn:.1f} - {emax_pn:.1f}':<20} | {'Check Capture'}")
    
    # Case B: Enhanced Nuclear Dissipation (Simulating Gamma resonance?)
    # If P-N fails in Case A, we increase K_nuc to see if we can restore capture.
    emin_pn_high, emax_pn_high = find_capture_window(True, False, 5.0)
    print(f"{'P-N (High K)':<15} | {'5.0 (Enhanced)':<20} | {f'{emin_pn_high:.1f} - {emax_pn_high:.1f}':<20} | {'Enhanced Capture'}")

if __name__ == "__main__":
    run_dissipation_test()