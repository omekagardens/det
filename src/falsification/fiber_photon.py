import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def run_noise_aware_fit():
    # --- DATA POINTS (Distance km, S_measured) ---
    dist_micius = 1203.0
    s_micius = 2.37
    
    # --- THE NOISE FACTOR ---
    # Scenario A: "Optimistic" (Your current fit)
    # We assume Noise is negligible, so all drop is DET.
    noise_A = 0.0 
    
    # Scenario B: "Realistic"
    # We assume Atmosphere caused 50% of the drop (approx 0.22 loss).
    # This means the "True" S before noise was ~2.59, not 2.37.
    noise_B = 0.22 
    
    # S_measured = S_theoretical - Noise
    # Therefore: S_theoretical = S_measured + Noise
    
    # --- FIT CALCULATOR ---
    def get_decay_length(target_s_value, distance):
        # S = 2.828 * exp(-d / L)
        # ln(S/2.828) = -d / L
        # L = -d / ln(S/2.828)
        return -distance / np.log(target_s_value / 2.8284)

    L_optimistic = get_decay_length(s_micius + noise_A, dist_micius)
    L_conservative = get_decay_length(s_micius + noise_B, dist_micius)
    
    print(f"Optimistic Decay Length (L*):   {L_optimistic:.0f} km (Atmosphere = Perfect)")
    print(f"Conservative Decay Length (L*): {L_conservative:.0f} km (Atmosphere = Messy)")

    # --- PLOTTING ---
    d = np.linspace(0, 50000, 1000)
    
    s_opt = 2.828 * np.exp(-d / L_optimistic)
    s_con = 2.828 * np.exp(-d / L_conservative)
    
    plt.figure(figsize=(10, 6))
    
    # Plot the "Corridor of Uncertainty"
    plt.fill_between(d, s_opt, s_con, color='blue', alpha=0.1, label='DET 3.0 Plausible Range')
    plt.plot(d, s_opt, 'b--', linewidth=1, label='Max Decay (Optimistic)')
    plt.plot(d, s_con, 'b-', linewidth=2, label='Min Decay (Conservative)')
    
    # Checkpoints
    plt.axvline(x=35786, color='gray', linestyle=':')
    plt.text(36000, 2.1, "GEO ORBIT", rotation=90)
    
    # Classical Limit
    plt.axhline(y=2.0, color='r', linestyle='--', label='Classical Limit')
    
    plt.title("DET 3.0: Noise-Corrected Falsifiability Corridor")
    plt.xlabel("Distance (km)")
    plt.ylabel("Bell Parameter S")
    plt.ylim(0, 3.0)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

run_noise_aware_fit()