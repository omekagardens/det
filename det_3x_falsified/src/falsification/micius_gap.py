import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- 1. THE REAL DATASET ---
# Format: (Distance_km, S_measured, Error_margin, "Label")
# We populate this with the "Hard Deck" data points we know.
# You can add new points here as you find them.
experiments = [
    (0.01,  2.828, 0.001, "Lab Ideal"),
    (143.0, 2.48,  0.28,  "Canary Is. (2012)"),  # High error bars in 2012
    (1203.0, 2.37,  0.09,  "Micius Sat (2017)")  # The Anchor
]

# Unpack for plotting
x_data = np.array([e[0] for e in experiments])
y_data = np.array([e[1] for e in experiments])
y_err  = np.array([e[2] for e in experiments])

# --- 2. DEFINE THE DET 3.0 MODEL ---
# S(d) = S_max * exp(-d / L_decay)
def det_decay_model(d, l_decay):
    S_MAX = 2.8284 # 2*sqrt(2)
    return S_MAX * np.exp(-d / l_decay)

# --- 3. FIT THE MODEL TO REALITY ---
# We ask Scipy: "What L_decay best explains these data points?"
# We provide an initial guess (p0) of 7000 km.
popt, pcov = curve_fit(det_decay_model, x_data, y_data, p0=[7000.0])

# Extract the calculated "Real" Decay Length
best_fit_L = popt[0]
perr = np.sqrt(np.diag(pcov)) # Standard deviation of the fit

print(f"--- DATA FITTING RESULTS ---")
print(f"Optimized DET Decay Length (L*): {best_fit_L:.2f} km")
print(f"Uncertainty (+/-):               {perr[0]:.2f} km")
print(f"------------------------------")

# --- 4. VISUALIZATION ---
plt.figure(figsize=(12, 7))

# A. Plot the Standard QM Prediction (Flat Line)
x_range = np.linspace(0, 40000, 1000) # Up to GEO orbit
plt.plot(x_range, [2.828]*len(x_range), 'k--', alpha=0.4, label='Standard QM (Invariant)')

# B. Plot the Fitted DET 3.0 Curve
y_fit = det_decay_model(x_range, best_fit_L)
plt.plot(x_range, y_fit, 'b-', linewidth=2, label=f'DET 3.0 Fit (L*={best_fit_L:.0f} km)')

# C. Plot the Classical Limit
plt.axhline(y=2.0, color='r', linestyle=':', label='Classical Limit (S=2)')

# D. Plot the Experimental Data Points
for (d, s, err, lbl) in experiments:
    plt.errorbar(d, s, yerr=err, fmt='o', capsize=5, label=lbl)
    # Annotate points
    plt.annotate(f"{lbl}\nS={s}", (d, s), xytext=(d+500, s+0.1), 
                 arrowprops=dict(arrowstyle="->", color='gray'))

# E. Highlight Falsification Zones
plt.axvline(x=35786, color='purple', linestyle='-.', alpha=0.5)
geo_prediction = det_decay_model(35786, best_fit_L)
plt.text(32000, 0.5, f"GEO ORBIT\nPredicted S={geo_prediction:.2f}\n(Classical)", color='purple')

# F. Styling
plt.title(f"DET 3.0: Data-Driven Decay Fit (L* = {best_fit_L:.0f} km)", fontsize=14)
plt.xlabel("Distance (km)", fontsize=12)
plt.ylabel("Bell Parameter S", fontsize=12)
plt.legend(loc='lower left')
plt.grid(True, alpha=0.2)
plt.ylim(0, 3.2)

plt.tight_layout()
plt.show()