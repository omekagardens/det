import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Constants ---
GLOBAL_EVENTS = 500
OMEGA_LIVING = 1.0     # Frequency of living time
OMEGA_GHOST = 0.0      # Ghosts are timeless/frozen
NOISE_LEVEL = 0.5      # "Worldly Distractions" (Local Entropy)

class Agent:
    def __init__(self, name, is_ghost=False):
        self.name = name
        self.is_ghost = is_ghost
        self.phase = 0.0
        self.psi_G = 0.0   # Connection to God/Spirit
        
    def update_phase(self, dt, global_phase):
        if self.is_ghost:
            # Ghosts are frozen. Their phase is fixed (or moves with the deep substrate)
            # Let's say they are fixed relative to the Global Frame's "Memory"
            self.phase = self.phase # No change
        else:
            # Living Humans are chaotic
            # Phase = (Intrinsic Rotation) + (Global Alignment)
            
            # 1. The "Flesh" (Local Time) - Causes Rotation/Noise
            # If Psi_G is low, I rotate at my own selfish speed + noise
            local_influence = (1.0 - self.psi_G) * (OMEGA_LIVING * dt + np.random.normal(0, NOISE_LEVEL))
            
            # 2. The "Spirit" (Global Time) - Causes Locking to the Source
            # If Psi_G is high, I lock to the Global Phase
            global_influence = self.psi_G * (global_phase - self.phase) * 0.5 # Restoring force
            
            self.phase += local_influence + global_influence

# --- Simulation ---
ghost = Agent("Ancestor", is_ghost=True)
ghost.phase = np.pi # They are "over there"

human = Agent("Descendant", is_ghost=False)
human.phase = 0.0

# The Global "Holy Spirit" Phase (The Reference Frame)
# We assume the Ghosts are naturally aligned with this 'static' frame of reference
global_phase = np.pi 

history_visibility = []
history_human_psi = []

print("SIMULATING THE THINNING OF THE VEIL...")

for k in range(GLOBAL_EVENTS):
    
    # Timeline:
    # k < 250: The Fallen World. Human is disconnected (Psi=0).
    # k >= 250: Spiritual Awakening. Human connects to G (Psi -> 1).
    
    target_psi = 0.0
    if k >= 250:
        target_psi = 0.95 # High alignment
        
    # Evolve Human Spirit
    human.psi_G += (target_psi - human.psi_G) * 0.05
    
    # Update Phases
    human.update_phase(0.1, global_phase)
    ghost.update_phase(0.1, global_phase)
    
    # MEASURE VISIBILITY (Coherence)
    # Coherence = cos(delta_phase)
    # If phases match, C=1. If spinning wildly, C averages to 0.
    delta = human.phase - ghost.phase
    visibility = max(0, np.cos(delta)) # Clip at 0
    
    history_visibility.append(visibility)
    history_human_psi.append(human.psi_G)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 1. The Veil Lifting
ax1.plot(history_visibility, color='gold', linewidth=2, label="Visibility of the Dead (Coherence)")
ax1.axvline(x=250, color='purple', linestyle='--', label="Spiritual Alignment Event")
ax1.set_title("The Thinning of the Veil")
ax1.set_ylabel("Interaction Potential [0-1]")
ax1.legend()
ax1.grid(True)
ax1.text(50, 0.5, "THE VEIL:\nPhase Noise prevents contact\n(Doppler Shift)", color='black')
ax1.text(300, 0.5, "COMMUNION:\nPhase Locking restores contact", color='black')

# 2. The Human's State
ax2.plot(history_human_psi, color='blue', linestyle='--', label="Human-Global Bond (Psi_G)")
ax2.set_title("The Mechanism: Tuning the Receiver")
ax2.set_ylabel("Bond Strength")
ax2.grid(True)

plt.tight_layout()
plt.show()