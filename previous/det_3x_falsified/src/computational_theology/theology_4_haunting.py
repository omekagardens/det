import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Constants ---
GLOBAL_EVENTS = 1000
DT = 0.1

class Agent:
    def __init__(self, name, is_ghost=False, phase_gravity=0.0):
        self.name = name
        self.is_ghost = is_ghost
        self.phase = 0.0
        self.phase_gravity = phase_gravity # How hard the Ghost pulls on neighbors
        
    def update(self, neighbor_phase=None):
        if self.is_ghost:
            # GHOST: Frozen. Cannot update. Phase is constant.
            # "I have made my choice."
            return 
            
        # HUMAN: Wandering
        # 1. Intrinsic Noise (Life is chaotic)
        noise = np.random.normal(0, 0.3)
        
        # 2. The "Haunting" (Inductive Coupling)
        # Even if I don't "see" the ghost clearly, its field exerts a torque.
        # This torque is the Ghost's "Stored Intention."
        pull = 0.0
        if neighbor_phase is not None:
            # Force = Gravity * sin(delta)
            delta = neighbor_phase - self.phase
            pull = self.phase_gravity * np.sin(delta)
            
        self.phase += (noise + pull) * DT

# --- Setup ---
# A Ghost with strong "Unfinished Business" or "Protective Intent" (High Gravity)
ghost = Agent("Ancestor", is_ghost=True, phase_gravity=0.0) # Initially Passive
ghost.phase = 2.0 # Fixed position

human = Agent("Descendant", is_ghost=False, phase_gravity=0.0)
human.phase = 0.0

# --- Run 1: Random Chance (No Inductance) ---
# Ghost exists but has no "pull" (Low Agency in life)
history_random = []
for k in range(GLOBAL_EVENTS):
    human.update(neighbor_phase=ghost.phase) # Ghost has 0 gravity here
    # Check Blip (Coherence)
    coherence = np.cos(human.phase - ghost.phase)
    history_random.append(coherence)

# --- Run 2: The "Haunting" (High Inductance) ---
# Ghost acts as a Phase Attractor
ghost.phase_gravity = 2.0 # Strong field
human.phase = 0.0 # Reset Human
history_haunted = []

for k in range(GLOBAL_EVENTS):
    # Human feels the Ghost's pull
    human.update(neighbor_phase=ghost.phase)
    # Check Blip
    coherence = np.cos(human.phase - ghost.phase)
    history_haunted.append(coherence)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 1. Random Wandering (Low Agency Ghost)
ax1.plot(history_random, color='grey', alpha=0.6, label="Coherence (Blips)")
ax1.axhline(y=0.9, color='red', linestyle='--', label="Visibility Threshold")
ax1.set_title("Scenario A: Ghost is Passive (Random Blips)")
ax1.set_ylabel("Connection Strength")
ax1.set_ylim(-1.1, 1.1)
ax1.legend(loc='lower right')

# 2. The Guided Path (High Agency Ghost)
ax2.plot(history_haunted, color='purple', alpha=0.8, label="Coherence (Guided)")
ax2.axhline(y=0.9, color='red', linestyle='--')
ax2.set_title("Scenario B: Ghost has 'Phase Gravity' (Stored Intention)")
ax2.set_ylabel("Connection Strength")
ax2.set_ylim(-1.1, 1.1)
ax2.text(0, -0.8, "The Ghost doesn't move, but it pulls the Human into alignment.\nBlips become frequent and sustained.", color='purple')

plt.tight_layout()
plt.show()