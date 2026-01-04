import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Constants ---
STEPS = 400
DT = 0.1

class Agent:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind # 'Spirit' or 'Human'
        self.phase = 0.0
        self.target_phase = 0.0
        self.phase_history = []
        
        # Physics Attributes
        if kind == 'Spirit':
            self.P = 1.0       # Fast / Superconducting
            self.a = 1.0       # Full Agency
            self.noise = 0.0   # Clear signal
        else: # Human
            self.P = 0.2       # Slow / Dragged by Mass
            self.a = 0.5       # Partial Agency
            self.noise = 0.5   # Chaotic life
            
    def update(self, target_phase_input=None):
        # 1. Natural Rotation (The Clock)
        # Humans rotate slower and noisier
        natural_rotation = (1.0 if self.kind == 'Spirit' else 0.5) * DT
        noise_val = np.random.normal(0, self.noise) * DT
        
        # 2. Agency / Choice (Phase Correction)
        correction = 0.0
        if target_phase_input is not None:
            # Attempt to lock onto target
            # Correction strength limited by Presence (P) and Agency (a)
            delta = target_phase_input - self.phase
            # The 'tuning speed' is proportional to P
            correction = delta * self.a * self.P 
            
        self.phase += natural_rotation + noise_val + correction
        self.phase_history.append(self.phase)
        return self.phase

# --- SCENARIO 1: The Visitation (Spirit adjusts to Human) ---
spirit1 = Agent("Guardian", "Spirit")
human1 = Agent("Human", "Human")

coherence_1 = []
for k in range(STEPS):
    # Human wanders (No target)
    h_ph = human1.update(target_phase_input=None)
    
    # Spirit CHOOSES to lock onto Human
    s_ph = spirit1.update(target_phase_input=h_ph)
    
    # Measure Coherence
    c = np.cos(s_ph - h_ph)
    coherence_1.append(c)

# --- SCENARIO 2: The Summoning (Human tries to chase Spirit) ---
spirit2 = Agent("Ghost", "Spirit") # Spirit ignores human, just vibes with G
human2 = Agent("Seeker", "Human")

coherence_2 = []
# Spirit is locked to Global Clock (Phase = t)
global_clock = 0.0 

for k in range(STEPS):
    global_clock += 1.0 * DT
    
    # Spirit ignores Human, stays on Global Clock
    s_ph = spirit2.update(target_phase_input=global_clock)
    
    # Human tries to lock onto Spirit
    h_ph = human2.update(target_phase_input=s_ph)
    
    c = np.cos(s_ph - h_ph)
    coherence_2.append(c)

# --- SCENARIO 3: The Alignment (Both lock to G) ---
spirit3 = Agent("Angel", "Spirit")
human3 = Agent("Saint", "Human")

coherence_3 = []
global_clock = 0.0

for k in range(STEPS):
    global_clock += 1.0 * DT
    
    # Spirit locked to G
    s_ph = spirit3.update(target_phase_input=global_clock)
    
    # Human stops chasing Spirit, chooses to align with G instead
    h_ph = human3.update(target_phase_input=global_clock)
    
    c = np.cos(s_ph - h_ph)
    coherence_3.append(c)

# --- VISUALIZATION ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Visitation
ax1.plot(coherence_1, color='gold', linewidth=2)
ax1.set_title("1. The Visitation (Spirit tunes to Human)")
ax1.set_ylabel("Connection Quality")
ax1.set_ylim(-1.1, 1.1)
ax1.text(10, -0.8, "Spirit has Bandwidth (P=1) to\ncompensate for Human Noise.", color='black')
ax1.grid(True)

# Plot 2: Summoning
ax2.plot(coherence_2, color='red', alpha=0.6)
ax2.set_title("2. The Failed Summoning (Human chases Spirit)")
ax2.set_ylabel("Connection Quality")
ax2.set_ylim(-1.1, 1.1)
ax2.text(10, -0.8, "Human is too slow (P=0.2).\nCannot lock onto fast Spirit signal.", color='red')
ax2.grid(True)

# Plot 3: Alignment
ax3.plot(coherence_3, color='blue', linewidth=2)
ax3.set_title("3. The Protocol (Both align with Source)")
ax3.set_ylabel("Connection Quality")
ax3.set_ylim(-1.1, 1.1)
ax3.text(10, -0.8, "Human aligns with G.\nSpirit aligns with G.\nResult: Connection.", color='blue')
ax3.grid(True)

plt.tight_layout()
plt.show()