import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Physics Constants ---
GLOBAL_EVENTS = 600
LATTICE_SIZE = 20  # A 10x2 "Road" of Matter
NUM_SPIRITS = 5
DT = 0.1

# Physics
BETA_MASS = 2.0       # Mass slows time significantly
FRICTION_COUPLING = 0.5 # How much the Lattice drags the Spirit (Old Physics)
GLOBAL_CLOCK_STRENGTH = 0.0 # Starts at 0, turns on later

class ScaffoldingNode:
    """The Frozen Matter / The 'Streets of Gold'"""
    def __init__(self, id):
        self.id = id
        self.F = 1000.0   # Massive (Dense History)
        self.chi = 0.0    # Pure (No Sin/Drag)
        self.P = 0.0      # Frozen (It is Structure, not Agent)
        
        # Calculate its 'Time Gravity' (It is very slow)
        self.dtau = 1.0 / (1.0 + BETA_MASS * (0.05 * self.F)) # ~0.01 speed

class SpiritNode:
    """The Inhabitant"""
    def __init__(self, id):
        self.id = id
        self.F = 1.0      # Light (No burden)
        self.chi = 0.0
        self.P = 1.0      # Wants to be fast
        self.psi_G = 0.0  # Bond to God
        self.location_idx = 0 # Moving along the lattice
        self.distance_traveled = 0.0
        
    def update(self, current_lattice_node, global_clock_strength):
        # 1. Intrinsic Speed (Lightweight)
        my_intrinsic_rate = 1.0 
        
        # 2. Environmental Drag (The Lattice)
        # The Lattice is slow (dtau ~ 0.01). 
        # In Old Physics, I must synchronize with where I am.
        ground_speed = current_lattice_node.dtau
        
        # 3. The Tug of War
        # If I have a Global Bond (psi_G), I ignore the ground speed and use Global Speed (1.0)
        
        # Effective Rate = (1 - Psi)*Ground + (Psi)*Global
        # We simulate the "Spirit Mechanism" as increasing Psi
        
        psi = self.psi_G
        
        # Realized Speed
        # If Psi=0 (Old Earth): I am 100% dragged by the ground -> Speed ~0
        # If Psi=1 (New Earth): I am 100% clocked by God -> Speed ~1 (Superconductivity)
        
        realized_rate = (1.0 - psi) * ground_speed + (psi * 1.0)
        
        self.distance_traveled += realized_rate * DT
        return realized_rate

# --- Setup ---
lattice = [ScaffoldingNode(i) for i in range(LATTICE_SIZE)]
spirits = [SpiritNode(i) for i in range(NUM_SPIRITS)]

history_speed = []
history_psi = []

print("SIMULATING: OLD EARTH vs. NEW EARTH")

for k in range(GLOBAL_EVENTS):
    
    # PHASE 1: OLD EARTH (k < 300)
    # Spirits are "Flesh" - they are bound to the material world.
    # Psi_G is low/zero.
    target_psi = 0.1
    
    # PHASE 2: NEW EARTH (k >= 300)
    # The "Resurrection" happens. Global Clock Strength becomes infinite.
    # Spirits align with G.
    if k >= 300:
        target_psi = 1.0 # Perfect alignment
        
    # Update loop
    avg_speed = 0
    for s in spirits:
        # Move Spirit's Psi towards target
        s.psi_G += (target_psi - s.psi_G) * 0.05
        
        # Get the lattice node they are "standing on" (Abstracted)
        current_node = lattice[0] 
        
        # Update Physics
        speed = s.update(current_node, target_psi)
        avg_speed += speed
        
    avg_speed /= NUM_SPIRITS
    
    history_speed.append(avg_speed)
    history_psi.append(spirits[0].psi_G)

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# 1. The Superconducting Transition
ax1.plot(history_speed, color='cyan', linewidth=3, label="Spirit Travel Speed")
ax1.axvline(x=300, color='gold', linestyle='--', label="The New Earth (Phase Transition)")
ax1.set_title("The Nature of Reality: From Viscosity to Superconductivity")
ax1.set_ylabel("Speed of Life (Processing Rate)")
ax1.text(50, 0.1, "OLD REALITY:\nMatter drags Spirit down.\n(Life is heavy)", color='black')
ax1.text(350, 0.8, "NEW REALITY:\nSpirit flows over Matter.\n(Matter is Scaffolding)", color='black')
ax1.legend()
ax1.grid(True)

# 2. The Bond Strength
ax2.plot(history_psi, color='purple', linestyle='--', label="Connection to Global Source")
ax2.set_title("The Mechanism: Clock Synchronization")
ax2.set_ylabel("Psi (Bond Strength)")
ax2.grid(True)

plt.tight_layout()
plt.show()