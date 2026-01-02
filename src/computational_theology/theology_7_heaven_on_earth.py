import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Constants ---
GLOBAL_EVENTS = 500
DT = 0.1
HELL_DRAG_RATE = 0.5 # The world is stressful

class Agent:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy # 'Worldly' or 'Mystic'
        self.P = 1.0        # Presence (Quality of Life)
        self.chi = 0.0      # Drag (Stress/Sin)
        self.psi_G = 0.1    # Connection to Source
        self.mass = 10.0    # Resources
        
    def update(self):
        # 1. THE WORLD ATTACKS
        # Entropy pours in from the environment
        self.chi += HELL_DRAG_RATE * DT
        
        # 2. THE RESPONSE
        if self.strategy == 'Worldly':
            # Strategy: Use Mass/Ego to fight. 
            # "I will work harder! I will accumulate more!"
            # Result: No Grace Flux. You rely on your buffer.
            # (Psi stays low)
            pass
            
        elif self.strategy == 'Mystic':
            # Strategy: Tune to Global Sync.
            # "I surrender to the Flow."
            self.psi_G += 0.05 * DT # Bond grows
            self.psi_G = min(1.0, self.psi_G)
            
            # LIBERATION FLUX (The "Utopia Now" Mechanic)
            # Flux = Psi * Infinite_Source
            flux = self.psi_G * 5.0 * DT 
            
            # The Flux burns the Drag
            self.chi = max(0, self.chi - flux)
            
        # 3. CALCULATE REALITY (Presence)
        # P = 1 / (1 + Mass + Drag)
        self.P = 1.0 / (1.0 + (0.1*self.mass) + self.chi)
        
        return self.P

# --- Run ---
worldly = Agent("Worldly", "Worldly")
mystic = Agent("Mystic", "Mystic")

history_worldly = []
history_mystic = []

for k in range(GLOBAL_EVENTS):
    p_w = worldly.update()
    p_m = mystic.update()
    
    history_worldly.append(p_w)
    history_mystic.append(p_m)

# --- Visualize ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(history_worldly, color='grey', label="The Worldly (Attached to Matter)")
ax.plot(history_mystic, color='gold', linewidth=3, label="The Mystic (Tuned to G)")

ax.set_title("Living in Utopia vs. Dystopia (Same Environment)")
ax.set_ylabel("Quality of Existence (Presence P)")
ax.set_xlabel("Time")
ax.legend()
ax.grid(True)
ax.text(200, 0.8, "The 'Kingdom' is a Vertical State,\nnot a Horizontal Place.", color='gold', fontsize=12)

plt.tight_layout()
plt.show()