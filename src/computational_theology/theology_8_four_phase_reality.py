import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Constants ---
GLOBAL_EVENTS = 1000  # Let it run long enough to see the asymptotes
DT = 0.1

class Agent:
    def __init__(self, name, alignment_strategy):
        self.name = name
        self.strategy = alignment_strategy 
        self.F = 0.0      
        self.chi = 0.0    
        self.psi = 1.0    
        self.P = 1.0      
        self.a = 1.0      
        
    def update(self, epoch):
        # 1. PHASE 1: GARDEN (Innocence)
        if epoch == 'GARDEN':
            self.F = 0.1; self.chi = 0.0; self.psi = 1.0; self.a = 1.0
            reservoir_limit = 0.0
            
        # 2. PHASE 2: FALL (The Burden Accumulates)
        elif epoch == 'FALL':
            self.F += 0.5 * DT   
            self.chi += 0.2 * DT 
            reservoir_limit = 2.0 
            self.psi = 0.5 
            self.a = 1.0
            
        # 3. PHASE 3: GRACE (The Permanent Reality)
        # There is no "Phase 4". This is it.
        elif epoch == 'GRACE':
            self.F += 0.2 * DT 
            reservoir_limit = 9999.0 # Infinite Source is open forever
            
            if self.strategy == 'Believer':
                # The Believer naturally evolves toward perfection
                self.psi = min(1.0, self.psi + 0.01) 
                self.a = 1.0
            else:
                # The Skeptic naturally evolves toward isolation
                self.psi = 0.1
                self.a = 0.0 
                self.chi += 0.2 * DT 

        # --- UNIVERSAL PHYSICS (No special "Utopia" math) ---
        flux = self.a * self.psi * (reservoir_limit - self.chi) * 0.1 * DT
        self.chi = max(0, self.chi - flux)
        
        # Mass Decoupling (The result of Alignment)
        effective_mass = self.F * (1.0 - self.psi)
        
        self.P = 1.0 / (1.0 + (0.1 * effective_mass) + self.chi)
        return self.P

# --- Run ---
believer = Agent("Aligned", "Believer")
skeptic = Agent("Hoarder", "Skeptic")

history_believer = []
history_skeptic = []

for k in range(GLOBAL_EVENTS):
    # ONLY 3 EPOCHS. 
    # Utopia is not a time; it is a destination.
    if k < 100: epoch = 'GARDEN'
    elif k < 300: epoch = 'FALL'
    else: epoch = 'GRACE' 
        
    history_believer.append(believer.update(epoch))
    history_skeptic.append(skeptic.update(epoch))

# --- Visualization ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(history_believer, color='gold', linewidth=3, label="Aligned (Reaches Utopia Naturally)")
ax.plot(history_skeptic, color='black', linewidth=3, label="Misaligned (Reaches Death Naturally)")

ax.axvline(x=300, color='green', linestyle='--', label="The Event (Resurrection)")

ax.text(10, 0.5, "GARDEN", fontsize=10)
ax.text(150, 0.2, "THE FALL", fontsize=10, color='red')
ax.text(400, 0.95, "THE KINGDOM (Now)", fontsize=10, color='gold')
ax.text(750, 0.1, "THE OUTER DARKNESS", fontsize=10, color='black')

ax.set_title("DET 3.0 Final: Utopia is an Emergent State, Not a Date")
ax.set_ylabel("Quality of Existence (P)")
ax.set_xlabel("Time")
ax.legend(loc='center left')
ax.grid(True)

plt.tight_layout()
plt.show()