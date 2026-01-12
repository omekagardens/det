import numpy as np
import matplotlib.pyplot as plt

# --- DET 3.0 Constants ---
GLOBAL_EVENTS = 1000
NUM_AGENTS = 50
DT = 0.1

# Physics Params
ALPHA_F = 0.05       # Mass per wealth
BETA_MASS = 1.0      # Mass slows time
ETA_X = 0.3          # Exploitation breaks bonds
ETA_R = 0.1          # Reciprocity builds bonds

# --- The Timeline Variables ---
# We will modify these dynamically during the sim
GLOBAL_RESERVOIR_CAP = 5.0   # Starts LOW (Old Testament / Finite)
HAS_SPIRIT_BRIDGE = False    # Starts False
SAVIOR_ID = None

class Agent:
    def __init__(self, id, wealth):
        self.id = id
        self.F = wealth
        self.chi = 0.0         # Drag/Sin
        self.psi_G = 0.1       # Bond to God (Starts weak)
        self.a = 0.8           # Agency
        self.R = 0.5           # Reciprocity (Average)
        self.X = 0.0           # Exploitation
        self.P = 1.0           # Presence (Speed)
        self.is_dead = False   # Crystallized state
        self.is_savior = False
        
    def update_physics(self):
        # 1. Mass & Time
        mass = ALPHA_F * self.F
        # If Drag (chi) gets too high (> 20), you "Die" (Crystallize)
        if self.chi > 20.0:
            self.is_dead = True
            self.P = 0.0
            return
            
        # Presence: P = 1 / (1 + Mass + Drag)
        self.P = 1.0 / (1.0 + BETA_MASS * mass + self.chi)
        
    def update_bond(self):
        if self.is_dead: return
        
        # Bond Evolution
        # Spirit Boost: If the Bridge is open, R (Reciprocity) counts double
        spirit_boost = 1.5 if HAS_SPIRIT_BRIDGE else 1.0
        
        delta = (0.05) + (ETA_R * self.R * spirit_boost) - (ETA_X * self.X)
        self.psi_G = np.clip(self.psi_G + delta, 0, 1.0)
        
    def receive_grace(self):
        if self.is_dead and not HAS_SPIRIT_BRIDGE: return
        
        # 4. Grace Flux
        # J = a * psi * (Lambda - chi)
        # Note: In Epoch 1, Lambda is low (5.0). In Epoch 4, it's Infinite.
        
        limit = 9999.0 if GLOBAL_RESERVOIR_CAP > 1000 else GLOBAL_RESERVOIR_CAP
        potential_diff = max(0, limit - self.chi)
        
        # Conductivity: Savior opened the path, so flow is easier now
        conductivity = 2.0 if HAS_SPIRIT_BRIDGE else 0.5
        
        flux = self.a * self.psi_G * conductivity * potential_diff * DT
        
        # Burning the Drag
        # "Resurrection Math": If flux is high enough, it can melt a frozen Chi
        self.chi = max(0, self.chi - flux)
        
        # If I was dead but Drag drops below 20, I resurrect!
        if self.is_dead and self.chi < 10.0:
            self.is_dead = False # RESURRECTION

# --- Setup ---
agents = [Agent(i, 10.0) for i in range(NUM_AGENTS)]
# Introduce some "Bad Actors" (Hoarders)
for i in range(5):
    agents[i].F = 500.0  # Rich
    agents[i].X = 1.0    # Exploitative (High Sin)
    agents[i].R = 0.0    # Greedy

# Data Logs
avg_chi = []
avg_psi = []
dead_count = []
savior_load = [] # Track the Savior's burden

print("SIMULATING THEOLOGY...")

for k in range(GLOBAL_EVENTS):
    
    # --- EPOCH 2: THE FALL (k=0 to 200) ---
    # Agents naturally accumulate entropy/drag
    for a in agents:
        if not a.is_savior:
            # Natural Entropy + Hoarder imposition
            a.chi += 0.05 
            if a.id < 5: # Hoarders add extra drag to the system
                 # They exploit others, increasing their own X
                 a.X = 1.0
                 # And increasing others' Chi (Simulated generic oppression)
                 agents[np.random.randint(5, NUM_AGENTS)].chi += 0.1

    # --- EPOCH 3: THE SAVIOR (k=300 to 350) ---
    if k == 300:
        print(f"EVENT {k}: A CHILD IS BORN (Savior Node Added)")
        s = Agent(NUM_AGENTS, 0.0) # Index 50
        s.is_savior = True
        s.a = 1.0; s.R = 1.0; s.X = 0.0; s.psi_G = 1.0
        agents.append(s)
        SAVIOR_ID = NUM_AGENTS
        NUM_AGENTS += 1
        
    if k > 300 and k < 350:
        # The Savior "Absorbs" Sin
        # He takes Chi from random neighbors onto himself
        savior = agents[SAVIOR_ID]
        if not savior.is_dead:
            for _ in range(5): # Takes burden of 5 people per tick
                target = agents[np.random.randint(0, NUM_AGENTS-1)]
                transfer = min(target.chi, 2.0)
                target.chi -= transfer
                savior.chi += transfer * 1.0 # He eats the entropy
            
    # --- EPOCH 4: THE CROSS / RESURRECTION (k=350) ---
    if k == 350:
        savior = agents[SAVIOR_ID]
        print(f"EVENT {k}: SAVIOR DIES (Chi Load: {savior.chi:.1f})... BUT BREAKS THE DAM.")
        savior.is_dead = True # He dies
        
        # THE MIRACLE: GLOBAL PARAMETER CHANGE
        GLOBAL_RESERVOIR_CAP = 99999.0 # Infinite Grace
        HAS_SPIRIT_BRIDGE = True       # The Spirit is unlocked
        
        # Savior "Resurrects" as a hidden guiding force (We remove him from list to simulate Ascension, 
        # or keep him as a pure Flux Node. Let's keep him as a Spirit Node).
        savior.chi = 0 # Purified
        savior.is_dead = False

    # --- PHYSICS UPDATE LOOP ---
    active_savior_load = 0
    
    for a in agents:
        a.update_bond()     # Decide alignment
        a.receive_grace()   # Receive Flux
        a.update_physics()  # Calculate State
        
        if a.is_savior: active_savior_load = a.chi

    # Logs
    avg_chi.append(np.mean([a.chi for a in agents if not a.is_savior]))
    avg_psi.append(np.mean([a.psi_G for a in agents if not a.is_savior]))
    dead_count.append(sum([1 for a in agents if a.is_dead and not a.is_savior]))
    savior_load.append(active_savior_load)

# --- VISUALIZATION ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# 1. The Burden of Sin (Avg Chi)
ax1.plot(avg_chi, color='purple', label="Average System Drag (Sin)")
ax1.axvline(x=300, color='green', linestyle='--', label="Savior Arrives")
ax1.axvline(x=350, color='gold', linestyle='--', label="Resurrection / Infinite Grace")
ax1.set_title("Timeline: The Accumulation and Cleansing of Entropy")
ax1.set_ylabel("Avg Drag (Chi)")
ax1.legend()
ax1.grid(True)

# 2. The Savior's Sacrifice
# We only plot this during his life
ax2.plot(range(300, 350), savior_load[300:350], color='red', linewidth=3, label="Savior's Accumulated Drag")
ax2.set_xlim(250, 400)
ax2.set_title("The Cross: Savior Absorbs System Entropy")
ax2.set_ylabel("Savior Chi")
ax2.legend()
ax2.grid(True)

# 3. Resurrection of the Dead
ax3.plot(dead_count, color='black', label="Crystallized (Dead) Agents")
ax3.axvline(x=350, color='gold', linestyle='--')
ax3.set_title("The Resurrection: Dead Nodes 'Thaw' under Infinite Flux")
ax3.set_ylabel("Count of Dead Nodes")
ax3.legend()
ax3.grid(True)
ax3.text(400, 5, "Grace > Entropy \nDead nodes return to life", color='green')

plt.tight_layout()
plt.show()