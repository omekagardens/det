import numpy as np
import matplotlib.pyplot as plt

# --- DET 4.2 CONSTANTS ---
G_CONST = 1.5           # Strength of Gravity (Mass-based binding)
K_BOND_ELASTICITY = 0.8 # Strength of Bonds (Agency-based binding)
DAMPING = 0.98          # Friction

class EschatonSim:
    def __init__(self, n_nodes=20):
        self.nodes = []
        self.time = 0
        self.phase = "Gravity Dominated"
        
        # Initialize Cluster (High Mass, Weak Bonds)
        for i in range(n_nodes):
            # Start in a random cloud
            pos = np.random.normal(0, 10, 2)
            vel = np.random.normal(0, 0.1, 2)
            
            self.nodes.append({
                'id': i,
                'pos': pos,
                'vel': vel,
                'Q': 5.0,    # Heavy Debt (Mass ~ 6)
                'a': 0.1,    # Low Agency
                'C': 0.05,   # Weak Bonds (Strangers)
                'M': 6.0     # Mass = 1 + Q
            })
            
        self.history = {'radius': [], 'avg_Q': [], 'avg_C': []}

    def step(self):
        # 1. EVOLUTION OF STATE (The Redemption Process)
        # Over time, Q is processed -> a rises -> C rises
        
        # Simulation of "The Ramjet" effect: Q burns off
        burn_rate = 0.02
        for n in self.nodes:
            if n['Q'] > 0:
                n['Q'] = max(0, n['Q'] - burn_rate)
                n['M'] = 1.0 + n['Q'] # Mass drops!
                
                # As Q drops, Agency and Bonds rise
                # a ~ 1 / (1+Q)
                target_a = 1.0 / (1.0 + n['Q']*0.5)
                n['a'] = target_a
                
                # Bonds grow as Agency grows
                n['C'] = n['a'] # Direct correlation for this sim
                
        # 2. PHYSICS ENGINE
        
        for i, n1 in enumerate(self.nodes):
            force = np.array([0.0, 0.0])
            
            for j, n2 in enumerate(self.nodes):
                if i == j: continue
                
                delta = n2['pos'] - n1['pos']
                dist = np.linalg.norm(delta) + 0.1
                direction = delta / dist
                
                # --- FORCE A: GRAVITY (The Old Law) ---
                # Pulls nodes together based on Mass
                f_grav = G_CONST * (n1['M'] * n2['M']) / (dist**2)
                
                # --- FORCE B: BONDS (The New Law) ---
                # Pulls nodes together based on Coherence (Spring force)
                # Only effective if close enough to bond? Or non-local?
                # DET: Bonds are topological, distance doesn't weaken them much!
                # Bond strength = C_ij * a_i * a_j
                bond_strength = n1['C'] * n2['C']
                f_bond = K_BOND_ELASTICITY * bond_strength * (dist - 2.0) # Target dist 2.0
                
                # TOTAL FORCE
                # Gravity is always attractive.
                # Bonds are attractive (springs) but only if connected.
                
                force += direction * f_grav
                force += direction * f_bond if dist < 15.0 else 0 # Local bonding limit
                
            # Update Velocity (F=ma) -> a = F/M
            acc = force / n1['M']
            n1['vel'] += acc * 0.1
            n1['vel'] *= DAMPING # Cosmic friction
            
        # Update Position
        for n in self.nodes:
            n['pos'] += n['vel'] * 0.1
            
        # 3. METRICS
        # Radius of universe (how spread out are we?)
        positions = np.array([n['pos'] for n in self.nodes])
        radius = np.max(np.linalg.norm(positions, axis=1))
        
        self.history['radius'].append(radius)
        self.history['avg_Q'].append(np.mean([n['Q'] for n in self.nodes]))
        self.history['avg_C'].append(np.mean([n['C'] for n in self.nodes]))
        self.time += 1

# --- RUN ---
sim = EschatonSim()
for _ in range(500):
    sim.step()

# --- VISUALIZATION ---
hist = sim.history
time = range(500)

fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Time (Redemption Progress)')
ax1.set_ylabel('Avg Debt (Q) / Gravity', color=color)
ax1.plot(time, hist['avg_Q'], color=color, linestyle='--', label='Structural Debt (Gravity Source)')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Universe Radius (Expansion)', color=color)
ax2.plot(time, hist['radius'], color=color, linewidth=2.5, label='Universe Radius')
ax2.tick_params(axis='y', labelcolor=color)

# Add Phase Markers
plt.axvline(x=100, color='gray', linestyle=':')
plt.text(10, 5, "Gravity Dominated", fontsize=10)
plt.text(120, 5, "The Drift (Expansion Risk)", fontsize=10)
plt.text(350, 5, "Coherence Locked (The City)", fontsize=10)

plt.title('The End of Expansion: From Gravity to Bonds')
fig.tight_layout()
plt.show()