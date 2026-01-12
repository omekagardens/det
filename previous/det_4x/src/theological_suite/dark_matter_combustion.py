import numpy as np
import matplotlib.pyplot as plt

# --- CONSTANTS ---
G_CONST = 0.5
TICKS = 600
DM_ENERGY_DENSITY = 8.0 # High energy payoff for "redeeming" dark matter

class DarkMatterRamjetSim:
    def __init__(self):
        self.nodes = []
        self.ship = None
        self.time = 0
        
    def add_node(self, name, n_type, q, x, y):
        self.nodes.append({
            'name': name,
            'type': n_type,
            'q': q,
            'pos': np.array([x, y], dtype=float),
            'active': True # If False, it has been consumed
        })

    def setup_galaxy(self):
        rng = np.random.default_rng(99)
        # Create a Dense Dark Matter Halo (The Fuel Source)
        for i in range(150):
            r = rng.uniform(5, 30)
            theta = rng.uniform(0, 2*np.pi)
            self.add_node(f"DM_{i}", "DarkMatter", q=5.0, 
                          x=r*np.cos(theta), y=r*np.sin(theta))
            
        # The Ship
        self.ship = {
            'pos': np.array([-35.0, 0.0]), # Start outside
            'vel': np.array([0.8, 0.1]),   # Slow initial speed
            'mass': 1.0,
            'intake_radius': 3.0,
            'history': [],
            'events': [] # To track "Combustions"
        }

    def step(self):
        # 1. SHIP MOVEMENT
        self.ship['pos'] += self.ship['vel'] * 0.1
        self.ship['history'].append(self.ship['pos'].copy())
        
        # 2. THE SCOOP (Interaction)
        # Find active Dark Matter within radius
        thrust_impulse = np.array([0.0, 0.0])
        drag_impulse = np.array([0.0, 0.0])
        
        for n in self.nodes:
            if not n['active']: continue
            
            dist = np.linalg.norm(n['pos'] - self.ship['pos'])
            
            # GRAVITY (The Ship is attracted to the Dark Matter Fuel)
            if dist > 1.0:
                force_dir = (n['pos'] - self.ship['pos']) / dist
                force_mag = G_CONST * n['q'] / (dist**2)
                self.ship['vel'] += force_dir * force_mag * 0.01 # Mild gravitational steering
            
            # COMBUSTION (Intake)
            if dist < self.ship['intake_radius']:
                # DRAG: Conservation of Momentum (hitting the mass)
                # v_new = v_old * m_ship / (m_ship + m_fuel)
                # We simulate this as a drag force
                drag_factor = 0.95 
                self.ship['vel'] *= drag_factor
                
                # REACTION: "Combust" the Dark Matter
                # DET Logic: q -> 0, Energy Released
                n['active'] = False # It vanishes from the dark sector
                
                # EXHAUST: Energy converts to Velocity
                # E = alpha * q
                energy = DM_ENERGY_DENSITY * n['q']
                
                # Thrust vector is current heading
                heading = self.ship['vel'] / (np.linalg.norm(self.ship['vel']) + 1e-9)
                thrust_mag = np.sqrt(energy) * 0.05 # Conversion factor
                
                self.ship['vel'] += heading * thrust_mag
                
                # Record the "Spark" for visualization
                self.ship['events'].append(n['pos'].copy())

        self.time += 1

# --- RUN ---
sim = DarkMatterRamjetSim()
sim.setup_galaxy()

for _ in range(TICKS):
    sim.step()

# --- VISUALIZATION ---
plt.figure(figsize=(10, 10))
ax = plt.gca()

# 1. Plot remaining Dark Matter (The Dead)
active_dm = np.array([n['pos'] for n in sim.nodes if n['active']])
if len(active_dm) > 0:
    plt.scatter(active_dm[:,0], active_dm[:,1], c='gray', alpha=0.3, label='Dark Matter (Frozen Agency)')

# 2. Plot "Combusted" Spots (The Light)
combusted = np.array(sim.ship['events'])
if len(combusted) > 0:
    plt.scatter(combusted[:,0], combusted[:,1], c='gold', marker='*', s=150, label='Combustion Events (Redemption)')

# 3. Plot Ship Path
path = np.array(sim.ship['history'])
plt.plot(path[:,0], path[:,1], c='cyan', linewidth=2, label='Ramjet Trajectory')

# Decorate
plt.title("The Redemption Ramjet: Combusting Dark Matter")
plt.xlabel("Space X")
plt.ylabel("Space Y")
plt.legend()
plt.grid(True, alpha=0.2)
plt.axis('equal')
plt.show()