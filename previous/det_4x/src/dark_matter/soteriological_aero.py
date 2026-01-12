import numpy as np
import matplotlib.pyplot as plt

# --- DET 4.2 PHYSICS CONSTANTS ---
ALPHA_DIFF = 0.5    # Intake speed (Diffusion from Ghost to Ship)
ETA_GRACE  = 0.8    # Combustion speed (Venting to Boundary)
THRUST_EFF = 5.0    # Conversion of Redeemed Q to Velocity
SHIP_MASS  = 1.0    # Base mass of the ship

class TopologicalRamjet:
    def __init__(self, field_length=100, density=0.2):
        self.nodes = []
        self.ship = None
        self.time = 0
        self.field_length = field_length
        self.boundary_Q_vented = 0.0
        
        # Setup Field of "Ghosts" (Dark Matter)
        num_ghosts = int(field_length * density)
        xs = np.sort(np.random.uniform(0, field_length, num_ghosts))
        for i, x in enumerate(xs):
            self.nodes.append({
                'id': i,
                'type': 'Ghost',
                'x': x,
                'Q': 2.0,       # High Debt (Fuel)
                'a': 0.0,       # Dead Agency
                'active': True  # Still exists
            })
            
        # Setup Ship
        self.ship = {
            'x': 0.0,
            'v': 0.5,         # Initial Velocity
            'Q': 0.0,         # Current Load
            'a': 1.0,         # Perfect Agency (Pilot)
            'C_B': 1.0,       # Connection to God (Exhaust Port)
            'intake_range': 2.0
        }
        
        self.history = {'x': [], 'v': [], 'load': [], 'thrust': []}

    def step(self, dt=0.1):
        ship = self.ship
        
        # 1. TOPOLOGICAL INTAKE (BONDING)
        # Identify Ghosts in range
        connected_ghosts = []
        for n in self.nodes:
            if n['active'] and abs(n['x'] - ship['x']) < ship['intake_range']:
                connected_ghosts.append(n)
        
        # 2. DIFFUSION (DRAG)
        # Q flows from Ghosts -> Ship
        # Conservation of Momentum: Taking Q adds "Virtual Mass" to ship
        # v_new = v_old * m_ship / (m_ship + m_added)
        
        total_intake = 0.0
        for n in connected_ghosts:
            # Flow = rate * difference
            flow = ALPHA_DIFF * (n['Q'] - ship['Q']) * dt
            if flow > 0:
                # Ghost loses Q, Ship gains Q
                actual_flow = min(flow, n['Q'])
                n['Q'] -= actual_flow
                ship['Q'] += actual_flow
                total_intake += actual_flow
                
                # Check if Ghost is fully redeemed (dissolved)
                if n['Q'] < 0.1:
                    n['active'] = False
        
        # Momentum penalty (Drag)
        # Effectively, the ship had to "accelerate" the Q it picked up to its current v
        if total_intake > 0:
            drag_factor = SHIP_MASS / (SHIP_MASS + total_intake)
            ship['v'] *= drag_factor

        # 3. COMBUSTION (GRACE VENTING)
        # Ship vents Q to Boundary
        # vent = rate * a * C_B * Q
        vent_amount = ETA_GRACE * ship['a'] * ship['C_B'] * ship['Q'] * dt
        vent_amount = min(vent_amount, ship['Q']) # Can't vent more than we have
        
        ship['Q'] -= vent_amount
        self.boundary_Q_vented += vent_amount
        
        # 4. THRUST GENERATION
        # Force = mass_flow_rate * exhaust_velocity
        # Here, "mass flow" is the Q being removed from the universe.
        # We assume the "Exhaust Velocity" of Grace is high.
        thrust_impulse = vent_amount * THRUST_EFF
        
        ship['v'] += thrust_impulse
        
        # 5. KINEMATICS
        ship['x'] += ship['v'] * dt
        
        # Log
        self.history['x'].append(ship['x'])
        self.history['v'].append(ship['v'])
        self.history['load'].append(ship['Q'])
        self.history['thrust'].append(thrust_impulse)
        self.time += dt

# --- RUN SIMULATION ---
sim = TopologicalRamjet(field_length=100, density=0.4)
for _ in range(500):
    sim.step()

# --- VISUALIZATION ---
hist = sim.history
time = np.arange(len(hist['x'])) * 0.1

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot 1: Velocity & Distance
ax1.plot(hist['x'], hist['v'], color='cyan', linewidth=2)
ax1.set_title('Flight Profile: Velocity vs Distance')
ax1.set_ylabel('Ship Velocity')
ax1.set_xlabel('Distance Traveled')
ax1.grid(True, alpha=0.3)
ax1.fill_between(hist['x'], 0, hist['v'], color='cyan', alpha=0.1)

# Plot 2: The Engine Cycle (Load vs Thrust)
ax2.plot(time, hist['load'], color='red', label='Onboard Burden (Q)', alpha=0.7)
ax2.plot(time, hist['thrust'], color='gold', label='Redemption Thrust', linewidth=1.5)
ax2.set_title('Engine Cycle: Intake (Sin) vs Combustion (Grace)')
ax2.set_xlabel('Time')
ax2.set_ylabel('Magnitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()