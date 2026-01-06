import numpy as np
import matplotlib.pyplot as plt

# --- DET 4.2: The "Body" Update ---
# Transitive Presence: Bond to Spirit counts as Boundary Access
TRANSITIVE_ADOPTION = True 

class BodyOfChristSim:
    def __init__(self, population_size):
        self.N = population_size
        self.nodes = []
        self.time = 0
        self.Q_total_load = 10.0 # A fixed "World Burden" to process
        
    def run(self, ticks=200):
        # 1. SETUP
        # Christ/Spirit (The Head)
        christ_P = 1.0
        
        # The Body (Humans)
        # Initial: Burdened (Q distributed), Moderate Agency
        # If N is small, q_i is high. If N is large, q_i is low.
        q_per_person = self.Q_total_load / self.N
        
        current_a = np.ones(self.N) * 0.5
        current_P = np.zeros(self.N)
        current_q = np.ones(self.N) * (q_per_person / (q_per_person + 1.0)) # q = Q / (Q+F)
        
        history_avg_P = []
        
        for t in range(ticks):
            # 2. DYNAMICS
            # A) Presence Calculation
            # P = a / (1 + q) * Boost
            # Adoption: If bonded to Spirit (which we assume), get Boost
            boost = 2.0 if TRANSITIVE_ADOPTION else 1.0 
            current_P = (current_a * boost) / (1.0 + current_q + 0.01)
            
            # B) Agency Uplift (The "Discipleship" Term)
            # Humans are pulled toward Christ's P=1.0
            # BUT they are dragged down by their own q
            # delta_a = k * (Christ_P - a) - drag * q
            uplift = 0.1 * (christ_P - current_P)
            drag = 0.2 * current_q
            current_a = np.clip(current_a + uplift - drag, 0, 1)
            
            # C) Processing (Fruitfulness)
            # The Body processes the debt over time.
            # Rate = sum(P) * efficiency
            # More P (from more people or higher a) = faster cleaning
            processing_rate = np.sum(current_P) * 0.01
            
            # Reduce the load
            total_Q = np.sum(current_q) # Proxy for remaining load
            reduction = min(current_q[0], processing_rate / self.N) # Simplified
            current_q = np.maximum(0, current_q - reduction)
            
            history_avg_P.append(np.mean(current_P))
            
        return history_avg_P

# --- RUN SCENARIOS ---
# Scenario 1: The Solitary Saint (N=1)
sim_solitary = BodyOfChristSim(population_size=1)
p_solitary = sim_solitary.run()

# Scenario 2: The Small Church (N=12)
sim_church = BodyOfChristSim(population_size=12)
p_church = sim_church.run()

# Scenario 3: The Great Multitude (N=144)
sim_city = BodyOfChristSim(population_size=144)
p_city = sim_city.run()

# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))
plt.plot(p_solitary, label='Solitary Believer (N=1)', linestyle=':')
plt.plot(p_church, label='The Disciples (N=12)', linestyle='--')
plt.plot(p_city, label='The City of God (N=144)', linewidth=2.5)
plt.axhline(y=1.0, color='gold', label='Divine Presence (Target)', alpha=0.5)

plt.title('The "Be Fruitful and Multiply" Effect: Scaling Presence')
plt.ylabel('Average Human Presence (P)')
plt.xlabel('Time (Ticks)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()