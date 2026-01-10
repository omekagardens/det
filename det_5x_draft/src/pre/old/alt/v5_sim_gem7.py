import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration: Gravity/Metric Test ---
BETA = -0.2         
ALPHA_LEARN = 1.0   
LAMBDA_DECAY = 0.05 
DT = 0.05           
STEPS = 1000        

class DET_v5_Sim:
    def __init__(self, run_name="Metric Contraction Test"):
        self.run_name = run_name
        self.num_nodes = 20 
        
        # State Variables
        self.F = np.zeros(self.num_nodes) 
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         
        self.k = np.zeros(self.num_nodes)      
        self.sigma = np.ones(self.num_nodes)   
        
        # Bond Matrix
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes - 1):
            self.C[i, i+1] = 0.01 
            self.C[i+1, i] = 0.01 

        self.history_C_avg = [] 

    def get_neighbors(self, i):
        row_nbs = np.where(self.C[i] > 0.001)[0]
        col_nbs = np.where(self.C[:, i] > 0.001)[0]
        return np.unique(np.concatenate((row_nbs, col_nbs)))

    def step(self):
        # 1. Phase Update
        d_theta = BETA * self.F
        self.theta += d_theta * DT
        self.theta = np.mod(self.theta, 2 * np.pi)

        # 2. Psi
        Psi = np.zeros(self.num_nodes, dtype=complex)
        for i in range(self.num_nodes):
            nbs = self.get_neighbors(i)
            # Add epsilon to prevent div/0
            norm = 1.0 / (self.F[i] + np.sum(self.F[nbs]) + 1e-9)
            Psi[i] = np.sqrt(self.F[i] * norm) * np.exp(1j * self.theta[i])

        # 3. Flow
        J_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes): 
                if self.C[i,j] < 0.001 and self.C[j,i] < 0.001: continue
                
                c_avg = 0.5 * (self.C[i,j] + self.C[j,i])
                term_Q = np.sqrt(c_avg) * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C = (1.0 - np.sqrt(c_avg)) * (self.F[i] - self.F[j])
                raw_drive = term_Q + term_C
                
                cond = self.C[i,j] if raw_drive > 0 else self.C[j,i]
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                flow = sigma_bond * cond * raw_drive
                J_matrix[i,j] = flow
                J_matrix[j,i] = -flow 

        # 4. Resource Update 
        delta_F = np.sum(J_matrix, axis=1) * DT
        self.F -= delta_F
        self.F = np.maximum(self.F, 0)
        
        # 5. History
        self.k += np.sum(np.abs(J_matrix), axis=1) * DT
        self.sigma = 1.0 + np.log(1.0 + self.k)

        # 6. Hebbian
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i == j: continue
                growth = ALPHA_LEARN * max(0, J_matrix[i,j])
                decay = LAMBDA_DECAY * self.C[i,j]
                self.C[i,j] += (growth - decay) * DT
                self.C[i,j] = np.clip(self.C[i,j], 0.0, 1.0)

        # Track Metric (Middle of the vacuum)
        total_C = 0
        count = 0
        for i in range(5, 15):
            total_C += self.C[i, i+1]
            count += 1
        self.history_C_avg.append(total_C / count)

    def run(self, mass_resource, vacuum_level):
        print(f"Running Metric Test: Mass={mass_resource}, Vacuum={vacuum_level}")
        
        # Initialize Vacuum (ZPE)
        self.F[:] = vacuum_level
        
        for k in range(STEPS):
            # Clamp Masses
            self.F[0] = mass_resource
            self.F[self.num_nodes-1] = mass_resource
            
            # Clamp Vacuum (The "reservoir" of space)
            # We refill the middle to maintain ZPE, simulating an infinite universe
            if k % 5 == 0:
                mask = np.ones(self.num_nodes, dtype=bool)
                mask[0] = False
                mask[-1] = False
                # Slowly restore vacuum to ZPE level to prevent depletion
                self.F[mask] = np.maximum(self.F[mask], vacuum_level)

            self.step()
            
        final_metric = self.history_C_avg[-1]
        print(f"  > Final Vacuum Conductivity: {final_metric:.4f}")
        return self.history_C_avg

def run_gravity_test():
    print("--- Running DET v5 Gravity Derivation ---")
    print("Hypothesis: Mass drags Vacuum Phase -> Steady Inflow -> Metric Contraction.")
    
    # CASE 1: Low Mass (Reference)
    # Mass is 5.0, Vacuum is 2.0 (Small delta)
    sim_low = DET_v5_Sim("Low Mass")
    metric_low = sim_low.run(mass_resource=5.0, vacuum_level=2.0)
    
    # CASE 2: High Mass
    # Mass is 25.0, Vacuum is 2.0 (Large delta -> Strong Phase Pull)
    sim_high = DET_v5_Sim("High Mass")
    metric_high = sim_high.run(mass_resource=25.0, vacuum_level=2.0)
    
    # Analysis
    final_low = metric_low[-1]
    final_high = metric_high[-1]
    
    ratio = final_high / (final_low + 1e-6)
    
    print("\n--- Analysis ---")
    print(f"Vacuum Conductivity (Low Mass):  {final_low:.4f}")
    print(f"Vacuum Conductivity (High Mass): {final_high:.4f}")
    print(f"Contraction Ratio: {ratio:.2f}x")
    
    if ratio > 1.1:
        print(">> SUCCESS: High Mass caused space to contract (Bonds strengthened).")
    else:
        print(">> FAILURE: Mass had no attractive effect.")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(metric_high, color='crimson', label='High Mass (M=25)')
    plt.plot(metric_low, color='blue', linestyle='--', label='Low Mass (M=5)')
    plt.title("Emergence of Gravity (Metric Contraction)")
    plt.ylabel("Vacuum Conductivity (Inverse Distance)")
    plt.xlabel("Time Step")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    run_gravity_test()