import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration ---
BETA = 1.0         # Phase-Resource coupling (1/h)
ALPHA_LEARN = 0.05 # Hebbian bond learning
LAMBDA_DECAY = 0.01 # Bond decay
DT = 0.1           # Time step
STEPS = 600        

class DET_v5_Sim:
    def __init__(self, run_name="Dynamic Sigma Test"):
        self.run_name = run_name
        self.num_nodes = 4
        
        # State Variables
        # Node 1 (Path A) starts with 5.0, Node 2 (Path B) with 0.0
        self.F = np.array([10.0, 5.0, 0.0, 0.0]) 
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         

        # --- NEW: Dynamic Sigma Variables ---
        self.k = np.zeros(self.num_nodes)      # Event/Flux Accumulator (History)
        self.sigma = np.ones(self.num_nodes)   # Processing Rate (starts at 1.0)
        
        # Bond Matrix
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        self.C[0,1] = 0.5; self.C[0,2] = 0.5
        self.C[1,3] = 0.5; self.C[2,3] = 0.5
        self.C = np.maximum(self.C, self.C.T)

        # History
        self.history_F = []
        self.history_Sigma = [] # Track Sigma evolution
        self.history_J_13 = [] 
        self.history_J_23 = [] 

    def get_neighbors(self, i):
        return np.where(self.C[i] > 0.01)[0] 

    def step(self):
        # 1. Update Phases
        self.theta += BETA * self.F * DT
        self.theta = np.mod(self.theta, 2 * np.pi)

        # 2. Local Wavefunction
        Psi = np.zeros(self.num_nodes, dtype=complex)
        for i in range(self.num_nodes):
            nbs = self.get_neighbors(i)
            norm_sum = self.F[i] + np.sum(self.F[nbs]) + 1e-6
            Psi[i] = np.sqrt(self.F[i] / norm_sum) * np.exp(1j * self.theta[i])

        # 3. Calculate Flow J
        J = np.zeros((self.num_nodes, self.num_nodes))
        for i in range(self.num_nodes):
            for j in self.get_neighbors(i):
                if i == j: continue
                
                term_Q = np.sqrt(self.C[i,j]) * np.imag(np.conj(Psi[i]) * Psi[j])
                term_C = (1.0 - np.sqrt(self.C[i,j])) * (self.F[i] - self.F[j])
                
                # SIGMA EFFECT: Flow is scaled by the average Sigma of the bond
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                J[i,j] = sigma_bond * (term_Q + term_C)

        # 4. Resource Update
        net_flow = np.sum(J, axis=1) 
        self.F -= net_flow * DT
        self.F = np.maximum(self.F, 0)
        
        self.F[0] = 10.0 # Source
        self.F[3] *= 0.9 # Detector Sink

        # --- NEW: Update History (k) and Sigma ---
        # k accumulates the total absolute flux processed by the node
        # This is "Participation"
        flux_processed = np.sum(np.abs(J), axis=1)
        self.k += flux_processed * DT
        
        # Sigma grows with History (Logarithmic saturation)
        # Sigma = 1 + ln(1 + k)
        self.sigma = 1.0 + np.log(1.0 + self.k)

        # 5. Dynamic Bond Update
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes): 
                if self.C[i,j] > 0:
                    flow_mag = abs(J[i,j])
                    phase_align = np.cos(self.theta[i] - self.theta[j])
                    delta_C = (ALPHA_LEARN * flow_mag * phase_align) - (LAMBDA_DECAY * self.C[i,j])
                    self.C[i,j] += delta_C * DT
                    self.C[i,j] = np.clip(self.C[i,j], 0.0, 1.0)
                    self.C[j,i] = self.C[i,j] 

        # Record History
        self.history_F.append(self.F.copy())
        self.history_Sigma.append(self.sigma.copy())
        self.history_J_13.append(J[1,3])
        self.history_J_23.append(J[2,3])

    def run(self):
        print(f"\n--- Running DET v5: {self.run_name} ---")
        for k in range(STEPS):
            self.step()
        self.analyze_sigma()
        self.plot_results()

    def analyze_sigma(self):
        final_sigma = self.sigma
        print(f"Final Processing Rates (Sigma):")
        print(f"  Source (Node 0): {final_sigma[0]:.4f}")
        print(f"  Path A (Node 1): {final_sigma[1]:.4f} (Started with Resource)")
        print(f"  Path B (Node 2): {final_sigma[2]:.4f} (Started Empty)")
        print(f"  Detector(Node 3):{final_sigma[3]:.4f}")
        
        diff = final_sigma[1] - final_sigma[2]
        print(f"  > Sigma Asymmetry: {diff:.4f}")
        
        if diff > 0.5:
            print("  [RESULT] Path A became a 'Superconductor' due to history.")
        else:
            print("  [RESULT] History impact was minimal.")

    def plot_results(self):
        steps = np.arange(STEPS)
        sigma_trace = np.array(self.history_Sigma)
        
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"DET v5 Dynamic Sigma: {self.run_name}", fontsize=16)

        # Plot 1: Sigma Evolution
        plt.subplot(2, 1, 1)
        plt.plot(steps, sigma_trace[:, 1], color='blue', label='Sigma Path A')
        plt.plot(steps, sigma_trace[:, 2], color='red', linestyle='--', label='Sigma Path B')
        plt.title("Evolution of Processing Rate (Measurability)")
        plt.ylabel("Sigma (Capacity)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Flow Comparison
        plt.subplot(2, 1, 2)
        plt.plot(steps, self.history_J_13, color='blue', alpha=0.6, label='Flow Path A')
        plt.plot(steps, self.history_J_23, color='red', alpha=0.6, label='Flow Path B')
        plt.title("Flow Volume (Impact of Sigma)")
        plt.ylabel("Flow J")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = DET_v5_Sim(run_name="Dynamic Sigma (History=Participation)")
    sim.run()