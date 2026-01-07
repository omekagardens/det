import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration ---
BETA = 1.0         # Phase-Resource coupling (1/h)
ALPHA_LEARN = 0.05 # Hebbian bond learning
LAMBDA_DECAY = 0.01 # Bond decay
DT = 0.1           # Time step
STEPS = 600        

class DET_v5_Sim:
    def __init__(self, run_name="Pre-trained Capacity Test"):
        self.run_name = run_name
        self.num_nodes = 4
        
        # State Variables
        self.F = np.zeros(self.num_nodes) 
        self.theta = np.zeros(self.num_nodes)    
        self.a = np.ones(self.num_nodes)         

        # --- Dynamic Sigma Variables ---
        self.k = np.zeros(self.num_nodes)      
        self.sigma = np.ones(self.num_nodes)   
        
        # Bond Matrix
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        self.C[0,1] = 0.5; self.C[0,2] = 0.5
        self.C[1,3] = 0.5; self.C[2,3] = 0.5
        self.C = np.maximum(self.C, self.C.T)

        # History
        self.history_F = []
        self.history_Sigma = [] 
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
        
        # --- Pulse Logic ---
        # Instead of constant refill, we inject a pulse at the start
        # but only AFTER the pre-training is done (handled in run())
        self.F[3] *= 0.9 # Detector Sink

        # 5. Update History (k) and Sigma
        flux_processed = np.sum(np.abs(J), axis=1)
        self.k += flux_processed * DT
        self.sigma = 1.0 + np.log(1.0 + self.k)

        # 6. Dynamic Bond Update
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
        
        # --- PHASE 1: PRE-TRAINING ---
        print("Phase 1: Pre-training Path A (Artificially boosting History)...")
        # Artificially give Path A (Node 1) a massive history
        self.k[1] = 5000.0
        self.sigma[1] = 1.0 + np.log(1.0 + self.k[1]) # Should be ~9.5
        print(f"  > Path A Sigma starts at: {self.sigma[1]:.4f}")
        print(f"  > Path B Sigma starts at: {self.sigma[2]:.4f}")

        # --- PHASE 2: THE PULSE ---
        print("Phase 2: Injecting Pulse at Source (Node 0)...")
        # Inject resource pulse
        pulse_duration = 50
        
        for t in range(STEPS):
            if t < pulse_duration:
                self.F[0] = 10.0 # Source On
            else:
                pass # Source Off (allow system to drain)
                
            self.step()
            
        self.analyze_sigma()
        self.plot_results()

    def analyze_sigma(self):
        # Calculate total flow processed by each path during the pulse
        total_J_A = np.sum(self.history_J_13)
        total_J_B = np.sum(self.history_J_23)
        
        print(f"\nResults:")
        print(f"  Total Flow through Pre-trained Path A: {total_J_A:.4f}")
        print(f"  Total Flow through Empty Path B:       {total_J_B:.4f}")
        
        ratio = total_J_A / (total_J_B + 1e-6)
        print(f"  > Preference Ratio (A/B): {ratio:.4f}")
        
        if ratio > 1.5:
            print("  [SUCCESS] System prefers the 'Experienced' path. History = Measurable Capacity.")
        else:
            print("  [FAIL] System does not discriminate based on history.")

    def plot_results(self):
        steps = np.arange(STEPS)
        
        plt.figure(figsize=(10, 8))
        plt.suptitle(f"DET v5: History vs Vacuum ({self.run_name})", fontsize=16)

        plt.subplot(2, 1, 1)
        plt.plot(steps, self.history_J_13, color='blue', label='Flow Path A (Experienced)')
        plt.plot(steps, self.history_J_23, color='red', linestyle='--', label='Flow Path B (Novice)')
        plt.title("Flow Comparison: Does Experience Matter?")
        plt.ylabel("Flow Rate J")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(steps, [h[1] for h in self.history_F], color='blue', label='Resource Path A')
        plt.plot(steps, [h[2] for h in self.history_F], color='red', linestyle='--', label='Resource Path B')
        plt.title("Resource Accumulation")
        plt.ylabel("Resource F")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    sim = DET_v5_Sim(run_name="Pre-trained Capacity Test")
    sim.run()