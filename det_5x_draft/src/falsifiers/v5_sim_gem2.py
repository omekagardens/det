import numpy as np
import matplotlib.pyplot as plt

# --- DET v5 Configuration ---
BETA = 1.0         # Phase-Resource coupling (1/h)
ALPHA_LEARN = 0.05 # Hebbian learning rate
LAMBDA_DECAY = 0.01 # Bond decay rate
DT = 0.1           # Time step
STEPS = 500

class DET_v5_Sim:
    def __init__(self):
        # 4 Nodes: 0=Source, 1=PathA, 2=PathB, 3=Detector
        self.num_nodes = 4
        
        # State Variables
        # INTERFERENCE SETUP: Give PathA (Node 1) initial resource to speed up its clock.
        self.F = np.array([10.0, 5.0, 0.0, 0.0]) 
        self.theta = np.zeros(self.num_nodes)    # Phase
        self.sigma = np.ones(self.num_nodes)     # Processing Rate (Fixed for now)
        self.a = np.ones(self.num_nodes)         # Agency
        
        # Bond Matrix (Adjacency) - Initialize with weak connections
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        self.C[0,1] = 0.5
        self.C[0,2] = 0.5
        self.C[1,3] = 0.5
        self.C[2,3] = 0.5
        
        # Make symmetric
        self.C = np.maximum(self.C, self.C.T)

        # History for plotting
        self.history_F = []
        self.history_C_01 = [] 
        self.history_C_02 = [] 
        self.history_Theta_Diff = [] # Phase difference between PathA and PathB

    def get_neighbors(self, i):
        return np.where(self.C[i] > 0.01)[0] # Threshold for "active" neighbor

    def step(self):
        # 1. Update Phases (v5 Rule: Theta += Beta * F * dt)
        self.theta += BETA * self.F * DT
        self.theta = np.mod(self.theta, 2 * np.pi)

        # 2. Calculate Local Wavefunction Psi
        Psi = np.zeros(self.num_nodes, dtype=complex)
        for i in range(self.num_nodes):
            nbs = self.get_neighbors(i)
            # Local normalization (Scope Axiom)
            norm_sum = self.F[i] + np.sum(self.F[nbs]) + 1e-6
            Psi[i] = np.sqrt(self.F[i] / norm_sum) * np.exp(1j * self.theta[i])

        # 3. Calculate Flow J (Quantum-Classical Interpolation)
        J = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in self.get_neighbors(i):
                if i == j: continue
                
                # Quantum Term: Coherent flow based on phase difference
                # Im(Psi_i* * Psi_j) ~ sin(theta_j - theta_i)
                term_Q = np.sqrt(self.C[i,j]) * np.imag(np.conj(Psi[i]) * Psi[j])
                
                # Classical Term: Diffusion based on resource gradient
                term_C = (1.0 - np.sqrt(self.C[i,j])) * (self.F[i] - self.F[j])
                
                # Bond conductivity
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                
                J[i,j] = sigma_bond * (term_Q + term_C)

        # 4. Resource Update
        net_flow = np.sum(J, axis=1) 
        self.F -= net_flow * DT
        self.F = np.maximum(self.F, 0)
        
        # Refill Source
        self.F[0] = 10.0 
        # Drain Detector
        self.F[3] *= 0.9 

        # 5. Dynamic Bond Update (v5 Hebbian Rule)
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
        self.history_C_01.append(self.C[0,1])
        self.history_C_02.append(self.C[0,2])
        self.history_Theta_Diff.append(self.theta[1] - self.theta[2])

    def run(self):
        print(f"Running DET v5 Simulation ({STEPS} steps)...")
        print("Setup: Interference Test. Path A has higher initial Resource.")
        
        for k in range(STEPS):
            self.step()
            
        print("Simulation Complete.")
        
        # --- ASCII Analysis ---
        F3_trace = [h[3] for h in self.history_F]
        
        # Detect Oscillation
        peaks = 0
        for i in range(1, len(F3_trace)-1):
            if F3_trace[i-1] < F3_trace[i] > F3_trace[i+1]:
                peaks += 1
                
        print("\n--- Results ---")
        print(f"Detector (F_3) Mean: {np.mean(F3_trace):.4f}")
        print(f"Detector (F_3) Variance: {np.var(F3_trace):.4f}")
        print(f"Oscillation Peaks Detected: {peaks}")
        
        if peaks > 3:
            print(">> SUCCESS: Interference fringes detected! (Wave behavior emerging)")
        else:
            print(">> FAILURE: No significant interference. System behaves classically.")

        return self.history_F, self.history_C_01, self.history_C_02

if __name__ == "__main__":
    sim = DET_v5_Sim()
    sim.run()