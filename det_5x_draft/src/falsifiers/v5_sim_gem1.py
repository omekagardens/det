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
        self.F = np.array([10.0, 0.0, 0.0, 0.0]) # Resource
        self.theta = np.zeros(self.num_nodes)    # Phase
        self.sigma = np.ones(self.num_nodes)     # Processing Rate (Fixed for now)
        self.a = np.ones(self.num_nodes)         # Agency
        
        # Bond Matrix (Adjacency) - Initialize with weak connections
        # 0 connects to 1 and 2. 1 and 2 connect to 3.
        self.C = np.zeros((self.num_nodes, self.num_nodes))
        self.C[0,1] = 0.5
        self.C[0,2] = 0.5
        self.C[1,3] = 0.5
        self.C[2,3] = 0.5
        
        # Make symmetric for undirected graph structure (though flow is directed)
        self.C = np.maximum(self.C, self.C.T)

        # History for plotting
        self.history_F = []
        self.history_C_01 = [] # Bond Source->PathA
        self.history_C_02 = [] # Bond Source->PathB
        self.history_Theta_Diff = [] # Phase difference between PathA and PathB

    def get_neighbors(self, i):
        return np.where(self.C[i] > 0.01)[0] # Threshold for "active" neighbor

    def step(self):
        # 1. Update Phases (v5 Rule: Theta += Beta * F * dt)
        # Higher resource = faster spinning clock
        self.theta += BETA * self.F * DT
        self.theta = np.mod(self.theta, 2 * np.pi)

        # 2. Calculate Local Wavefunction Psi
        # Psi_i = sqrt(F_i / LocalSum) * e^(i*theta)
        Psi = np.zeros(self.num_nodes, dtype=complex)
        for i in range(self.num_nodes):
            nbs = self.get_neighbors(i)
            # Local normalization sum (Scope Axiom)
            norm_sum = self.F[i] + np.sum(self.F[nbs]) + 1e-6
            Psi[i] = np.sqrt(self.F[i] / norm_sum) * np.exp(1j * self.theta[i])

        # 3. Calculate Flow J (Quantum-Classical Interpolation)
        # J_ij = Sigma_ij * [ sqrt(C)*Im(Psi*Psi) + (1-sqrt(C))*(F_i - F_j) ]
        J = np.zeros((self.num_nodes, self.num_nodes))
        
        for i in range(self.num_nodes):
            for j in self.get_neighbors(i):
                if i == j: continue
                
                # Quantum Term: Coherent flow based on phase difference
                # Im(Psi_i* * Psi_j) ~ sin(theta_j - theta_i)
                term_Q = np.sqrt(self.C[i,j]) * np.imag(np.conj(Psi[i]) * Psi[j])
                
                # Classical Term: Diffusion based on resource gradient
                term_C = (1.0 - np.sqrt(self.C[i,j])) * (self.F[i] - self.F[j])
                
                # Bond conductivity (averaged sigma)
                sigma_bond = 0.5 * (self.sigma[i] + self.sigma[j])
                
                J[i,j] = sigma_bond * (term_Q + term_C)

        # 4. Resource Update
        # F_new = F_old - net_flow
        net_flow = np.sum(J, axis=1) # Positive J_ij means flow AWAY from i
        self.F -= net_flow * DT
        
        # Constrain F >= 0
        self.F = np.maximum(self.F, 0)
        
        # Refill Source (driving the experiment)
        self.F[0] = 10.0 
        # Drain Detector (simulating measurement)
        self.F[3] *= 0.9 

        # 5. Dynamic Bond Update (v5 Hebbian Rule)
        # dC = Alpha * |J| * cos(dTheta) - Lambda * C
        # Bonds grow if flow is high AND phases are aligned.
        # Bonds decay if unused or incoherent.
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes): # Iterate unique pairs
                if self.C[i,j] > 0:
                    flow_mag = abs(J[i,j])
                    phase_align = np.cos(self.theta[i] - self.theta[j])
                    
                    delta_C = (ALPHA_LEARN * flow_mag * phase_align) - (LAMBDA_DECAY * self.C[i,j])
                    
                    self.C[i,j] += delta_C * DT
                    self.C[i,j] = np.clip(self.C[i,j], 0.0, 1.0)
                    self.C[j,i] = self.C[i,j] # Keep symmetric

        # Record History
        self.history_F.append(self.F.copy())
        self.history_C_01.append(self.C[0,1])
        self.history_C_02.append(self.C[0,2])
        self.history_Theta_Diff.append(self.theta[1] - self.theta[2])

    def run(self):
        print(f"Running DET v5 Simulation ({STEPS} steps)...")
        print("Setup: Node 0 (Source) -> [Node 1, Node 2] -> Node 3 (Detector)")
        
        for k in range(STEPS):
            self.step()
            
        print("Simulation Complete.")
        
        # --- ASCII Analysis ---
        avg_F3 = np.mean([h[3] for h in self.history_F[-50:]])
        final_C01 = self.history_C_01[-1]
        final_C02 = self.history_C_02[-1]
        
        print("\n--- Results ---")
        print(f"Final Detector Resource (F_3): {avg_F3:.4f}")
        print(f"Final Bond Strength (Source->PathA): {final_C01:.4f}")
        print(f"Final Bond Strength (Source->PathB): {final_C02:.4f}")
        
        if abs(final_C01 - final_C02) > 0.1:
            print(">> Symmetry Broken: One path dominated (Hebbian Selection).")
        else:
            print(">> Symmetry Maintained: Both paths active.")

        return self.history_F, self.history_C_01, self.history_C_02

if __name__ == "__main__":
    sim = DET_v5_Sim()
    sim.run()