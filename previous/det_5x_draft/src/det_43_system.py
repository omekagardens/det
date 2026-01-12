import numpy as np

class DETSystem:
    def __init__(self, num_nodes, adjacency_matrix, params=None):
        """
        Initializes the DET system state.
        
        Args:
            num_nodes (int): Number of agents/nodes.
            adjacency_matrix (np.ndarray): Weighted adjacency matrix (sigma_ij). 
                                           Self-loops should be 0.
            params (dict): System parameters (alpha, epsilon, etc.)
        """
        self.N = num_nodes
        self.adj = adjacency_matrix  # sigma_ij
        
        # --- Default Parameters (DET 4.2) ---
        self.params = {
            'alpha_b': 0.1,    # Baseline smoothing scale (Appendix G)
            'alpha_q': 0.05,   # Structural locking rate
            'gamma_0': 0.0,    # Phase coupling gain
            'omega_0': 0.1,    # Intrinsic frequency
            'kappa': 1.0,      # Gravity coupling constant
            'epsilon': 1e-9,   # Numerical stabilizer
            'F_min': 0.0,      # Minimum resource threshold for grace
            'c_star_sq': 10.0, # Squared speed of light (for clock mapping)
            'iterations_b': 10,# Solver iterations for baseline
            'iterations_phi': 10 # Solver iterations for gravity
        }
        if params:
            self.params.update(params)

        # --- State Variables (Section II) ---
        # Creature variables
        self.F = np.ones(self.N) * 10.0      # Resource
        self.q = np.zeros(self.N)            # Structural debt (0 to 1)
        self.a = np.ones(self.N)             # Agency (0 to 1)
        self.theta = np.random.rand(self.N) * 2 * np.pi # Phase
        self.tau = np.zeros(self.N)          # Proper time
        
        # Bond variables
        self.C = np.ones((self.N, self.N))   # Coherence (0 to 1)
        # Mask C where there is no bond
        self.C[self.adj == 0] = 0
        
        # Derived/Diagnostic Fields
        self.b = np.zeros(self.N)            # Baseline structural field
        self.Phi = np.zeros(self.N)          # Gravity potential

    def step(self, dt_global=1.0):
        """
        Executes one canonical global update step (k -> k+1).
        Follows Section X and Appendix B.3 of README4.3.md.
        """
        p = self.params
        eps = p['epsilon']
        
        # =========================================================
        # Step A: Coordination + Presence (Section III)
        # =========================================================
        # Option B: Coherence-Weighted Load
        # H_i = Sum_j (sqrt(C_ij) * sigma_ij)
        sqrt_C = np.sqrt(self.C)
        H = np.sum(sqrt_C * self.adj, axis=1)
        
        # Presence (P_i)
        # P_i = a_i * sigma_i * (1/(1+F_op)) * (1/(1+H_i))
        # Note: In standard DET, sigma_i is often sum(sigma_ij) or a fixed rate.
        # Here we assume sigma_i = 1 for simplicity unless defined otherwise, 
        # or derived from adjacency sum. Let's use the node's total capacity.
        sigma_i = np.sum(self.adj, axis=1)
        # Assuming all F is operational for this implementations
        P = self.a * sigma_i * (1.0 / (1.0 + self.F)) * (1.0 / (1.0 + H))
        
        # Local time increment
        d_tau = P * dt_global
        self.tau += d_tau

        # =========================================================
        # Step B: Wavefunction + Flow (Section IV)
        # =========================================================
        # Wavefunction psi
        # Normalizer Z_psi = Sum_k F_k (in neighborhood)
        # We can use matrix mult for neighborhood summation
        F_neighbors = self.adj.dot(self.F) + eps # Weighted sum or plain sum? 
        # Text says "sum_{k in N(i)} F_k". Usually assumes weight 1 if connected.
        # Let's use binary adjacency for the scope sum to match "N_R(i)" strictly.
        adj_binary = (self.adj > 0).astype(float)
        Z_psi = adj_binary.dot(self.F) + eps
        
        psi_mag = np.sqrt(self.F / Z_psi)
        psi = psi_mag * np.exp(1j * self.theta)
        
        # Flow J_ij
        # Need pairwise matrices. 
        # J_matrix[i,j] is flow FROM i TO j.
        # Computed using numpy broadcasting.
        
        # Term 1: Quantum (Im(psi_i* psi_j))
        psi_i_grid = psi[:, np.newaxis] # Column
        psi_j_grid = psi[np.newaxis, :] # Row
        quantum_term = np.imag(np.conj(psi_i_grid) * psi_j_grid)
        
        # Term 2: Classical (F_i - F_j)
        F_i_grid = self.F[:, np.newaxis]
        F_j_grid = self.F[np.newaxis, :]
        classical_term = F_i_grid - F_j_grid
        
        # J_ij = sigma_ij * [ sqrt(C)*Q + (1-sqrt(C))*Class ]
        flow_matrix = self.adj * (
            sqrt_C * quantum_term + 
            (1.0 - sqrt_C) * classical_term
        )
        
        # =========================================================
        # Step C: Dissipation (Section V.2)
        # =========================================================
        # D_i = Sum_j |J_ij| * d_tau_i
        # Note: The text says D_i depends on J_{i->j}. 
        abs_flow = np.abs(flow_matrix)
        D = np.sum(abs_flow, axis=1) * d_tau

        # =========================================================
        # Step D: Boundary Operators (Section V)
        # =========================================================
        # Grace Injection
        # n_i = max(0, F_min - F_i)
        n = np.maximum(0, p['F_min'] - self.F)
        w = self.a * n
        
        # Z_i neighborhood normalizer for grace
        # Sum_{k in N(i)} w_k
        Z_grace = adj_binary.dot(w) + eps
        
        # I_g -> i
        I_grace = D * (w / Z_grace)
        # Enforce: if a=0, I_g=0 (Implicit in w=a*n, but good to ensure)
        I_grace[self.a == 0] = 0.0

        # Bond Healing (Reconciliation)
        # u_ij = a_i * a_j * (1 - C_ij)
        a_i_grid = self.a[:, np.newaxis]
        a_j_grid = self.a[np.newaxis, :]
        u_matrix = a_i_grid * a_j_grid * (1.0 - self.C) * adj_binary
        
        # Z^C_ij (Sum of u in bond neighborhood)
        # Edge neighborhood usually means edges connected to i or j.
        # For simplicity in matrix form, we often approximate or iterate.
        # The strict definition: E_R(i,j). Let's assume simpler bond-local scope 
        # for this code snippet or just use the u_matrix directly scaled.
        # To strictly implement Z^C without hypergraph structures is complex;
        # Standard DET numerical ablation often uses Z^C = Sum(u) over local star.
        # Let's use Z_healing_norm = Sum_k u_ik + Sum_k u_jk
        row_sum_u = np.sum(u_matrix, axis=1)
        Z_healing = row_sum_u[:, np.newaxis] + row_sum_u[np.newaxis, :] + eps
        
        D_bond = 0.5 * (D[:, np.newaxis] + D[np.newaxis, :])
        delta_C_grace = D_bond * (u_matrix / Z_healing)
        
        # Update Coherence (clipped)
        self.C = np.clip(self.C + delta_C_grace, 0, 1)
        self.C[self.adj == 0] = 0 # Enforce topology

        # =========================================================
        # Step E: Resource Update (Section IV.3)
        # =========================================================
        # F_i+ = F_i - Sum_j J_ij * d_tau_i + I_g
        # Note: J_ij is flow i->j. Outgoing is positive in formula J_i->j.
        net_flow_out = np.sum(flow_matrix, axis=1)
        
        # Calculate F change for structural locking usage
        delta_F_transport = -net_flow_out * d_tau + I_grace
        
        self.F += delta_F_transport
        # Ensure F >= 0 physically
        self.F = np.maximum(self.F, 0)

        # =========================================================
        # Step F: Structural Locking (Appendix B.3)
        # =========================================================
        # q+ = clip(q + alpha_q * max(0, -Delta F))
        # "Net inward flow increases frozen structure"
        # Delta F here refers to the change in the step.
        structure_growth = p['alpha_q'] * np.maximum(0, -delta_F_transport)
        self.q = np.clip(self.q + structure_growth, 0, 1)

        # =========================================================
        # Step G: Agency Update (Section VI)
        # =========================================================
        # a+ = clip(a + (P - P_avg_neighbor) - q)
        P_avg = adj_binary.dot(P) / (np.sum(adj_binary, axis=1) + eps)
        
        # To handle isolated nodes where sum is 0
        P_avg[np.sum(adj_binary, axis=1) == 0] = P[np.sum(adj_binary, axis=1) == 0]

        delta_a = (P - P_avg) - self.q
        self.a = np.clip(self.a + delta_a, 0, 1)

        # =========================================================
        # Step H: Phase Update (Section V.0)
        # =========================================================
        # theta+ = theta + omega * d_tau
        self.theta += p['omega_0'] * d_tau
        
        # Optional coupling
        if p['gamma_0'] > 0:
            sin_diff = np.sin(self.theta[np.newaxis, :] - self.theta[:, np.newaxis])
            coupling = np.sum(
                p['gamma_0'] * self.adj * sqrt_C * a_i_grid * a_j_grid * sin_diff, 
                axis=1
            )
            self.theta += coupling * d_tau
            
        self.theta = np.mod(self.theta, 2 * np.pi)

        # =========================================================
        # Step I: Baseline & Gravity (Appendix G)
        # =========================================================
        # 1. Solve for Baseline b: (L_sigma - alpha)b = -alpha q
        # Iterative update: b_i = (Sum_j sigma_ij b_j + alpha q_i) / (Sum_j sigma_ij + alpha)
        # This is strictly local per iteration.
        
        sigma_sum = np.sum(self.adj, axis=1)
        denom = sigma_sum + p['alpha_b']
        
        # Run a few relaxation steps
        for _ in range(p['iterations_b']):
            neighbor_sum_b = self.adj.dot(self.b)
            self.b = (neighbor_sum_b + p['alpha_b'] * self.q) / denom
            
        # 2. Gravity Source
        rho = self.q - self.b
        
        # 3. Solve for Potential Phi: L_sigma Phi = -kappa rho
        # Laplacian L_sigma Phi ~ Sum sigma_ij (Phi_i - Phi_j)
        # Sum sigma_ij Phi_i - Sum sigma_ij Phi_j = -kappa rho
        # Phi_i (Sum sigma_ij) = Sum sigma_ij Phi_j - kappa rho
        # Phi_i = (Sum sigma_ij Phi_j - kappa rho) / (Sum sigma_ij)
        # Note: Standard Poisson requires boundary conditions or gauge fixing.
        # In this relaxation, the mean level might drift unless anchored, 
        # but relative differences (forces) remain valid.
        
        denom_phi = sigma_sum + eps
        for _ in range(p['iterations_phi']):
            neighbor_sum_phi = self.adj.dot(self.Phi)
            self.Phi = (neighbor_sum_phi - p['kappa'] * rho) / denom_phi

    def get_state(self):
        return {
            "F": self.F.copy(),
            "q": self.q.copy(),
            "a": self.a.copy(),
            "P": self.a * np.sum(self.adj, axis=1) * (1/(1+self.F)) * (1/(1+np.sum(np.sqrt(self.C)*self.adj, axis=1))), # Recalculate P for current state
            "b": self.b.copy(),
            "Phi": self.Phi.copy()
        }