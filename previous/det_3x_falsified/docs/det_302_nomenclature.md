# DET Unified Nomenclature
Symbol	Name	Status	Meaning
e	Event	Primitive	Causal occurrence
≺	Causal order	Primitive	Partial order on events
i,j	Nodes	Primitive	Network agents
\tau_i	Proper time	Primitive	Local experienced time
k	Event index	Gauge	Simulation ordering only
k_i	Local counter	Gauge	Node bookkeeping
P_i	Presence (operational)	Derived	d\tau_i/dk (local operational clock)
M_i	Coordination debt	Derived	1 + M_i^{struct} + M_i^{op}
M_i^{struct}	Structural coordination debt	Derived	Persistent structure; gravity/inertia source
M_i^{op}	Operational coordination debt	Derived	Circulating load; clock-constrained
F_i	Resource (total)	Primitive	Stored + circulating quantity
F_i^{op}	Operational resource	Derived	Actively circulating / processing load
\sigma_i	Processing rate	Primitive	Node clock factor
\sigma_{ij}	Edge conductivity	Primitive	Transport weight
\Psi_{ij}	Bond state	Primitive	(C_{ij},\phi_{ij},U_{ij})
C_{ij}	Coherence	Primitive	Quantum channel strength
\phi_{ij}	Relational phase	Primitive	U(1) connection
U_{ij}	Gauge connection	Extension	SU(2) lift
\psi_i	Wavefunction	Derived	\sqrt{R_i}e^{i\theta_i}
\Phi_i	Throughput potential	Derived	c_*^2\ln(M_i/M_0)
V_{\text{res}}	Reservoir potential	Primitive	Boundary condition
 c_*	Light speed	Emergent	Frozen fixed-point propagation speed (\dot c_* \approx 0 today)
\kappa	Gravity coupling	Emergent	Network equilibration
\lambda_{ij}	Decoherence rate	Primitive	Environment + kinematic coherence loss (\lambda_0=0 core)
\chi_i	Bureaucratic drag	Extension	Admin complexity
\Omega_i	Dead capital	Extension	Ghost mass
\Xi_i	Structural density	Extension	Rest-like persistent structural content

\beta_{op}	Clock-load coupling	Derived	Operational load → clock perturbation (precision-constrained)