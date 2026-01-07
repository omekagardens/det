DET v5: Postulate Update (Inertia & Motion)
Status: Validated via Simulation (det_v5_simulation.py) Date: Post-Soliton Success
1. The Fundamental Shift: Directed Space
In DET v4.2, bonds were symmetric ($C_{ij} = C_{ji}$). New Postulate: Space is fundamentally Directed.
$$C_{i \to j} \neq C_{j \to i}$$
* Physical Interpretation: $C_{i \to j}$ represents the "permeability" or "expectation" of flow in a specific direction.
* Emergent Inertia: Momentum is not a variable on the node; it is the asymmetry of the local bonds. If $C_{\text{right}} \gg C_{\text{left}}$, the particle must move right.
2. The Phase-Energy Relation
New Postulate: Resource drives Phase Velocity negatively.
$$\dot{\theta}_i = -\beta \cdot F_i$$
* Mechanism:
    * High $F$ (Particle) spins backwards relative to Low $F$ (Vacuum).
    * This creates a phase gradient $\nabla \theta$ that "pulls" the particle forward into the vacuum.
    * This matches the quantum mechanical time evolution operator $e^{-iEt/\hbar}$.
3. The Flow-Structure Loop (The Soliton Engine)
Motion is a self-sustaining cycle:
1. Drive: Phase difference ($\Delta \theta$) + Gradient ($\Delta F$) pushes Resource forward.
2. Rectification: The flow $J_{i \to j}$ activates the Hebbian growth of $C_{i \to j}$.
3. Pruning: The lack of backflow allows $C_{j \to i}$ to decay.
4. Result: The particle builds its own road ahead and burns the bridge behind.
4. Derived Definition of "Mass"
In this new regime, "Mass" is resistance to changing direction.
* A moving particle has high $C_{\text{forward}}$ and low $C_{\text{backward}}$.
* To stop or reverse it, you must fight the existing $C$ structure (inertia).
* Mass is the structural anisotropy of the neighborhood.
5. Next Research Target: The Speed of Light ($c$)
Since we now have motion, we have a speed.
* Hypothesis: The speed of the soliton is limited by the "Learning Rate" $\alpha$ and the "Phase Velocity" $\beta$.
* We can now attempt to derive $c$ as a function of these graph parameters, rather than setting it arbitrarily.



Deep Existence Theory (DET) v5.0 — Technical Specification
Version: 5.0 (Draft) Status: Validated via Simulation (Soliton, Speed Limit, Metric Contraction) Core Shift: From "Passive Graph" to "Active, Directed, Learning Manifold"
1. The Fundamental Ontology
The universe consists of a graph of Agents (Nodes) and Relations (Bonds).
* Space is Directed: $C_{ij} \neq C_{ji}$. The universe is a "Diode Network".
* Space is Plastic: The metric ($C$) evolves based on use (Hebbian Learning).
* Time is Local: Every agent has a private clock ($\theta$) driven by its Resource ($F$).
2. The Five Core Equations
I. The Phase-Energy Relation (Time)
Time is the vibration of existence (Resource). High energy rotates "backwards" relative to the vacuum, creating suction.
$$\dot{\theta}_i = -\beta \cdot F_i$$
* $\beta$: Fundamental Phase Coupling constant (Inverse Planck, $1/\hbar$).
* Note: The negative sign is crucial for attractive gravity and forward inertia.
II. The Quantum-Classical Flow (Transport)
Matter moves due to Phase Gradients (Quantum) and Pressure Gradients (Classical).
$$J_{i \to j} = \sigma_{ij} \cdot C_{i \to j} \cdot \left[ \sqrt{C_{ji}} \sin(\theta_j - \theta_i) + (1-\sqrt{C_{ji}})(F_i - F_j) \right]$$
* $J$: Flux from $i$ to $j$.
* Rectification: Flux only flows if the "Drive" is positive; otherwise it is blocked by the diode nature of space.
III. The Hebbian Metric Update (Gravity & Inertia)
Space contracts (bonds strengthen) when flow passes through it.
$$\dot{C}_{i \to j} = \alpha \cdot \max(0, J_{i \to j}) - \lambda \cdot C_{i \to j}$$
* $\alpha$: Learning Rate (Plasticity of Spacetime).
* $\lambda$: Entropy (Decay of Unused Space).
* Gravity: Constant flow into mass strengthens local bonds $\rightarrow$ Metric Contraction.
* Inertia: A moving particle strengthens the path ahead $\rightarrow$ Self-Sustaining Soliton.
IV. Dynamic Capacity (Measurability)
An agent's processing rate is the log of its total life experience.
$$\sigma_i(t) = 1 + \ln\left(1 + \int_0^t |J_{in} + J_{out}| d\tau \right)$$
* Interpretation: "To him who has, more will be given." Old agents are superconductors.
V. The Vacuum Condition (ZPE)
Space is never empty. Gravity requires a medium to act upon.
$$F_{vac} > 0$$
* Without $F_{vac}$, there is no phase to drag, and thus no gravity.
3. Derived Constants
We no longer set $c$ and $G$. They emerge from the graph parameters.
Constant	Symbol	Derivation	Physical Meaning
Speed of Light	$c$	$\sim f(\alpha, \beta)$	The saturation velocity where Phase Rotation aliasing balances Hebbian Road-Building. Simulation value: $\approx 0.33$ nodes/tick.
Gravitational Constant	$G$	$\sim \alpha \cdot F_{vac}$	The efficiency with which a Phase Drag (Mass) converts ZPE flow into Bond Strength.
Planck's Constant	$h$	$1 / \beta$	The energy cost to rotate phase by $1$ radian.
4. Next Frontier: The Particle Zoo (Standard Model)
Now that we have Solitons (Moving Matter), we need Topological Defects.
* Proton: A soliton trapped in a stable loop (Toroidal knot).
* Electron: A phase vortex without a structural core?
* Strong Force: The extreme Hebbian binding of very close nodes ($C \approx 1.0$).
Hypothesis: Stable particles are not point-masses, but Closed-Loop Geodesics formed by solitons biting their own tails.