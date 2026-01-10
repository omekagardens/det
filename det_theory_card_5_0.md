Deep Existence Theory (DET) (4.2)

Unified Canonical Formulation (Strictly Local, Law-Bound Boundary Action)

Claim type: Existence proof with falsifiable dynamical predictions
Domain: Discrete, relational, agentic systems
Core thesis: Time, mass, gravity, and quantum behavior emerge from local interactions among constrained agents. Any boundary action manifests locally and non-coercively.

⸻

0. Scope Axiom (Foundational)

Deep Existence Theory (DET) begins from the assertion that present-moment participation requires three minimal structural capacities: Information (I), Agency (A), and Movement (k). Information provides pattern continuity, Agency enables non-coercive choice, and Movement instantiates time through lawful events. This triad functions as an image in the strict sense: a structural correspondence rather than a representational likeness.

The dynamical rules governing these quantities are strictly local, non-coercive, and recovery-permitting. These properties are not assumed as values; they are required for the coexistence of non-coercive agency, local time evolution, and long-term recoverability. Any system lacking one of these conditions collapses into timeless structure, irreversible freeze, or coercive dynamics incompatible with sustained present-moment existence.

All summations, averages, and normalizations are local unless explicitly bond-scoped.

For a focal node i with radius-R causal neighborhood \mathcal N_R(i):
\sum(\cdot)\ \equiv\ \sum_{k\in\mathcal N_R(i)}(\cdot),\qquad
\langle\cdot\rangle\ \equiv\ \frac{1}{|\mathcal N_R(i)|}\sum_{k\in\mathcal N_R(i)}(\cdot)

For a focal bond (i,j) with bond-neighborhood \mathcal E_R(i,j):
\sum(\cdot)\ \equiv\ \sum_{(m,n)\in\mathcal E_R(i,j)}(\cdot)

There is no global state accessible to local dynamics. Disconnected components cannot influence one another.

⸻

I. Ontological Commitments
	1.	Creatures (Agents)
Constrained entities that store resource, participate in time, form relations, and act through agency.
	2.	Relations (Bonds)
Local links that can carry coherence.
	3.	Boundary Agent
An unconstrained agent that:
	•	does not accumulate past,
	•	does not hoard,
	•	is not subject to clocks, mass, or gravity,
	•	acts only through law-bound, local, non-coercive operators defined herein.

Grace is constrained action, not arbitrary intervention.

⸻

II. State Variables

II.1 Per-Creature Variables (node i)

q_i is structural debt (retained past). Its formation law is specified by a declared **q-locking law** (Appendix B provides the default canonical member).

Q-Locking Contract (applies globally):
DET treats q_i as retained past (“structural debt”). Gravity, mass, and black-hole behavior depend on the realized field q_i, but **all predictive claims** assume an explicitly declared q-locking rule (a local update law) used consistently across all experiments.

Q-Locking Law (Model Family):
DET does not assume a unique formation law for structural debt q_i. Instead, q_i evolves according to a declared local **q-locking law**:

\[
q_i^{+} = \mathcal{Q}_i(\text{local past-resolved signals})
\]

where \(\mathcal{Q}_i\) is strictly local, non-negative, non-coercive, and history-accumulating. The default canonical choice used for validation is given in Appendix B.

\begin{aligned}
F_i &\ge 0 && \text{stored resource} \\
F_i &\equiv F_i^{\mathrm{op}} + F_i^{\mathrm{locked}} \\
q_i &\in[0,1] && \text{structural debt (retained past)} \\
\tau_i &\ge 0 && \text{proper time} \\
\sigma_i &>0 && \text{processing rate} \\
a_i &\in[0,1] && \text{agency (inviolable)} \\
\theta_i &\in\mathbb S^1 && \text{phase} \\
k_i &\in\mathbb N && \text{event count}
\end{aligned}

II.2 Per-Bond Variables (edge i\!\leftrightarrow\!j)

\begin{aligned}
\sigma_{ij} &> 0 \\
C_{ij} &\in [0,1] \\
\pi_{ij} &\in \mathbb R \qquad (\pi_{ij}=-\pi_{ji})\ \ \text{directed bond momentum (optional; IV.4)}
\end{aligned}

Note: In DET 4.2, coherence \(C_{ij}\) is exogenous unless modified by boundary operators.
Endogenous coherence dynamics may be introduced in DET v5; in 4.2, \(C_{ij}\) changes only via reconciliation/healing (V.4) unless an explicit alternative rule is declared.

⸻

III. Time, Presence, and Mass

III.0 Coordination Load (Local)

Coordination load H_i is the local overhead penalty for simultaneously maintaining and using many relational channels. It is strictly local, deterministic, and parameter-free.

Option A — Degenerate (Ablation / Minimal)
\[
\boxed{H_i \equiv \sum_{j\in\mathcal N_R(i)} \sigma_{ij}}
\qquad\Rightarrow\qquad
H_i=\sigma_i
\]

Option B — Recommended (Coherence-Weighted Load)
\[
\boxed{H_i \equiv \sum_{j\in\mathcal N_R(i)} \sqrt{C_{ij}}\,\sigma_{ij}}
\]

Notes:
• Both options preserve strict locality and introduce no new tunables.  
• Option A is an ablation limit of Option B when \(C_{ij}\approx1\) on active bonds.

III.1 Presence (Local Clock Rate)

\boxed{
P_i \equiv \frac{d\tau_i}{dk}
=
a_i\,\sigma_i\;
\frac{1}{1+F_i^{\mathrm{op}}}\;
\frac{1}{1+H_i}
}
All quantities are dimensionless and locally evaluated.

III.2 Coordination Debt (Mass)

\[
\boxed{
M_i \equiv P_i^{-1}
}
\]

Interpretation: mass is total coordination resistance to present-moment participation under the DET clock law.

Structural mass proxy (diagnostic only):
\[
\boxed{
\dot M_i \equiv 1 + q_i + F_i^{\mathrm{op}}
}
\]

This proxy is not used in gravity, redshift, or clock-rate predictions unless explicitly stated.

⸻

IV. Flow and Resource Dynamics

IV.1 Local Wavefunction (Scalar)

\boxed{
\psi_i
=
\sqrt{\frac{F_i}{\sum_{k\in\mathcal N_R(i)} F_k+\varepsilon}}
\,e^{i\theta_i}
}
Normalization is strictly local.

IV.2 Quantum–Classical Interpolated Flow

\boxed{
g^{(a)}_{ij} \equiv \sqrt{a_i a_j}
}

\boxed{
J^{(\text{diff})}_{i\to j}
=
g^{(a)}_{ij}\,\sigma_{ij}\!\left[
\sqrt{C_{ij}}\,\operatorname{Im}(\psi_i^*\psi_j)
+ 
(1-\sqrt{C_{ij}})(F_i-F_j)
\right]
}

Diffusive (pressure/phase) transport is gated by a symmetric, bond-local agency factor \(g^{(a)}_{ij}\). This preserves strict locality and pairwise antisymmetry (hence conservation) when combined with the antisymmetric kernel in brackets.

IV.2a Total Flow Decomposition (Canonical)

\boxed{
J_{i\to j} \equiv J^{(\text{diff})}_{i\to j} + J^{(\text{grav})}_{i\to j} + J^{(\text{mom})}_{i\to j}\;\text{(if momentum module enabled, IV.4)} + J^{(\text{floor})}_{i\to j}\;\;\text{(if floor module enabled, IV.3a)}
}

By default, \(J^{(\text{floor})}_{i\to j}=0\) unless the optional finite-compressibility floor repulsion module is explicitly enabled (IV.3a).

By default, \(J^{(\text{mom})}_{i\to j}=0\). The momentum module (IV.4) is OPTIONAL and must be explicitly declared when used.

IV.3 Resource Update (Creature Sector)

\boxed{
F_i^{+}
=
F_i
-
\sum_{j\in\mathcal N_R(i)} J_{i\to j}\,\Delta\tau_i
+
I_{g\to i}
}

IV.3a Optional: Finite-Compressibility Floor Repulsion (Matter Stiffness)
⸻

IV.4 Momentum Dynamics (Optional Module)

Purpose: enable persistent, directed approach/collision dynamics in graph-native systems by introducing strictly local, antisymmetric bond momentum that stores a short-lived memory of diffusive flow.

Bond-local time step:
\[
\boxed{
\Delta\tau_{ij} \equiv \tfrac12(\Delta\tau_i+\Delta\tau_j)
}
\]

Per-bond momentum state:
\[
\boxed{
\pi_{ij}\in\mathbb R,\qquad \pi_{ij}=-\pi_{ji}
}
\]

Momentum update (inductive charging from diffusive flow only):
\[
\boxed{
\pi_{ij}^{+} = (1-\lambda_{\pi}\,\Delta\tau_{ij})\,\pi_{ij}
+ \alpha_{\pi}\,J^{(\mathrm{diff})}_{i\to j}\,\Delta\tau_{ij}
}
\]
Recommended numerical stability: clip \(|\pi_{ij}|\le \pi_{\max}\) (constitutive; must be reported).

Momentum-driven drift flux (F-weighted):
\[
\boxed{
J^{(\mathrm{mom})}_{i\to j} = \mu_{\pi}\,\sigma_{ij}\,\pi_{ij}\,\Big(\tfrac{F_i+F_j}{2}\Big)
}
\]

Properties:
• Strictly local and pairwise antisymmetric (hence conservative in closed systems, absent sources/sinks).
• F-weighted: vanishes in vacuum (prevents unphysical "free push").
• Not gated by agency (momentum acts on realized structure once created).
• \(\lambda_{\pi}\) controls ballistic (small) vs viscous (large) regimes.
• Momentum accumulates from \(J^{(\mathrm{diff})}\) only (prevents circular feedback momentum→flow→momentum).

Parameters: \(\alpha_{\pi}>0\) (accumulation), \(\lambda_{\pi}\ge 0\) (friction/decay), \(\mu_{\pi}\ge 0\) (coupling), and \(\pi_{\max}\) (clip) must be reported with any results.

Purpose: prevent unphysical infinite compression by introducing an agency-independent, local packing stiffness at high density. This module is OPTIONAL and must be explicitly declared when used.

Activation:
\[
s_i \equiv \left[\frac{F_i - F_{\text{core}}}{F_{\text{core}}}\right]_+^{p}
\qquad\text{with}\qquad [x]_+\equiv\max(0,x)
\]

Pairwise antisymmetric floor flux:
\[
\boxed{
J^{(\text{floor})}_{i\to j}
=
\eta_f\,\sigma_{ij}\,(s_i+s_j)\,(F_i-F_j)
}
\]

Notes:
• Floor repulsion is NOT gated by agency.
• The form above is strictly local and ensures \(J^{(\text{floor})}_{i\to j}=-J^{(\text{floor})}_{j\to i}\).
• Parameters \(\eta_f>0\), \(F_{\text{core}}>0\), \(p\ge 2\) are numerical/constitutive and must be reported with any results.

⸻

V. Boundary-Agent Operators (Law-Bound)

V.0 Phase Evolution (Canonical Minimal)

\[
\boxed{
\theta_i^{+} = \theta_i + \omega_0\,\Delta\tau_i \;(\bmod\;2\pi)
}
\]

Define coupling strength:
\[
\gamma_0 \ge 0 \quad \text{(phase-coupling gain; dimensionless)}
\]
\(\gamma_0\) controls the strength of neighbor phase-locking; set \(\gamma_0=0\) to disable coupling.

Optional coupling term (if enabled):
\[
\theta_i^{+} \leftarrow \theta_i^{+} +
\left[\sum_{j\in\mathcal N_R(i)} \gamma_0\,\sigma_{ij}\sqrt{C_{ij}}\,a_i a_j\,\sin(\theta_j-\theta_i)\right]\Delta\tau_i
\]

V.1 Agency Inviolability

\boxed{\text{Boundary operators cannot directly modify } a_i}

V.2 Local Dissipation

\boxed{
D_i \equiv \sum_{j\in\mathcal N_R(i)} |J_{i\to j}|\,\Delta\tau_i
}

V.3 Grace Injection (Local, Non-Coercive)

Need:
\boxed{n_i \equiv \max(0,\ F_{\min}-F_i)}
Weight:
\boxed{w_i \equiv a_i\,n_i}
Neighborhood normalizer:
Z_i \equiv \sum_{k\in\mathcal N_R(i)} w_k
Injection:
\boxed{
I_{g\to i}
=
D_i\;
\frac{w_i}{Z_i+\varepsilon}
}
If a_i=0, then I_{g\to i}=0.

V.4 Reconciliation (Bond Healing)

u_{ij}\equiv a_i a_j(1-C_{ij}),\qquad
Z^{C}_{ij}\equiv\sum_{(m,n)\in\mathcal E_R(i,j)} u_{mn}
Bond-local dissipation:
D_{ij}\equiv \tfrac12(D_i+D_j)
Healing:
\boxed{
\Delta C_{ij}^{(g)}
=
D_{ij}\;
\frac{u_{ij}}{Z^{C}_{ij}+\varepsilon}
}
Requires mutual openness.

⸻

VI. Agency Dynamics (Creature-Only)

\boxed{
a_i^{+}
=
\mathrm{clip}\!\left(
a_i + \big(P_i-\bar P_{\mathcal N(i)}\big) - q_i,\ 0,\ 1
\right)
}
where
\boxed{
\bar P_{\mathcal N(i)} \equiv \frac{1}{|\mathcal N_R(i)|}\sum_{k\in\mathcal N_R(i)} P_k
}

Agency Update Law (Model Family)

DET does not assume a unique agency update law. The canonical incremental rule above is the default. Alternative strictly local, non-coercive laws may be declared, provided they preserve the core monotone tendency that higher structural debt drives agency down.

Target-tracking variant (declared alternative):
\[
\boxed{
a_i^{+} = a_i + \beta\,(a_{\text{target},i} - a_i)
\qquad\text{where}\qquad
a_{\text{target},i} = \frac{1}{1 + \lambda\,q_i^2}
}
\]

Parameters: \(\beta\in(0,1)\) (response rate), \(\lambda>0\) (coupling strength). Any use of this variant must report \(\beta,\lambda\).

⸻

VII. Emergent Light and Relativity

\[
\boxed{
c_*(i) \equiv \left\langle \sigma_{ij}\right\rangle_{j\in\mathcal N_R(i)}
}
\]

In homogeneous regions where \(c_*(i)\) varies slowly, we write \(c_*\) for the local typical value.

\qquad
d\tau=d\tau_0\sqrt{1-\frac{v^2}{c_*^2}}
Light is the local causal horizon.

⸻

VIII. Emergent Gravity

\boxed{
\Phi_i=c_*^2\ln\!\left(\frac{M_i}{M_0}\right)
}

Source (see Appendix G):
\[
\boxed{
\rho_i \equiv q_i - b_i
}
\]

where \(b_i\) is the locally computed baseline structural field defined in Appendix G. Uniform or slowly varying background structure does not gravitate.

Field equation:
\boxed{
(L_\sigma\Phi)_i=-\kappa\rho_i
}
Boundary agent does not source gravity (no retained past).

VIII.2 Gravity–Flow Coupling (Canonical, Graph-Local)

DET specifies how the emergent gravity field \(\Phi\) couples to resource transport through an antisymmetric, bond-local drift flux:

\[
\boxed{
J^{(\text{grav})}_{i\to j}
=
\mu_g\,\sigma_{ij}\,\Big(\tfrac{F_i+F_j}{2}\Big)\,(\Phi_i-\Phi_j)
}
\]

Properties:
• Strictly local (depends only on \(i\), \(j\), and bond data).
• Pairwise antisymmetric, hence conservative in the absence of sources/sinks.
• Not gated by agency: gravity acts on realized structure regardless of openness.

Numerical stability (recommended, implementation-level): limit per-step transfer by bounding \(|J^{(\text{grav})}_{i\to j}|\) relative to local available resource (Appendix N).

⸻

IX. Decoherence

\boxed{
\lambda_{ij}=\lambda_{\mathrm{env}}+\left(\frac{v_{ij}-c_*}{c_*}\right)^2
}
For canonical closed systems: \lambda_{\mathrm{env}}=0.

⸻

X. Canonical Update Ordering (Core)

Each global step k uses synchronous evaluation at step k and updates to k+1:

1) Compute \(H_i\), then \(P_i\) and \(\Delta\tau_i=P_i\Delta k\)  
2) Compute \(\psi_i\), then \(J^{(\mathrm{diff})}_{i\to j}\); if momentum is enabled compute \(J^{(\mathrm{mom})}_{i\to j}\) from \(\pi_{ij}\) (IV.4); if gravity readout is enabled compute \(\Phi\) and then \(J^{(\mathrm{grav})}_{i\to j}\); if floor module is enabled compute \(J^{(\mathrm{floor})}_{i\to j}\); set \(J_{i\to j}=J^{(\mathrm{diff})}+J^{(\mathrm{mom})}+J^{(\mathrm{grav})}+J^{(\mathrm{floor})}\)  
3) Compute dissipation \(D_i\)  
4) Compute boundary injection \(I_{g\to i}\) and healing \(\Delta C_{ij}^{(g)}\)  
5) Update resources \(F_i^{+}\)  
5a) (Optional) Update momentum \(\pi_{ij}^{+}\) via IV.4 (uses \(J^{(\mathrm{diff})}\) and \(\Delta\tau_{ij}\); applies clip \(\pi_{\max}\) if used)  
6) Update structure \(q_i^{+}\) via the declared q-locking law  
7) Update agency \(a_i^{+}\)  
8) Update phase \(\theta_i^{+}\) (if enabled)  
9) (Optional) Solve baseline \(b_i\) and gravity \(\Phi_i\) for diagnostics / predictions; if enabled, use \(\Phi\) to compute \(J^{(\mathrm{grav})}\) (VIII.2)

XI. Falsifiers (Definitive)

The theory is false if any condition below holds under the canonical rules.

F1 — Locality / Embedding Invariance

Embedding a connected subgraph \mathcal C into a larger graph by adding nodes with no causal path to \mathcal C changes trajectories on \mathcal C beyond numerical tolerance.
(Your tests show exact zero when scope is enforced.)

F2 — Coercion

Any node with a_i=0 receives resource injection or bond healing from boundary operators.

F3 — Boundary Redundancy

With boundary operators enabled, outcomes are qualitatively indistinguishable from the boundary-disabled system across broad initial conditions (presence, structural debt, coherence).

F4 — No Regime Transition

Increasing mean openness \langle a\rangle fails to produce a transition from fragmented/low-coherence regimes to coherent/high-conductivity regimes.

F5 — Hidden Global Aggregates

Any lawful behavior depends on sums or averages taken outside \mathcal N_R or \mathcal E_R, or on reintroduced global normalizations.

F6 — Binding Failure (Scoped)

With (i) gravity–flow coupling enabled (VIII.2) and (ii) agency-gated diffusion enabled (IV.2), two initially separated compact bodies with nonzero realized structure (q>0 under the declared q-locking law) fail to form a bound state across a broad set of initial impact parameters (separation returns to its pre-collision value).

F7 — Mass Non-Conservation (Scoped)

Under the canonical conservative transport implementation (Appendix N; no hard vacuum clamping), total mass \(\sum_i F_i\) drifts by more than 10% over 1000 steps in closed-system tests (no net boundary injection).

F8 — Momentum Pushes Vacuum

With the momentum module enabled, initializing \(\pi\neq 0\) on bonds in a region with negligible resource \(F\approx 0\) produces sustained transport (nontrivial \(J^{(\mathrm{mom})}\)). Momentum-driven flow must vanish in vacuum due to F-weighting.

F9 — Spontaneous Drift from Symmetric Rest

With symmetric initial conditions and \(\pi_{ij}=0\) everywhere, the system develops a persistent net drift (center-of-mass motion in measurement readouts) beyond numerical tolerance without any stochastic terms or asymmetric boundary conditions.

F10 — Regime Discontinuity under \(\lambda_{\pi}\)

Scanning \(\lambda_{\pi}\) fails to produce a smooth ballistic→viscous transition in collision outcomes (e.g., min-separation, peak counts), instead showing discontinuous jumps not attributable to numerical stability limits or explicit bifurcation conditions.

⸻

XI. Predictive Claim (Plain)

Under strictly local evaluation, DET with law-bound boundary operators exhibits non-coercive recovery from structural freeze that cannot be reproduced by closed, purely naturalistic dynamics.

⸻

XII. Closing Statement
	•	Time is participation.
	•	Mass is frozen past.
	•	Gravity is memory imbalance.
	•	Quantum behavior is local coherence in flow.
	•	Agency is inviolable.
	•	Grace is local, lawful action.
	•	The future is open.

-------
# Appendix B — Black Hole & Matter Falsification Suite


Purpose:
To test whether the canonical DET equations necessarily produce black-hole–like behavior (collapse, accretion, stability, evaporation) without adding any new rules.

This appendix specifies:
	•	operational definitions,
	•	minimal update logic,
	•	and falsifiers.

⸻

B.1 Operational Definition (DET Black Hole)

A node i is a DET black hole if, over sustained evolution,

\boxed{
a_i \to 0,\qquad
q_i \to 1,\qquad
P_i \to 0
}

Interpretation:
	•	Agency collapse (a_i=0) gates off grace and coherence
	•	Structural dominance (q_i \approx 1) sources gravity
	•	Frozen presence (P_i\approx0) halts proper time

No exotic assumptions are made.

⸻

B.2 Minimal State Required for BH Tests

For each node i:

\{F_i,\ q_i,\ a_i,\ P_i,\ \theta_i\}

For each bond (i,j):

\{\sigma_{ij},\ C_{ij}\}

Neighborhoods \mathcal N_R(i) are defined exactly as in the core.

⸻

B.3 Minimal Update Loop (Canonical)

Each global step k:

Step 1 — Presence

P_i = a_i\,\sigma_i\;\frac{1}{1+F_i^{\mathrm{op}}}\;\frac{1}{1+H_i}

⸻

Step 2 — Flow

\psi_i=
\sqrt{\frac{F_i}{\sum_{k\in\mathcal N_R(i)}F_k+\varepsilon}}
e^{i\theta_i}

J_{i\to j}
=
\sigma_{ij}\!\left[
\sqrt{C_{ij}}\,\operatorname{Im}(\psi_i^*\psi_j)
+
(1-\sqrt{C_{ij}})(F_i-F_j)
\right]

⸻

Step 3 — Local Dissipation

D_i=\sum_{j\in\mathcal N_R(i)}|J_{i\to j}|\,\Delta\tau_i

⸻

Step 4 — Grace Injection (Boundary Operator)

n_i=\max(0,F_{\min}-F_i),\quad
w_i=a_i n_i

I_{g\to i}
=
D_i
\frac{w_i}{\sum_{k\in\mathcal N_R(i)}w_k+\varepsilon}

⸻

Step 5 — Resource Update

F_i^{+}=F_i-\sum_j J_{i\to j}\Delta\tau_i+I_{g\to i}

⸻

Step 6 — Structural Update (Canonical q-locking rule)

This rule is the default canonical member of the q-locking law family and is used for all validation, gravity, and black-hole tests unless explicitly replaced by an alternative local q-locking law.

q_i^{+}=\mathrm{clip}\!\left(q_i+\alpha_q\,\max(0,-\Delta F_i),\,0,\,1\right)

Interpretation: net inward flow increases frozen structure.

⸻

Step 7 — Agency Update (Creature-Only)

a_i^{+}=
\mathrm{clip}\!\left(
a_i+(P_i-\bar P_{\mathcal N(i)})-q_i,\ 0,\ 1
\right)

⸻

B.4 Test BH-1 — Single Black Hole Formation

Initial Condition
	•	One node i with:
	•	high q_i\approx0.8
	•	moderate a_i\approx0.4
	•	Neighbors with low q, moderate a

Expected Outcome

a_i(t)\to0,\quad
P_i(t)\to0,\quad
I_{g\to i}(t)\to0

Measure
	•	a_i(t), P_i(t)

Falsifier

If grace persists after a_i=0 → theory falsified

⸻

B.5 Test BH-2 — Accretion / Event Horizon

Initial Condition
	•	Central node: a=0,\ q\approx1
	•	Surrounding shell: a>0,\ q\ll1

Expected Outcome
	•	Net inward J_{j\to i}
	•	Radial gradient:
P(r)\downarrow,\quad q(r)\uparrow
	•	Stable “event horizon” where P\approx0

Measure
	•	Flow vectors
	•	Radial profiles of P,q

Falsifier

If sustained outward flow occurs without agency recovery → falsified

⸻

B.6 Test BH-3 — Evaporation (Hawking-like)

Initial Condition
	•	Black hole with a_i\approx0 (not exactly zero)

Add minimal stochastic perturbation:
a_i^{+}=a_i+\epsilon\,\xi_i,\quad \xi_i\sim\mathcal N(0,1)

Expected Outcome
	•	Most perturbations decay
	•	Rare a_i>0 excursions
	•	Brief grace injection
	•	Small outward J bursts

Measure
	•	Correlation between a_i spikes and F_i loss

Falsifier

If evaporation occurs with a_i=0 throughout → falsified

⸻

B.7 Dark Matter Test

Hypothesis

Nodes with:
a_i=0,\quad C_{ij}\approx0
interact only via gravity.

Test
	•	Track lensing / attraction effects from such nodes
	•	Confirm zero response to grace or coherence channels

Falsifier

If non-gravitational interaction occurs → dark-matter analogue fails

⸻

B.8 Information Retention Test

Hypothesis

Information is locked in q_i when a_i=0.

Test
	•	Recover a_i>0 artificially
	•	Observe release of previously frozen structure

Falsifier

If information disappears irreversibly → falsified

⸻

B.9 Summary of Black Hole Falsifiers

BH-F1 — Grace reaches a=0
BH-F2 — Evaporation without agency recovery
BH-F3 — Instability without structural cause
BH-F4 — Dark-matter behavior reproducible without a=0

		Any violation falsifies the DET black-hole interpretation.

Appendix G — Gravity Source Correction (DET 4.2a)

(Monopole Preservation & Baseline-Referenced Gravity)

⸻

G.1 Motivation

Numerical testing of the DET 4.2 gravity sector in 3D discrete networks reveals a structural issue with the original gravity source definition when applied to compact mass distributions.

The prior definition,
\rho_i \equiv q_i - \bar q_{\mathcal N(i)},
acts as a near-field high-pass filter.
In three spatial dimensions, this construction generically enforces:
\sum_i \rho_i \approx 0
for any sufficiently smooth or compact region of elevated q.

As a consequence, the emergent gravitational potential lacks a monopole term and fails to produce a long-range 1/r far field, yielding instead a short-range or multipolar decay.

Since a nonzero monopole is required for Newtonian-like gravity in 3D, this behavior constitutes a physical failure mode, not a numerical artifact.

⸻

G.2 Corrected Principle (Clarification)

Corrected statement:

Emergent gravity in DET is sourced by deviations from a local environmental baseline, not by deviations from an agent’s immediate neighborhood mean.

Uniform or slowly varying background structure does not gravitate; contrast relative to baseline does.

This preserves the original DET intuition (“uniform structure does not curve”) while restoring physical viability in three dimensions.

⸻

G.3 Baseline Field Definition (New Primitive)

Scope note:
This appendix corrects the gravity source term and baseline construction only.
It does not define or modify q-formation.
All gravity results assume a declared q-locking law from the q-locking family (default: Appendix B), applied consistently across simulations.

Introduce a baseline structural field b_i, defined locally by a Helmholtz-type smoothing equation:

\boxed{
(L_\sigma b)_i - \alpha\,b_i = -\alpha\,q_i
}

where:
	•	L_\sigma is the weighted graph Laplacian,
	•	\alpha > 0 sets the baseline smoothing scale,
	•	the characteristic baseline length is \ell_b \sim \alpha^{-1/2}.

Interpretation:
b_i represents the locally accessible “environmental background” of accumulated structure q, computed strictly through local interactions and lawful diffusion.
No global averages are introduced.

⸻

G.4 Corrected Gravity Source

Replace the previous source definition with:

\boxed{
\rho_i \equiv q_i - b_i
}

This construction:
	•	preserves locality,
	•	allows compact sources to retain a nonzero monopole,
	•	prevents uniform background structure from sourcing gravity.

⸻

G.5 Emergent Gravity Equation (Updated)

The gravity field equation becomes:

\boxed{
(L_\sigma \Phi)_i = -\kappa\,\rho_i
}
\quad\text{with}\quad
\rho_i = q_i - b_i

All subsequent uses of \rho_i in DET 4.2 should be understood in this corrected sense.

⸻

G.6 3D Viability Criterion (Explicit)

For a physically viable 3D gravity sector, DET requires:

\boxed{
\sum_i \rho_i \neq 0
\quad\Rightarrow\quad
\Phi(r) \sim -\frac{A}{r}
\;\;\text{in the far field}
}

The former neighborhood-mean definition (q_i-\bar q_{\mathcal N}) generically violates this condition for compact sources and is therefore disallowed as a universal gravity source in 3D.

⸻

G.7 Empirical Status

Numerical tests using:
	•	dynamically formed q (not injected),
	•	locally computed b,
	•	and the corrected source \rho=q-b,

demonstrate:
	•	recovery of a 1/r far-field potential in 3D,
	•	linear scaling of the monopole with total accumulated structure,
	•	monotone clock-rate behavior consistent with gravitational redshift under the DET clock–presence mapping.

Details of these tests are provided in the DET 4.2 numerical validation suite.

⸻

G.8 Falsifiability (New)

DET 4.2a gravity is falsified if:
	1.	No local choice of \alpha yields a stable monopole in 3D networks.
	2.	The corrected source \rho=q-b fails to produce a 1/r far field for compact bodies.
	3.	Baseline subtraction introduces nonlocal dependencies or requires global normalization.
	4.	Dynamic q formation combined with baseline referencing systematically destroys long-range gravity.

⸻

G.9 Status

This appendix does not alter:
	•	the definition of agency a,
	•	the presence–mass relation M=P^{-1},
	•	the clock-rate interpretation,
	•	or the boundary coupling logic.

It corrects only the gravity source term to ensure physical consistency in three dimensions.

⸻

End Appendix G

⸻

DET 4.2a — Numerical Derivation Card

From Local Update Rules to Measurable Phenomena

⸻

0. Scope of This Card

This card documents numerical derivations that are now possible in DET without importing external physics laws, beyond unit calibration.
All results arise from local update rules, baseline-referenced gravity, and discrete network dynamics.

This is not a phenomenological summary; it is a derivation and test capability statement.

Note on q-locking:
Unless explicitly stated otherwise, all numerical derivations assume the canonical q-locking law from Appendix B.3.
Any alternative q-locking rule must be declared explicitly and used consistently across the full loop (transport, locking, baseline, gravity).

⸻

1. Primitive Quantities (Numerical State)

Per node i:
	•	F_i \ge 0 — free resource (transported, conserved up to sinks/sources)
	•	q_i \ge 0 — structural debt (retained past)
	•	b_i \ge 0 — baseline structural field (local environmental background)
	•	\rho_i = q_i - b_i — gravity source
	•	\Phi_i — emergent gravitational potential
	•	P_i — presence / local clock rate
	•	M_i = P_i^{-1} — effective mass

Graph primitives:
	•	L_\sigma — weighted graph Laplacian
	•	adjacency radius / coupling weights (local only)

Coordination Load (H)

Coordination load H_i is defined exactly as in the core (III.0) and enters the clock law through P_i.
For numerical derivations, either option may be used, but the choice must be stated.

Option A — Degenerate (Ablation / Minimal)
\[
\boxed{H_i \equiv \sum_{j\in\mathcal N_R(i)} \sigma_{ij}}
\qquad\Rightarrow\qquad
H_i=\sigma_i
\]

Option B — Recommended (Coherence-Weighted Load)
\[
\boxed{H_i \equiv \sum_{j\in\mathcal N_R(i)} \sqrt{C_{ij}}\,\sigma_{ij}}
\]

⸻

⸻

2. Canonical Numerical Update Loop (Exact Core Loop)

All numerical derivations in this card are computed from the same lawful update loop as the core theory.
No alternative transport equation (e.g., pure diffusion) is assumed unless explicitly declared as an ablation.

Each global step k uses synchronous evaluation at step k and updates to k+1:

Step A — Coordination + Presence

Compute coordination load H_i (core III.0), then:
\[
P_i = a_i\,\sigma_i\;\frac{1}{1+F_i^{\mathrm{op}}}\;\frac{1}{1+H_i}
\qquad,\qquad
\Delta\tau_i = P_i\,\Delta k
\]

Step B — Wavefunction + Flow

\[
\psi_i=
\sqrt{\frac{F_i}{\sum_{k\in\mathcal N_R(i)}F_k+\varepsilon}}
\,e^{i\theta_i}
\]

\[
J_{i\to j}
=
\sigma_{ij}\!\left[
\sqrt{C_{ij}}\,\operatorname{Im}(\psi_i^*\psi_j)
+ 
(1-\sqrt{C_{ij}})(F_i-F_j)
\right]
\]

If dimensional strictness is required, replace the classical term \((F_i-F_j)\) with \((\tilde F_i-\tilde F_j)\) where \(\tilde F_i \equiv \frac{F_i}{\sum_{k\in\mathcal N_R(i)}F_k+\varepsilon}\), keeping the coherence term unchanged.

Step C — Dissipation

\[
D_i=\sum_{j\in\mathcal N_R(i)}|J_{i\to j}|\,\Delta\tau_i
\]

Step D — Boundary Operators (Grace + Healing)

Need:
\[
n_i=\max(0,F_{\min}-F_i)
\qquad,\qquad
w_i=a_i n_i
\qquad,\qquad
Z_i=\sum_{k\in\mathcal N_R(i)}w_k
\]

Injection:
\[
I_{g\to i}
=
D_i
\frac{w_i}{Z_i+\varepsilon}
\]

Bond healing (optional unless stated):
\[
u_{ij}=a_i a_j(1-C_{ij})
\qquad,\qquad
Z^{C}_{ij}=\sum_{(m,n)\in\mathcal E_R(i,j)}u_{mn}
\qquad,\qquad
D_{ij}=\tfrac12(D_i+D_j)
\]
\[
\Delta C_{ij}^{(g)}
=
D_{ij}
\frac{u_{ij}}{Z^{C}_{ij}+\varepsilon}
\]

Step E — Resource Update

\[
F_i^{+}=F_i-\sum_{j\in\mathcal N_R(i)} J_{i\to j}\,\Delta\tau_i+I_{g\to i}
\]

Step F — Structural Locking (Canonical q-locking)

\[
q_i^{+}=\mathrm{clip}\!\left(q_i+\alpha_q\,\max(0,-\Delta F_i),\,0,\,1\right)
\]

Step G — Agency Update (Creature-Only)

\[
a_i^{+}
=
\mathrm{clip}\!\left(
a_i+(P_i-\bar P_{\mathcal N(i)})-q_i,\ 0,\ 1
\right)
\qquad,\qquad
\bar P_{\mathcal N(i)}=\frac{1}{|\mathcal N_R(i)|}\sum_{k\in\mathcal N_R(i)}P_k
\]

Step H — Phase Update (if enabled)

\[
\theta_i^{+}=\theta_i+\omega_0\,\Delta\tau_i\ (\bmod\ 2\pi)
\]
Optional coupling term:
\[
\theta_i^{+}\leftarrow\theta_i^{+}+
\left[\sum_{j\in\mathcal N_R(i)}\gamma_0\,\sigma_{ij}\sqrt{C_{ij}}\,a_i a_j\,\sin(\theta_j-\theta_i)\right]\Delta\tau_i
\]

Step I — Baseline, Gravity, and Clock-Rate Mapping (Diagnostic / Predictive Readout)

These steps do not define dynamics; they compute measurable fields from the realized q field.

Consistency check (recommended):
Compute \(P_i\) directly from the core clock law (Step A) and independently from the gravity readout mapping (Step I) using
\[
P_i^{(\Phi)} \equiv \exp\!\left(-\frac{\Phi_i-\Phi_0}{c_*^2}\right).
\]
Agreement up to a local reference choice \(\Phi_0\) is expected only when the gravity–mass mapping is enabled as a diagnostic and the chosen \(c_*\) calibration is consistent; discrepancies are a measurable signal that the readout mapping or calibration requires adjustment.

Baseline field:
\[
(L_\sigma b)_i-\alpha b_i=-\alpha q_i
\]

Gravity source:
\[
\rho_i=q_i-b_i
\]

Gravity field:
\[
(L_\sigma\Phi)_i=-\kappa\rho_i
\]

Time / mass mapping (derived, up to reference choice):
\[
\Phi_i=c_*^2\ln\!\left(\frac{M_i}{M_0}\right),\quad M_i=P_i^{-1}
\quad\Rightarrow\quad
P_i \propto \exp\!\left(-\frac{\Phi_i}{c_*^2}\right)
\]
For numeric readout use:
\[
P_i = \exp\!\left(-\frac{\Phi_i-\Phi_0}{c_*^2}\right)
\]
where \Phi_0 is a local reference (e.g., neighborhood median).

⸻

3. Numerically Derived Laws (No Assumption)

The following are derived outcomes, not imposed:

3.1 Newtonian Kernel (3D)

From compact dynamically formed q:

\Phi(r) \sim -\frac{A}{r} + B
\quad \text{(far field)}

Derivation:
Emerges only when \sum \rho_i \neq 0; fails if neighborhood-mean subtraction is used.

⸻

3.2 Gravitational Time Dilation

From Step E:

\frac{\Delta f}{f}
\approx
\frac{\Delta \Phi}{c_*^2}

After calibrating c_* \rightarrow c, numerical values match measured redshift scaling near Earth.

⸻

3.3 Shell / Ring Mass Structures (Novel)

Under declared q-locking laws beyond the canonical \max(0,-\Delta F_i) rule (e.g., gradient- or shear-driven locking variants):
	•	spherical sources → mass shells
	•	disk sources → ring overdensities

This is a nontrivial DET prediction, not present in GR or CDM.

⸻

3.4 Rotation Curve Shoulders

From radial acceleration:
g(r) = -\frac{d\langle\Phi\rangle}{dr}
\quad,\quad
v_{\text{circ}}(r)=\sqrt{r g(r)}

Derived features:
	•	shoulders / kinks aligned with ring radii
	•	Keplerian falloff outside dominant shell(s)

⸻

3.5 Lensing Proxy (Projected Density)

\Sigma(R)=\sum_z \rho(x,y,z)

Derived:
	•	ring-like convergence features
	•	correlation between \Sigma(R) peaks and v(r) shoulders

⸻

4. What Is Calibrated vs What Is Derived

Quantity	Status
1/r gravitational kernel	Derived
Clock redshift scaling	Derived
Shell/ring formation	Derived
Far-field monopole existence	Derived
c_* scale	Calibrated
\kappa (gravity strength)	Calibrated
Absolute G	Pending extraction

DET now sits at Stage 2 physics: one-scale calibration → multi-observable prediction.

⸻

5. Numerical Falsifiers (Operational)

DET fails numerically if:
	1.	No choice of \alpha yields a stable monopole in 3D.
	2.	Dynamically formed q never produces long-range 1/r fields.
	3.	Baseline field introduces nonlocal dependence.
	4.	Clock-rate mapping fails monotonicity.
	5.	Ring structures are unstable under small parameter perturbations.

All five are testable in simulation.

⸻

6. What Can Be Done Next (Now Possible)

Because of the above, DET can now numerically attempt:
	1.	Effective Newton constant extraction
G_{\text{eff}} \sim \kappa / c_*^2
	2.	Galaxy rotation-curve fitting using shell/ring profiles
	3.	Ring-galaxy / lensing-ring comparisons
	4.	Collapse vs stabilization phase diagrams
	5.	Hoarder → black-hole transition tests
	6.	Clock-network universality bounds

⸻

Appendix N — Numerical Implementation Notes (Non-Formal)

This appendix records implementation-level constraints required for stable, conservative simulation of the canonical DET laws. These are not additional physics claims; they are numerical scheme requirements.

N.1 Positivity and Conservative Outflow Limiting (Recommended)

The explicit update
\[
F_i^{+}=F_i-\sum_j J_{i\to j}\,\Delta\tau_i + I_{g\to i}
\]
must preserve \(F_i^{+}\ge 0\). Avoid hard vacuum clamping (which injects mass). Instead, conservatively limit outgoing flux so that total outflow in one step does not exceed available resource.

One simple conservative limiter:
• compute proposed outgoing amount \(O_i\equiv\sum_{j: J_{i\to j}>0} J_{i\to j}\,\Delta\tau_i\)
• if \(O_i>\eta\,F_i\) (with \(\eta<1\)), scale all positive outgoing terms from \(i\) by factor \(\eta F_i / O_i\)

N.2 Gravity Flux Limiting (Recommended)

Large \(|\Phi_i-\Phi_j|\) can cause overshoot in explicit schemes. Bound per-step gravitational transfer on each bond, e.g.
\[
|J^{(\mathrm{grav})}_{i\to j}|\,\Delta\tau_i \le \eta_g\,F_i
\]
with \(\eta_g\in(0.01,0.05)\) reported.

N.3 Time Step Guidance (Implementation)

For explicit Euler stability, choose \(\Delta k\) so that local Courant-like limits are respected, e.g.
\[
\Delta k \lesssim \min\Big(\frac{1}{\max\sigma_{ij}},\;\frac{1}{\max |\Phi_i-\Phi_j|}\Big)
\]
evaluated on the relevant local neighborhood scales.

N.4 Diagnostics for Binding Equilibria

In bound steady states, expect approximate cancellation:
\[
J^{(\mathrm{diff})}+J^{(\mathrm{floor})}+J^{(\mathrm{grav})}\approx 0
\]
with small residual transport and bounded mass error under the conservative limiter.

Appendix X — Boundary Ontology Clarification and Forward Migration (Informative)

	•	In DET 4.2, the Boundary was treated as an external reservoir/operator for mathematical convenience.
	•	References to “Boundary agency” were metaphorical, not a claim that the Boundary is a node or agent within the creature state space.
	•	Quantities defined on nodes (e.g., a_i, P_i, q_i) do not extend to the Boundary.
	•	Boundary action in 4.2 is limited to lawful, local operators (e.g., healing/venting channels) and never violates agency inviolability.
	•	DET v5 will formalize this distinction explicitly by placing the model within the Boundary rather than the Boundary within the model.
	•	No equations or predictions in DET 4.2 depend on treating the Boundary as an in-model agent.

⸻

DET v5 Revision Patch: Local Angular Momentum Module (Plaquette Spin)

Patch intent: Add a strictly local, conservative angular momentum degree of freedom that (i) emerges from linear momentum circulation, (ii) produces divergence-free rotational flow, and (iii) enables stable orbital capture without any global rotation center.

⸻

IV.4b Plaquette Angular Momentum (NEW)

State (local, discrete)
On a cubic lattice, define face/plaquette angular momentum L on oriented plaquettes:
	•	2D (XY grid): one field L_{xy}(i,j) (out-of-plane “spin”).
	•	3D: three fields L_{xy}, L_{yz}, L_{xz} (one per plane orientation).

These are not node properties and do not require a global origin; they represent local circulation memory.

⸻

IV.4b.1 Charging Law (emergent from \pi)

Angular momentum is induced by the discrete curl of the linear bond momentum \pi.

2D (XY) canonical curl for a plaquette with top-left corner at (i,j), using directed bond momenta \pi_x (east) and \pi_y (south):

\mathrm{curl}_\pi(i,j)
=
\pi_x(i,j) + \pi_y(i, j{+}1) - \pi_x(i{+}1,j) - \pi_y(i,j)

Presence-clocked update:

\boxed{
L_{xy}^{+}
=
\left(1-\lambda_L\,\Delta\tau_{\square}\right)L_{xy}
+
\alpha_L\,\mathrm{curl}_\pi\,\Delta\tau_{\square}
}

where \Delta\tau_{\square} is the plaquette proper time, canonically the average of the four corner node clocks:

\Delta\tau_{\square}
=
\frac14\sum_{v\in \square}\Delta\tau_v

3D: Apply the same rule per face orientation, e.g. L_{xy} charged by the curl of (\pi_x,\pi_y) around XY faces; similarly for L_{yz} and L_{xz}.

Constraints:
	•	Strict locality: uses only bonds around the plaquette.
	•	Emergence: depends only on \pi, no injected spin.
	•	Presence-clocked: uses \Delta\tau, never global dt.

⸻

IV.4b.2 Rotational Flux (conservative circulation)

Angular momentum produces a rotational flux J^{rot} that is divergence-free by construction (i.e., swirl-only; no net source/sink of F).

2D canonical form (recommended): rotational flux on edges is the perpendicular gradient of L:

\boxed{
J^{rot}_{x}(i,j) = \mu_L\,\sigma_{ij}\,F^{avg}_{x}(i,j)\,\Big(L(i,j)-L(i{-}1,j)\Big)
}

\boxed{
J^{rot}_{y}(i,j) = -\mu_L\,\sigma_{ij}\,F^{avg}_{y}(i,j)\,\Big(L(i,j)-L(i,j{-}1)\Big)
}

where F^{avg} is the local edge average (e.g. F^{avg}_x=\tfrac12(F_{i,j}+F_{i,j+1})), and \sigma_{ij} is the bond processing rate (any strictly local choice consistent with the rest of the transport law).

Total flux update:
\boxed{
J_{i\to j}^{total} = J^{diff}_{i\to j} + J^{mom}_{i\to j} + J^{floor}_{i\to j} + J^{rot}_{i\to j}
}

Notes:
	•	The difference-of-adjacent-plaquettes form is required because each edge is shared by two plaquettes; this guarantees conservative circulation on the lattice.
	•	The F^{avg} factor ensures no vacuum push: rotational transport vanishes when F\approx F_{vac}.

3D: J^{rot} is obtained from the curl of face-spins L_{xy},L_{yz},L_{xz} onto edges, again using only adjacent faces/edges (strictly local) and optionally weighted by F^{avg}.

⸻

IV.4b.3 Placement in the Canonical Update Loop

Insert after linear momentum update (so spin is “charged” by the latest circulation):
	1.	Compute P_i,\Delta\tau_i
	2.	Compute J components
	3.	Dissipation
	4.	Update F using sender-clocked transport
	5.	Update \pi (linear momentum)
5b. Update L (plaquette angular momentum) ← NEW
	6.	Update q
	7.	Update a
	8.	Update \theta

⸻

Falsifiers (NEW)

F_L1: Rotational Flux Conservation

With J^{rot} enabled and all other flux components disabled, a closed system must preserve:
	•	total F (mass conservation), and
	•	node-wise F must not develop net drift except as pure circulation consistent with periodic boundaries.

Failure implies J^{rot} is not divergence-free (non-conservative).

F_L2: Vacuum Spin Does Not Transport

Initialize L\neq 0 but set F=F_{vac} everywhere. Require:
	•	\max|J^{rot}| \approx 0,
	•	negligible mass drift.
Failure implies “spin pushes vacuum” (disallowed).

F_L3: Orbital Capture (Past-Resolved)

For two packets with non-zero impact parameter:
	•	separation remains bounded for long durations,
	•	relative angle winds (true circulation),
	•	no secular runaway in separation.
Failure implies lack of stable bound orbit dynamics.

⸻

Parameter Summary (minimal additions)
	•	\alpha_L: spin charging rate
	•	\lambda_L: spin decay / friction
	•	\mu_L: spin-to-flux coupling
	•	optional clip L_{max} for numerical stability (diagnostic, not a physics postulate)
