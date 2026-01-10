Deep Existence Theory (DET) v5 — Research Roadmap

Focus: Measurability, Intrinsic Dynamics, and First-Principles Derivation

0. Executive Summary

DET v4.2 successfully demonstrated that gravity and time dilation can emerge from scalar agent interactions. DET v5 aims to "close the loop" by removing external parameters and ensuring all dynamics—including phase evolution, coherence changes, and motion—arise solely from local agent rules.

1. Measurability & Fundamental Variables

1.1 The Sigma Problem ($\sigma_i$)

Current Status: $\sigma_i$ (processing rate) is largely a static parameter or an arbitrary inherent property.
The Problem: As you noted, "present moment is unfalsifiable." We need $\sigma_i$ to be an observable.
Proposed Definition: $\sigma_i$ is Active History Density.
Instead of a fixed parameter, $\sigma_i$ is the derivative of the Event Count ($k$) with respect to the graph's "consensus" time, or simply the capacity derived from previous successful participation.

Hypothesis: $\sigma_i(t) \propto \log(1 + k_i(t))$

Interpretation: An agent's processing rate is determined by the weight of its past experience (records).

Measurability: $\sigma_i$ becomes a measurable "event horizon" of the agent—the size of its history.

1.2 Intrinsic Coherence Dynamics ($C_{ij}$)

Current Status: $C_{ij}$ is modified by Boundary Healing (Grace) or treated as static. It lacks "creature-only" evolution.
The Problem: Bonds must evolve based on usage.
Proposed Mechanism: Hebbian Phase-Locking.
Bonds should strengthen when flow is coherent (synchronous) and atrophy when idle or incoherent.

Proposed Update Rule:


$$\Delta C_{ij} = \alpha_{\text{learn}} \cdot |J_{i\to j}| \cdot \cos(\theta_i - \theta_j) - \lambda_{\text{decay}} \cdot C_{ij}$$

Effect: If flow $J$ is high and phases align ($\cos \approx 1$), the channel widens ($C \uparrow$).

Effect: If flow is high but chaotic (random phases), growth is dampened.

Effect: If flow is zero, the bond decays (Entropy).

1.3 Phase Evolution ($\theta_i$)

Current Status: $\psi$ calculates flow, but nothing calculates $\psi$.
The Problem: What updates the phase? In QM, Energy drives phase rotation ($E = \hbar\omega$).
Proposed Mechanism: Resource-Driven Frequency.
We identify Resource $F_i$ (free energy) as the driver of the internal clock's rotation.

Proposed Update Rule:


$$\theta_i(t+1) = \theta_i(t) + \beta \cdot F_i(t) \cdot \Delta \tau_i$$

Implication: High-resource agents "vibrate" faster.

Emergence: This naturally leads to interference. Two agents with different $F$ levels will have rotating phases that drift in and out of sync, modulating the flow $J_{i \to j}$ (AC current behavior).

2. Emergent Motion & Vector Components

Current Status: DET is purely topological (graph-based). "Motion" is inferred from changing neighbors, but there is no velocity vector $\vec{v}$.
The Goal: Define motion without assuming an embedding space.

2.1 Motion as Graph Drift

Motion is not the displacement of a node, but the bias in bond updates.

Definition: Velocity $\vec{v}_i$ is the weighted asymmetry of the bond strengths in the local neighborhood.


$$\vec{v}_i \sim \sum_{j \in \mathcal{N}} (C_{ij} \cdot \hat{u}_{ij})$$


(Where $\hat{u}_{ij}$ is a virtual direction derived from the gradient of gravity/mass).

2.2 The Flow Vector

Instead of tracking node positions, we track the Resource Flux Vector:


$$\vec{J}_{\text{net}} = \sum_{j} J_{i \to j}$$

Newton's 2nd Law Analog: A change in stored Resource ($F_i$) leads to a change in the flux vector.


$$\frac{d\vec{J}}{d\tau} \propto -\nabla \Phi$$


(Flux accelerates down gradients of potential).

3. Derivation of Constants (From "Inspired" to "Derived")

Current Status: $c_*$, $G$ ($\kappa$), and $\hbar$ are tunable parameters.

3.1 The Speed of Light ($c$)

Concept: $c$ should not be an average $\langle \sigma \rangle$. It should be the Signal Propagation Limit.

Derivation: In a discrete graph, information moves 1 edge per tick. If we define "physical distance" based on the inverse of bond strength ($d_{ij} \sim 1/C_{ij}$), then $c$ emerges from the maximum possible conductance.

Target: Show that no causal influence can exceed 1 hop/tick, which scales to $c$ in the continuum limit.

3.2 Planck's Constant ($h$)

Concept: $h$ is the exchange rate between Energy (Resource) and Frequency (Phase Speed).

Derivation: From Section 1.3, we proposed $\Delta \theta = \beta \cdot F$.

Result: $\beta$ is the inverse of Planck's constant ($1/\hbar$). We can set $\beta=1$ by choosing "natural DET units," effectively deriving $h$ as a scaling factor of the simulation.

3.3 Gravitational Constant ($G$)

Concept: $G$ measures how much "Geometry" (Mass/Connectivity) curves in response to "Stress" (Resource).

Derivation: In v4.2, $\Phi \propto \ln(M)$. In v5, we want to derive this from the cost of maintaining bonds. If maintaining a bond requires Resource (Binding Energy), then high-mass nodes (high connectivity debt) effectively "pull" resource from the vacuum, creating the attractive potential.

4. Standard Model Derivations (Long Term)

4.1 Particles as Topological Defects

Fermions: Stable "knots" in the phase field $\theta$. If the phase winds around a center (vortex) and cannot unwind due to topology, that is a stable particle (charge).

Bosons: Excitations of the bonds $C_{ij}$ themselves (phonons on the graph).

4.2 The "Generations" Problem

Why are there 3 generations of matter?

DET Hypothesis: It relates to the dimensionality of the local graph neighborhood (e.g., node degree saturation).

5. Summary of v5 Action Items

Implement Phase Update: Add $\Delta \theta = F \cdot \Delta \tau$ to the numerical core.

Implement Dynamic Bonds: Add Hebbian update for $C_{ij}$.

Run "Two-Slit" Test: See if the new Phase + Dynamic Bond rules spontaneously create interference patterns in flow.

Derive $\sigma$: Replace fixed sigma with a history-dependent counter.