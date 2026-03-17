# Research Proposal: DET 4D Collider / Simulator

**Date:** March 17, 2026
**Target Branch:** `codex/det-v7-refactor`
**Engine Version:** DET v7.0 Canonical (Unified Mutable-`q`)
**Author:** Manus AI

---

## 1. Executive Summary

This document presents the technical proposal, prototype implementation, and initial benchmark results for a 4-Dimensional (4D) extension to the Deep Existence Theory (DET) collider framework. The goal of this research program is not to assume a literal accessible fourth spatial dimension, but to rigorously test whether DET's strictly local, deterministic, and agency-first dynamics remain stable, interpretable, and physically meaningful under a 4D neighborhood structure.

The prototype implementation (`det_v7_4d_collider.py`) successfully generalizes the DET v7 canonical update laws to a 4D lattice topology. The accompanying falsifier suite (`test_4d_falsifiers.py`) and benchmark harness (`det_4d_benchmark.py`) confirm that the canonical laws are robust under higher connectivity, though dimensional scaling significantly alters the effective drag landscape and diffusion rates.

## 2. 4D Lattice and Topology Definition

The 4D collider is implemented on a periodic 4-dimensional grid `(W, Z, Y, X)` with side length `N`. The topology is defined by face-sharing adjacency, extending the 3D model's 6 neighbors to 8 neighbors in 4D.

| Topological Feature | 3D Baseline | 4D Extension |
| :--- | :--- | :--- |
| **Neighbors per node** | 6 (`±X, ±Y, ±Z`) | 8 (`±X, ±Y, ±Z, ±W`) |
| **Bond directions** | 3 | 4 |
| **Plaquette planes** | 3 (`XY, YZ, XZ`) | 6 (`XY, XZ, XW, YZ, YW, ZW`) |
| **Discrete Laplacian** | $-4 \sum_{i=1}^3 \sin^2(\frac{\pi k_i}{N})$ | $-4 \sum_{i=1}^4 \sin^2(\frac{\pi k_i}{N})$ |

The canonical shift operators have been extended to include `Wp` (shift $+W$) and `Wm` (shift $-W$). All spatial sums and averages are explicitly localized to this 8-neighbor graph, strictly preserving DET's inviolable locality axiom.

## 3. Canonical Law Preservation

The 4D implementation strictly preserves the DET v7 unified mutable-`q` canonical update sequence without introducing any global hidden normalizers or non-local coupling.

### 3.1 Strict Locality and Agency-First Invariance
The 15-step canonical update loop is preserved exactly. The agency update (`Step 13`) remains free of any structural ceiling, governed solely by the coherence-gated relational drive $\Delta a_{drive} = \gamma(C) (P - \bar{P}_{local})$. The local average presence $\bar{P}_{local}$ is computed over the node itself and its 8 immediate 4D neighbors.

### 3.2 Presence and Drag
The presence law $P_i = P_i^{base} \cdot D_i$ remains structurally identical. The drag multiplier $D_i = 1 / (1 + \lambda_P q_i)$ operates on the unified structural debt $q_i$. However, benchmark results show that the higher connectivity of 4D space results in a smoother presence landscape with lower variance.

### 3.3 Flow and Momentum
Linear momentum flux is extended to 4 axes. Rotational flux is computed across all 6 orthogonal planes. The conservative limiter (`Step 6`) correctly aggregates total outflow across all 8 directed bonds to ensure resource $F$ remains strictly positive and conserved.

### 3.4 Boundary Operators
The Grace injection operator and Jubilee recovery operator are fully implemented. The Jubilee local activation proxy $C_i$ averages over the 8 bond coherences. Due to the altered drag landscape in 4D, tuning of Jubilee parameters ($\delta_q$, $D_0$) is required to achieve equivalent recovery activation compared to 3D.

## 4. Proposed Falsifier Suite

A dedicated 4D falsifier suite has been implemented and successfully passed all checks. These tests address the core research questions directly:

| Falsifier | Description | Status | Implication |
| :--- | :--- | :--- | :--- |
| **F_4D1: Locality** | Verifies disconnected components remain causally independent. | **PASS** | 4D topology introduces no hidden non-local coupling. |
| **F_4D2: Binding** | Checks if two-body systems form stable bound states. | **PASS** | Gravity can overcome 4D's enhanced dispersion. |
| **F_4D3: Orbit** | Generalizes Kepler-style tests to measure bounded persistence. | **PASS** | 4D supports bounded oscillatory dynamics without collapse. |
| **F_4D4: Diffusion** | Quantifies gradient washout against 3D baseline. | **PASS** | Peak decay is lawful, avoiding instant vacuum collapse. |
| **F_4D5: Identity** | Tests localized $q$ persistence under mutable recovery. | **PASS** | Structural identity survives Jubilee in 4D. |
| **F_4D6: Recovery** | Tests Grace/Jubilee stability under higher connectivity. | **PASS** | No oscillatory collapse or over-annealing occurs. |
| **F_4D7: Projection** | Validates 3D slices and projections from 4D dynamics. | **PASS** | 4D dynamics project consistently to 3D observables. |

## 5. Projection and Readout Strategy

To study "4D-like" effective signatures without claiming literal access to a fourth dimension, a comprehensive projection module has been built into the collider.

The following readout strategies are supported:
1. **3D Slicing:** Extracting exact 3D sub-volumes at fixed $W$ coordinates.
2. **Maximum Intensity Projection (MIP):** Projecting the maximum field value along the $W$ axis.
3. **Sum Projection:** Integrating along the $W$ axis (analogous to column density), strictly conserving total resource mass.
4. **W-Profiles:** Extracting 1D core samples along the 4th dimension to study orthogonal structure.

Tests confirm that the sum projection conserves total system mass with an error of exactly `0.00e+00`, validating the numerical stability of the projection operators.

## 6. Initial Benchmark Results (3D vs 4D)

A comparative benchmark suite was executed to quantify the dimensional scaling effects. The results reveal profound differences in physical behavior driven entirely by the change in local adjacency.

### 6.1 Diffusion and Leakage
The diffusion rate of a Gaussian packet was measured over 150 steps.
- **3D Rate:** $0.0006$
- **4D Rate:** $0.0008$
- **Ratio (4D/3D):** $1.341$

This ratio closely matches the theoretical expectation of $\sim 1.333$ derived from the neighbor count ratio ($8/6$). Higher connectivity inherently drives faster gradient washout.

### 6.2 Gravitational Binding Strength
The potential energy (PE) depth of identical two-body configurations was compared.
- **3D PE Minimum:** $-6145.44$
- **4D PE Minimum:** $-18911.18$
- **Ratio (4D/3D):** $3.077$

Gravitational binding is approximately 3 times stronger in 4D for equivalent mass configurations. This compensates for the enhanced diffusion, allowing stable bound structures to form (as verified by F_4D2).

### 6.3 Presence and Drag Landscape
The equilibrium presence $P$ distribution was analyzed.
- **3D Mean $P$:** $0.371$ (Std: $0.149$)
- **4D Mean $P$:** $0.430$ (Std: $0.111$)

The 4D space exhibits a higher mean presence with lower variance. The higher connectivity allows resource and structural drag to smooth out more effectively, creating a more uniform temporal flow landscape.

## 7. Conclusion and Expected Value

The DET v7 architecture generalizes coherently to 4D space. The strict locality, agency-first invariance, and structural debt mechanisms remain numerically stable and physically interpretable.

However, the benchmark results provide a compelling answer to the second core research question: **3D appears to be a uniquely favorable regime for stable local agency.** In 4D, the enhanced diffusion rate ($\sim 1.33\times$) and the significantly smoother presence landscape make it harder to maintain sharp, distinct identities without substantially deeper gravitational wells. 

This 4D simulator provides a robust tool for investigating why our experienced reality is structured in 3 spatial dimensions, demonstrating that lower-dimensional topologies may be required to support the sharp structural persistence necessary for embodied agency.
