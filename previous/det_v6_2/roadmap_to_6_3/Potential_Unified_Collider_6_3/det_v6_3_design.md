# DET v6.3 Unified Collider Suite: Design Specification

## 1. Overview
The DET v6.3 Unified Collider Suite aims to provide a robust, high-performance, and theoretically consistent implementation of the Discrete Emergence Theory (DET) for 1D, 2D, and 3D simulations. This version integrates PyTorch for hardware acceleration, implements full raytracing support for visualization and interaction, and incorporates the latest theoretical advancements in boundary operators and lattice corrections.

## 2. Core Architecture
The suite will be built around a unified `DETCollider` base class with dimension-specific implementations.

### 2.1 PyTorch Integration
- **Tensor-based State:** All state variables (F, q, a, pi, C, sigma, Phi, g) will be stored as PyTorch tensors.
- **Hardware Acceleration:** Support for CPU and CUDA/MPS backends.
- **FFT Solvers:** Use `torch.fft` for solving Helmholtz and Poisson equations.
- **Vectorized Operations:** All updates (flux, presence, momentum, etc.) will be fully vectorized to eliminate Python loops.

### 2.2 Unified Physics (v6.3)
- **Gravity Module:** Helmholtz baseline, Poisson potential, and F-weighted gravitational flux.
- **Lattice Correction:** Automatic calculation of η based on lattice size N to ensure physical consistency of G.
- **Boundary Operators (Grace v6.4):** Antisymmetric edge flux formulation with bond-local quantum gating.
- **Momentum-Gravity Coupling:** Extended momentum update where gravitational fields charge bond momentum.
- **Agency-Gated Dynamics:** All transport and boundary operators are gated by the agency field `a`.

## 3. Raytracing Support
A new `DETRaytracer` module will be implemented to provide:
- **Volumetric Rendering:** Direct rendering of the resource field `F` and potential `Phi`.
- **Path Integration:** Ray-marching through the DET lattice to visualize density and gradients.
- **Interaction:** Support for "virtual rays" to probe the state at specific coordinates.

## 4. Falsifier Suite
The suite will include a comprehensive set of falsifiers to ensure theoretical integrity:
- **F2 (Coercion):** Verify agency-gated operators respect `a=0` nodes.
- **F3 (Redundancy):** Verify boundary operators provide non-redundant recovery.
- **F6 (Binding):** Verify gravitational binding in 3D.
- **F7 (Conservation):** Verify mass conservation across all modules.
- **F_G1-G7 (Grace v6.4):** Specific tests for the antisymmetric grace flux formulation.

## 5. Implementation Plan
1. **`det_v6_3_core.py`:** Base classes and PyTorch utilities.
2. **`det_v6_3_colliders.py`:** 1D, 2D, and 3D implementations.
3. **`det_v6_3_raytracer.py`:** Raytracing and visualization module.
4. **`det_v6_3_falsifiers.py`:** Unified test and validation suite.

## 6. Technical Specifications
- **Lattice Sizes:** Support up to 256³ on modern GPUs.
- **Precision:** Default to `torch.float64` for theoretical validation, `torch.float32` for performance.
- **Boundary Conditions:** Periodic (default) with support for fixed/absorbing boundaries.
