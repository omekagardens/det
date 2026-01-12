# DET v6.2 Gravity & Boundary Operator Review

**Date:** January 2026  
**Purpose:** Audit gravity implementations for DET consistency; implement boundary operators for F2/F3

---

## Part 1: Gravity Implementation Audit

### 1.1 Theory Card Requirements (Section V)

DET gravity is **emergent**, not an intrinsic force. Key equations:

```
Gravity source (relative structure):
    ρ_i = q_i - b_i

Baseline field (screened Poisson):
    (L_σ b)_i - α b_i = -α q_i

Gravitational potential:
    (L_σ Φ)_i = -κ ρ_i

Gravitational flux:
    J^(grav)_{i→j} = μ_g · σ_{ij} · (F_i + F_j)/2 · (Φ_i - Φ_j)
```

**Critical Readout Discipline:**
- Φ is a computational intermediary, NOT a direct input to clock rates
- Time dilation comes from F-redistribution via J^(grav)
- The correct DET relation is `P/P_∞ = (1+F_∞)/(1+F)`, NOT `P/P_∞ = 1+Φ`

### 1.2 Current Implementation Analysis

#### 1D Collider (det_v6_1d_collider_gravity.py)

**Helmholtz Baseline:** ✓ CORRECT
```python
def _solve_helmholtz(self, source: np.ndarray) -> np.ndarray:
    source_k = fft(source)
    b_k = -self.p.alpha_grav * source_k / self.H_k
    return np.real(ifft(b_k))
```

**Poisson Potential:** ✓ CORRECT
```python
def _solve_poisson(self, source: np.ndarray) -> np.ndarray:
    source_k = fft(source)
    source_k[0] = 0  # Compatibility condition
    Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
    Phi_k[0] = 0
    return np.real(ifft(Phi_k))
```

**Gravity Computation Flow:** ✓ CORRECT
```python
def _compute_gravity(self):
    self.b = self._solve_helmholtz(self.q)      # Step 1
    rho = self.q - self.b                        # Step 2
    self.Phi = self._solve_poisson(rho)          # Step 3
    self.g = -0.5 * (R(self.Phi) - L(self.Phi))  # Step 4
```

**Gravitational Flux:** ✓ CORRECT (equivalent form)
```python
J_grav_R = p.mu_grav * self.sigma * g_bond_R * F_avg_R
```
This is equivalent to theory card since `g = -∇Φ ≈ (Φ_i - Φ_j)/Δx`.

#### Issues Found

**Issue 1: Undocumented Gravity-Momentum Coupling**
```python
# In momentum update:
if p.gravity_enabled:
    dpi_grav = 5.0 * p.mu_grav * g_bond_R * Delta_tau_R  # ← Magic 5.0
```
The factor `5.0` is undocumented. This adds gravitational acceleration to momentum charging, which is physically reasonable but not in the theory card.

**Recommendation:** Document this as an implementation choice or add to theory card as optional momentum-gravity coupling:
```
π^+_{ij} = (1 - λ_π Δτ)π_{ij} + α_π J^{diff} Δτ + β_g · g_{ij} Δτ
```

**Issue 2: Comments Reference "Newtonian" Behavior**
```python
# For attractive gravity (Newtonian):
# We want Φ < 0 near mass (potential well)
```
This is fine for calibration context, but should clarify DET's emergent nature.

**Issue 3: Presence Formula Correct**
```python
self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
```
This correctly does NOT include Φ directly. Time dilation emerges from F-redistribution. ✓

### 1.3 Gravity Audit Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Helmholtz baseline | ✓ CORRECT | All dimensions |
| Relative source ρ = q - b | ✓ CORRECT | All dimensions |
| Poisson potential | ✓ CORRECT | All dimensions |
| Gravitational flux | ✓ CORRECT | Equivalent form |
| F-weighting | ✓ CORRECT | Uses F_avg |
| Readout discipline | ✓ CORRECT | P doesn't use Φ directly |
| Momentum-gravity coupling | ⚠ UNDOCUMENTED | Factor 5.0 arbitrary |

**Overall: Gravity is DET-consistent with minor documentation issues.**

---

## Part 2: Boundary Operator Audit

### 2.1 Theory Card Requirements (Section VI)

**Agency Inviolability (VI.1):**
```
Boundary operators cannot directly modify a_i
```

**Grace Injection (VI.5):**
```
Need:     n_i = max(0, F_min - F_i)
Weight:   w_i = a_i · n_i
Injection: I_{g→i} = D_i · w_i / (Σ_{k∈N_R(i)} w_k + ε)
```
where `D_i = Σ_j |J_{i→j}| Δτ_i` is local dissipation.

**Bond Healing (implied by F2):**
The theory card mentions "bond healing" in F2 but doesn't fully specify the operator. From VI.3:
```
Coherence dynamics include detector-driven decoherence but no explicit healing operator.
```

### 2.2 Current Implementation Status

**Grace Injection:** ❌ NOT IMPLEMENTED

Looking at the collider step() functions:
```python
# Resource update (IV.7)
dF = inflow - outflow
self.F = np.clip(self.F + dF, p.F_MIN, 1000)  # ← No I_{g→i} term!
```

The grace injection `I_{g→i}` is missing from all colliders.

**Bond Healing:** ❌ NOT IMPLEMENTED

The coherence dynamics in VI.3 only include:
- Flow-driven coherence building: `+α_C |J| Δτ`
- Natural decay: `-λ_C C Δτ`
- Detector-driven decoherence: `-λ_M m g √C Δτ`

No explicit boundary healing operator.

### 2.3 Implications for Falsifiers

**F2 (Coercion):** Currently trivially passes because no boundary action exists!

**F3 (Boundary Redundancy):** Currently FAILS because boundary-on and boundary-off are identical!

---

## Part 3: Implementation Plan

### 3.1 Grace Injection Implementation

```python
def compute_grace_injection(self, D: np.ndarray) -> np.ndarray:
    """
    Grace injection per DET VI.5.
    
    I_{g→i} = D_i · w_i / (Σ w_k + ε)
    
    where:
    - n_i = max(0, F_min - F_i)  # Need
    - w_i = a_i · n_i            # Weight (agency-gated)
    - D_i = local dissipation
    
    Returns injection amount per node.
    """
    # Need: how far below F_min
    n = np.maximum(0, self.p.F_MIN - self.F)
    
    # Weight: agency-gated need
    w = self.a * n
    
    # Local normalization (within neighborhood)
    # For lattice implementation, use local sum
    w_sum = periodic_local_sum(w, self.p.R) + 1e-12
    
    # Injection
    I_g = D * w / w_sum
    
    return I_g
```

### 3.2 Bond Healing Implementation (Optional)

```python
def compute_bond_healing(self, D: np.ndarray) -> np.ndarray:
    """
    Optional bond healing operator, agency-gated.
    
    ΔC^{heal}_{ij} = η_h · g^{(a)}_{ij} · (1 - C_{ij}) · D̄_{ij} · Δτ_{ij}
    
    where g^{(a)}_{ij} = sqrt(a_i · a_j) is the agency gate.
    """
    # Only heals if BOTH endpoints are open
    g_E = np.sqrt(self.a * E(self.a))
    g_S = np.sqrt(self.a * S(self.a))
    
    # Healing proportional to (1-C) and local dissipation
    D_avg_E = 0.5 * (D + E(D))
    D_avg_S = 0.5 * (D + S(D))
    
    dC_heal_E = self.p.eta_heal * g_E * (1 - self.C_E) * D_avg_E * self.Delta_tau
    dC_heal_S = self.p.eta_heal * g_S * (1 - self.C_S) * D_avg_S * self.Delta_tau
    
    return dC_heal_E, dC_heal_S
```

### 3.3 F2 Test Implementation

From the user's document:

**Test A: Hard-zero agency sentinel (grace injection)**
- Create sentinel node with `a_s = 0` and `F_s << F_min`
- Surround with open, needy neighbors
- Verify `I_{g→s} = 0` exactly

**Test B: Bond-heal coercion**
- Pick bond (i,j) with `a_i = 0, a_j = 1`, low coherence
- Create conditions that would trigger healing
- Verify `ΔC^{heal}_{ij} = 0`

**Test C: Agency-flip invariance**
- Same setup, compare `a_s = 0` vs `a_s = ε`
- At `a_s = 0`: zero boundary help
- At `a_s > 0`: positive share

### 3.4 F3 Test Implementation

**Test D: Scarcity collapse vs recovery**
- Create harsh world where dissipation drains F
- Boundary OFF: system collapses
- Boundary ON: system recovers

**Test E: Local crisis, local response**
- Two disconnected components
- Apply crisis only to component A
- Verify only A shows boundary response

**Test F: A/B toggle mid-run**
- Single deterministic run
- Toggle boundary at t* during stress
- Observe phase change in trajectories

---

## Part 4: Recommended Code Changes

### 4.1 Parameters to Add

```python
@dataclass
class DETParams:
    # ... existing params ...
    
    # Boundary operators (VI.5)
    boundary_enabled: bool = True
    F_MIN_grace: float = 0.05      # F threshold for grace injection
    eta_heal: float = 0.05         # Bond healing rate (if implemented)
```

### 4.2 Step Function Modifications

```python
def step(self):
    # ... existing flow computation ...
    
    # Compute dissipation (needed for grace)
    D = (np.abs(J_E_lim) + np.abs(J_W_lim) + 
         np.abs(J_S_lim) + np.abs(J_N_lim)) * self.Delta_tau
    
    # ... existing F update ...
    
    # Grace injection (VI.5)
    if self.p.boundary_enabled:
        I_g = self.compute_grace_injection(D)
        self.F = self.F + I_g
    
    # Bond healing (optional)
    if self.p.boundary_enabled and self.p.eta_heal > 0:
        dC_E, dC_S = self.compute_bond_healing(D)
        self.C_E = np.clip(self.C_E + dC_E, self.p.C_init, 1.0)
        self.C_S = np.clip(self.C_S + dC_S, self.p.C_init, 1.0)
```

---

## Next Steps

1. Implement grace injection in 1D collider (simplest test case)
2. Write F2 test harness
3. Write F3 test harness
4. Verify both pass
5. Port to 2D and 3D colliders
6. Document gravity-momentum coupling in theory card
