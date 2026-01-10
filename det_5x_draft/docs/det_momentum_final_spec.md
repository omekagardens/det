# DET v5 Momentum Specification (Final)

## Formulation

### State Variable (per-bond)

```
π_{ij} ∈ ℝ,  with  π_{ij} = -π_{ji}
```

### Bond-Local Time Step

```
Δτ_{ij} ≡ ½(Δτ_i + Δτ_j)
```

### Momentum Update

```
\boxed{
π_{ij}^{+} = (1 - λ_π Δτ_{ij}) π_{ij} + α_π J^{(diff)}_{i→j} Δτ_{ij}
}
```

### Momentum-Driven Flow (F-weighted)

```
\boxed{
J^{(mom)}_{i→j} = μ_π σ_{ij} π_{ij} \frac{F_i + F_j}{2}
}
```

### Total Flow

```
J_{i→j} = J^{(diff)} + J^{(grav)} + J^{(mom)} + J^{(floor)}
```

---

## Key Properties

1. **F-weighted flux**: Momentum only pushes stuff that exists
   - In vacuum (F→0), J^{(mom)} → 0
   - Prevents unphysical "free push"

2. **Bond-local time**: Prevents asymmetries between fast/slow clocks

3. **Accumulates from J^{(diff)} only**: 
   - Prevents circular feedback (momentum → flow → more momentum)
   - Acts as "inductance": flow charges memory, memory produces future drift

4. **Friction control**: λ_π tunes ballistic ↔ viscous regime

---

## Parameters

| Parameter | Symbol | Role | Typical Range |
|-----------|--------|------|---------------|
| Accumulation | α_π | How fast flow builds momentum | 0.05 - 0.2 |
| Decay | λ_π | Friction/damping rate | 0.001 - 0.1 |
| Coupling | μ_π | Momentum-flow strength | 0.1 - 0.5 |
| Stability | π_max | Clip bound | 2 - 5 |

---

## Regime Control

| λ_π | Behavior | Physics |
|-----|----------|---------|
| ≪ 0.01 | Ballistic | Persistent momentum, long mean free path |
| ~ 0.01 | Intermediate | Balance of inertia and friction |
| ≫ 0.01 | Viscous | Rapidly damped, overdamped motion |

---

## Test Results

### Collision from Separation

| Momentum | Min Separation | Outcome |
|----------|----------------|---------|
| p=0 | 101.3 | Stationary |
| p=0.5 | **13.8** | Deep collision |

### Friction Scan (p=0.5)

| λ_π | Min Sep | Final |Π|| 
|-----|---------|-------|
| 0.001 | 9.1 | 3.70 |
| 0.01 | 19.8 | 1.08 |
| 0.1 | 75.2 | 0.06 |

---

## Inductance Analogy

```
Electronics:        DET:
──────────────────  ──────────────────
Capacitance         F (resource)
  stores charge       stores "stuff"
  creates voltage     creates gradients

Inductance          π (momentum)  
  stores current      stores "flow memory"
  creates EMF         creates drift
```

The momentum system acts as the "inductive" complement to the "capacitive" resource storage.

---

## Integration with DET v5

### Section II.2 (add to per-bond variables)

```latex
\pi_{ij} \in \mathbb{R} \quad \text{(directed momentum, } \pi_{ij}=-\pi_{ji}\text{)}
```

### New Section IV.4

```
IV.4 Momentum Dynamics (Optional Module)

Per-bond momentum accumulates from diffusive flow and produces F-weighted drift.

Bond-local time:
\[
\Delta\tau_{ij} \equiv \tfrac{1}{2}(\Delta\tau_i + \Delta\tau_j)
\]

Momentum update:
\[
\boxed{
\pi_{ij}^{+} = (1 - \lambda_\pi \Delta\tau_{ij})\,\pi_{ij} 
+ \alpha_\pi\,J^{(\mathrm{diff})}_{i\to j}\,\Delta\tau_{ij}
}
\]

Momentum-driven flow:
\[
\boxed{
J^{(\mathrm{mom})}_{i\to j} = \mu_\pi\,\sigma_{ij}\,\pi_{ij}\,
\Big(\tfrac{F_i+F_j}{2}\Big)
}
\]

Properties:
• Strictly local and pairwise antisymmetric
• F-weighted: vanishes in vacuum
• Not gated by agency
• λ_π controls ballistic (small) vs viscous (large) regime

Parameters α_π, λ_π, μ_π must be reported with results.
```

### Section X (Update Ordering)

Add between Step 5 and Step 6:

```
5a) Update momentum: π_{ij}^+ via IV.4 (if enabled)
```

---

## Falsifiers

### F8 - Momentum Behavior in Vacuum

Isolated momentum (π > 0 in region with F ≈ 0) should not generate flow.
If J^{(mom)} persists with negligible F → falsified.

### F9 - Regime Continuity

Collision dynamics should vary smoothly with λ_π.
If discontinuous transitions occur → investigate.

### F10 - Bound State Formation

With gravity enabled, two bodies with initial momentum toward each other
should be able to form bound orbits (not just scatter).
