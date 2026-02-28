# DET v6.3 Parameter Metrology

## Parameter Classification

DET parameters fall into three buckets with distinct roles:

### Bucket A: Unit/Scale Parameters
Define units or baseline scales. Can be set arbitrarily to establish measurement conventions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| DT | 0.02 | Time step (defines time unit) |
| N | 32-64 | Grid size |
| R | 2 | Neighborhood radius |
| F_VAC | 0.01 | Vacuum resource level |
| F_MIN | 0.0 | Minimum resource |
| pi_max | 3.0 | Maximum momentum |
| L_max | 5.0 | Maximum angular momentum |

### Bucket B: Physical Law Parameters
Control physical couplings. Must be measured from experiments or fitted to observations.

| Parameter | Default | Description | Measurement Rig |
|-----------|---------|-------------|-----------------|
| alpha_pi | 0.12 | Momentum charging gain | Pulse injection |
| lambda_pi | 0.008 | Momentum decay rate | Decay envelope |
| mu_pi | 0.35 | Momentum mobility | Transport rate |
| alpha_L | 0.06 | Angular momentum charging | Rotational pulse |
| lambda_L | 0.005 | Angular momentum decay | Spin-down rate |
| mu_L | 0.18 | Rotational mobility | Circulation rate |
| eta_f | 0.12 | Floor stiffness | Compression test |
| F_core | 5.0 | Onset density | Threshold fit |
| kappa | 5.0 | Poisson coupling | Potential profile |
| mu_g | 2.0 | Gravity mobility | Drift velocity |
| **beta_g** | 10.0 | Gravity-momentum coupling (v6.3) | Infall rate |
| alpha_C | 0.04 | Coherence growth rate | Steady flow |
| lambda_C | 0.002 | Coherence decay rate | Decay envelope |
| alpha_q | 0.012 | Q-locking rate | Structure buildup |
| a_coupling | 30.0 | Agency coupling | Agency response |
| a_rate | 0.2 | Agency rate | Timescale |

### Bucket C: Numerical Parameters
Control numerical stability. Should not affect physical predictions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| outflow_limit | 0.2 | Maximum outflow fraction |
| R_boundary | 2 | Boundary radius |
| toggles | - | Feature enables/disables |

## Measurement Rigs

### Gravity Constants (kappa, mu_g, alpha_grav, beta_g)

**Setup:**
1. Place compact source with known q at grid center
2. Measure potential profile Phi(r) radially
3. Track drift velocities of test cloud

**Extraction:**
- kappa from potential amplitude
- mu_g from drift mobility
- beta_g from momentum buildup during infall

### Momentum Constants (alpha_pi, lambda_pi, mu_pi)

**Setup:**
1. Inject brief diffusive flux pulse
2. Measure momentum rise and decay

**Extraction:**
- alpha_pi from initial rise rate
- lambda_pi from exponential decay
- mu_pi from transport velocity

### Angular Momentum Constants (alpha_L, lambda_L, mu_L)

**Setup:**
1. Excite rotational pulse around plaquette
2. Measure L rise and circulation

**Extraction:**
- alpha_L from curl charging rate
- lambda_L from spin-down rate
- mu_L from circulation velocity

### Floor Parameters (eta_f, F_core, p)

**Setup:**
1. Quasi-static compression: squeeze two bodies
2. Controlled collisions: measure rebound

**Extraction:**
- F_core from onset threshold
- eta_f from stiffness
- p from nonlinearity

### Coherence Dynamics (alpha_C, lambda_C)

**Setup:**
1. Steady flow through bridge
2. Measure C growth and decay

**Extraction:**
- alpha_C from saturation level
- lambda_C from decay after flow stops

## Lattice Correction Factor (v6.3)

The discrete Laplacian creates systematic corrections:

### Origin
Discrete eigenvalues: lambda(k) = -4 * sum(sin^2(k_i/2))
Continuum eigenvalues: lambda(k) = -k^2

### Correction Values

| N | eta |
|---|-----|
| 32 | 0.901 |
| 64 | 0.955 |
| 96 | 0.968 |
| 128 | 0.975 |

### Physical G Extraction

```
G_physical = (1/eta) * kappa / (4*pi)
```

This ensures the extracted G matches continuum physics regardless of lattice resolution.

## Recommended Experimental Protocol

### Phase 1: Unit Calibration
1. Set DT, N, F_VAC to convenient values
2. Establish length/time scales

### Phase 2: Internal Consistency
1. Run falsifier suite (all 15 should pass)
2. Verify conservation laws

### Phase 3: External Calibration
1. Two-body problem: extract effective G
2. Compare with Newtonian prediction
3. Fit galaxy rotation curves

### Phase 4: Predictive Science
1. Cosmological scaling
2. Black hole thermodynamics
3. Quantum-classical transition
