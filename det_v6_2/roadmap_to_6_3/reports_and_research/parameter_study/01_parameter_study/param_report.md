# Tunable Parameters in DET v6.2 – classification, function, and measurement

**Context.** Deep Existence Theory (DET) v6.2 is a discrete, strictly local theory where resource **F**, agency **a**, coherence **C**, structural debt **q**, bond conductivity **σ**, phase **θ**, momentum **π**, angular momentum **L** and other variables evolve according to well‑defined local rules.  The theory card and code in `det_v6_2` provide a collection of configurable constants that tune time scales, couplings and numeric stability.  To give DET predictive power, these constants should not be arbitrarily tuned for each scenario.  Below we perform a deep search through the v6.2 folder to list all tunable parameters, classify them, describe their role in the theory, and propose measurement protocols.

## 1 Classification framework

The user’s guidelines group parameters into three buckets:

| bucket | definition | examples | notes |
|---|---|---|---|
| **A – Unit/scale setters** | Define units of measurement or fix a baseline scale (e.g., time step, grid size, baseline resource).  These can be chosen by convention once and should remain fixed. | time step `DT`, grid size `N`, neighbourhood radius `R` or `R_boundary`, baseline resource levels `F_VAC`, `F_MIN`, maximum momentum `π_max` and angular‑momentum `L_max`, phase frequency `ω_0` and damping `γ_0`. | Changing these rescales the simulation’s rulers or clocks.  They do not alter the physical content once fixed. |
| **B – Physical‑law constants** | Appear in the equations of motion and control physical couplings and time scales.  They must be measured from reproducible experiments and transferred across scenarios without retuning. | momentum module constants `α_π`, `λ_π`, `μ_π` (charging gain, decay rate, mobility) defined in the momentum update and drift flux【398607633750785†L213-L223】; angular‑momentum constants `α_L`, `λ_L`, `μ_L` in the rotational dynamics【398607633750785†L235-L244】; floor‑repulsion parameters `η_f`, `F_core`, exponent `p` in the finite‑compressibility law【398607633750785†L253-L263】; gravity constants `κ` and `μ_g` in the Poisson equation and gravitational flux【398607633750785†L307-L319】; coherence dynamics `α_C`, `λ_C` and detector coupling `λ_M`【398607633750785†L355-L363】; q‑locking `α_q` and any thresholds; agency coupling `λ_a` and rate `β` in the target‑tracking rule【398607633750785†L346-L347】; pointer‑record parameters `α_r`, `η_r`【398607633750785†L374-L384】; healing rate `η_heal`; gravity‐momentum coupling β\_g (implemented in the unified colliders). | Each of these must be tied to a direct observable via a **standard measurement rig**.  Their values should be fitted once on a calibration lattice and re‑used. |
| **C – Numerical/discretization parameters** | Control numerical stability or discretization but should not influence physical predictions when the simulation is converged. | outflow limit (max fraction of resource allowed to leave a node), diffusive neighbourhood radius `R`, boundary normalisation radius `R_boundary`, toggles such as `diff_enabled`, `momentum_enabled`, `angular_momentum_enabled`, `boundary_enabled`, `grace_enabled`, `healing_enabled`. | Predictions must be invariant under reduction of `Δt` and `Δx` (refined grid), larger domains, and different boundary types.  If results change when these are varied, the physics is not yet converged. |


## 2 List of tunable parameters and their role

The table below summarises every parameter identified in the **`det_v6_2`** folder.  A short description of each parameter’s function is provided alongside its classification.  Parameters labelled **(B)** require measurement; the suggested experimental rig is given in section 3.  Parameters labelled **(A)** are unit/scale setters; they should be fixed by convention.  Parameters labelled **(C)** are purely numerical and should not affect final claims.  Where a parameter appears only in specific modules (e.g., unified colliders), this is noted.

| parameter | classification | role/function (summary) | relevant equation or file |
|---|---|---|---|
| **`DT`** | A | Time‑step in lattice units (sets the clock). | All collider classes define `DT` as the physical time increment per simulation step; once chosen, never retune. |
| **`N`** | A | Number of lattice nodes per dimension (domain size). | Sets spatial resolution; influences memory/time; part of discretisation. |
| **`R`** | C | Local neighbourhood radius used in diffusive flux and resource normalisation. | Appears in definitions of `∑` and `⟨·⟩` in the theory card【398607633750785†L18-L23】; influences locality but should converge. |
| **`F_VAC`** | A | Baseline (vacuum) resource level; ensures nonzero resource in absence of matter. | Provided as default in colliders; sets baseline `F` scale. |
| **`F_MIN`** | A/C | Minimum allowed resource; prevents negative `F`.  The boundary module injects grace when `F_i` falls below `F_MIN`. | Used in boundary rules【398607633750785†L399-L407】. |
| **`π_max`** | A/C | Maximum magnitude of bond momentum; prevents numerical blow‑up. | Parameter in colliders; saturates `π_{ij}` before applying flux. |
| **`L_max`** | A/C | Maximum magnitude of plaquette angular momentum; prevents blow‑up. | Parameter in 2D/3D colliders. |
| **`ω_0`, `γ_0`** | A | Intrinsic frequency and damping for phase dynamics. | Phase update rule【398607633750785†L391-L392】; used in optional phase module. |
| **`R_boundary`** | C | Radius for boundary normalisation when computing grace injection and bond healing. | Introduced in boundary operator files to enforce locality; influences numeric stability. |
| **`outflow_limit`** | C | Maximum fraction of resource that may leave a node in one time step; ensures stability and prevents overshoot. | Implementation detail in collider update functions. |
| **`α_π` (alpha_pi)** | **B** | **Momentum charging gain**.  In the momentum update, diffusive flux charges the bond momentum `π` with rate `α_π`【398607633750785†L213-L215】.  Larger `α_π` means diffusive pulses create larger momentum memory. | Momentum module (theory card IV.4) and `det_v6_2d_collider.py`. |
| **`λ_π` (lambda_pi)** | **B** | **Momentum decay time constant**.  The factor `(1 − λ_π Δτ)` in the update law causes `π` to decay exponentially with time【398607633750785†L213-L215】. | Momentum update【398607633750785†L213-L215】. |
| **`μ_π` (mu_pi)** | **B** | **Momentum mobility constant**.  Controls how strongly stored momentum drives directed resource transport in `J^(mom)`【398607633750785†L221-L223】.  Higher values produce longer travel distances before momentum dissipates. | Momentum-driven flux【398607633750785†L221-L223】. |
| **`α_L` (alpha_L)** | **B** | **Angular‑momentum charging gain**.  Generated from the curl of `π` around each plaquette; sets how rapidly angular momentum accumulates【398607633750785†L235-L244】. | Rotational dynamics【398607633750785†L235-L244】. |
| **`λ_L` (lambda_L)** | **B** | **Angular‑momentum decay rate**.  Factor `(1 − λ_L Δτ)` causes `L` to decay toward zero【398607633750785†L235-L244】. | Rotational update【398607633750785†L235-L244】. |
| **`μ_L` (mu_L)** | **B** | **Rotational mobility**.  Coupling from stored `L` to rotational (divergence‑free) flux `J^(rot)`【398607633750785†L242-L244】. | Angular‑momentum flux【398607633750785†L242-L244】. |
| **`η_f` (eta_f)** | **B** | **Floor stiffness strength**.  Multiplies the floor repulsion flux; controls repulsive pressure when density exceeds `F_core`【398607633750785†L259-L263】. | Floor repulsion law【398607633750785†L253-L263】. |
| **`F_core`** | **B** | **Onset density for floor repulsion**.  The floor strength function uses $(F_i − F_{core})/F_{core}$ raised to an exponent to compute repulsion【398607633750785†L253-L263】. | Floor repulsion law. |
| **`p` (floor_power)** | **B** | **Exponent of non‑linearity in floor repulsion**.  Controls how sharply pressure grows with compression【398607633750785†L253-L263】. | Floor repulsion law. |
| **`α_grav` (α)** | **B** | **Screening parameter** for the baseline field `b_i` in the gravity module.  Appears in the screened Poisson equation `(L_σ b)_i − α b_i = −α q_i`【398607633750785†L297-L299】; sets the range of the baseline filter.  In code this may be fixed or set equal to `κ`. | Gravity baseline equation【398607633750785†L297-L299】. |
| **`κ` (kappa_grav)** | **B** | **Poisson coupling constant**.  Relates structural debt `ρ_i = q_i − b_i` to gravitational potential via `(L_σ Φ)_i = −κ ρ_i`【398607633750785†L307-L319】. | Gravity module【398607633750785†L307-L319】. |
| **`μ_g` (mu_grav)** | **B** | **Gravity‑driven mobility**.  Scales the gravitational drift flux `J^(grav)` via `μ_g σ (F_i+F_j)/2 (Φ_i − Φ_j)`【398607633750785†L315-L319】.  Larger `μ_g` yields faster free fall. | Gravity flux【398607633750785†L315-L319】. |
| **`β_g`** | **B** | **Gravity–momentum coupling coefficient**.  Not explicitly defined in the theory card but implemented in unified colliders as `β_g ≈ 5 μ_g`; couples gravitational potential differences to momentum update, analogous to gravitational redshift of momentum. | Implementation in unified colliders. |
| **`α_C` (alpha_C)** | **B** | **Coherence growth rate**.  Increases bond coherence `C_{ij}` in proportion to the absolute flow |J_{i→j}|【398607633750785†L355-L363】. | Coherence dynamics【398607633750785†L355-L363】. |
| **`λ_C` (lambda_C)** | **B** | **Coherence decay rate**.  Causes `C_{ij}` to decay toward zero【398607633750785†L355-L363】. | Coherence dynamics. |
| **`λ_M` (lambda_M)** | **B** | **Detector coupling**.  Additional coherence decay term `−λ_M m_{ij} g^{(a)}_{ij} √C_{ij}` models detector-induced decoherence【398607633750785†L355-L363】. | Coherence dynamics (optional). |
| **`α_q` (alpha_q)** | **B** | **Structural‑debt (q-lock) formation rate**.  Controls how quickly structural debt builds from past interactions; used in unified colliders and structural module.  Higher values mean faster memory accumulation. | Q‑locking law (Appendix B in code; not explicit in the card). |
| **`α_r` (alpha_r)** | **B** | **Pointer‑record charging gain**.  Determines how quickly the record `r_i` accumulates from local dissipation in the presence of a detector【398607633750785†L374-L377】. | Pointer record update【398607633750785†L374-L377】. |
| **`η_r` (eta_r)** | **B** | **Pointer‑record reinforcement strength**.  Amplifies bond conductivity via `σ_eff,ij = σ_{ij}(1 + η_r \bar{r}_{ij}/(1 + \bar{r}_{ij}))`【398607633750785†L381-L384】.  Large `η_r` yields strong memory of measurement outcomes. | Pointer record reinforcement【398607633750785†L381-L384】. |
| **`λ_a` (a_coupling)** | **B** | **Agency coupling**.  In the target‑tracking agency rule, the desired agency target decreases with structural debt via `a_target = 1/(1+λ_a q_i^2)`【398607633750785†L346-L347】.  Larger `λ_a` means structural debt suppresses agency more strongly. | Agency update rule【398607633750785†L346-L347】. |
| **`β` (a_rate)** | **B** | **Agency rate**.  Controls how quickly actual agency `a_i` relaxes toward `a_target` in each time step【398607633750785†L346-L347】. | Agency update rule. |
| **`η_heal`** | **B** | **Bond‑healing rate** used in boundary operators.  Determines how quickly coherence or conductivity on broken bonds is restored during healing. | Boundary operator files. |
| **`F_MIN_grace`** | A/B | **Grace‑injection threshold**.  Specifies the resource level below which boundary agents inject grace into a node.  Since it determines when boundary action occurs, it behaves like a physical threshold rather than a numeric artifact.  It should be chosen once and fixed. | Grace injection rule【398607633750785†L399-L407】. |
| **`healing_enabled`, `grace_enabled`, `boundary_enabled`, `momentum_enabled`, `angular_momentum_enabled`, `diff_enabled`, `phase_enabled`** | C (toggles) | Boolean switches enabling/disabling modules.  They change the physical model, not the parameter values, and should be kept consistent across experiments. | Implementation in collider files. |

### Notes on parameter roles

* **Momentum and angular momentum modules**: The parameters `α_π`, `λ_π`, `μ_π`, `α_L`, `λ_L`, `μ_L` control how past diffusive flows are stored as momentum and how momentum generates directed or rotational resource transport【398607633750785†L213-L223】【398607633750785†L235-L244】.  They determine collision behaviour (sticking vs rebound) and orbital binding.
* **Floor repulsion**: The combination of `η_f`, `F_core` and exponent `p` sets the equation of state of the medium.  At densities above `F_core`, the repulsive flux grows as `[ (F−F_core)/F_core ]_+^p`【398607633750785†L253-L263】, preventing unphysical compression.  These parameters mimic material stiffness.
* **Gravity constants**: `κ` and `μ_g` define an emergent gravitational potential from structural debt and the drift mobility of resource along potential gradients【398607633750785†L307-L319】.  `α_grav` governs the screening of the baseline field.  The unified colliders also include a hard‑coded `β_g` coupling between gravitational potential and momentum update.
* **Coherence and pointer dynamics**: `α_C`, `λ_C` and `λ_M` control how coherence grows with flow and decays spontaneously or due to measurement【398607633750785†L355-L363】.  Pointer record parameters `α_r` and `η_r` stabilise measurement outcomes by reinforcing conductivity【398607633750785†L374-L384】.
* **Agency and q‑locking**: `λ_a` and `β` set how structural debt suppresses agency and how quickly the suppression is realised【398607633750785†L346-L347】.  `α_q` governs the rate at which structural debt accumulates from past events.
* **Boundary and healing constants**: `η_heal` sets the rate of bond healing in boundary modules, while `F_MIN_grace` determines when grace is injected.  Their values should be fixed by convention or measured from the physical process being modelled.


## 3 Measurement rigs for bucket‑B parameters

The DET theory emphasises that physical‑law constants (bucket B) must be *measurable and transferable*.  Below are standard experimental rigs—adaptable to either numerical simulation or physical analogues—that define each constant’s empirical meaning.  The real‑world analogues use small mechanical, electrical or fluid networks to emulate the lattice dynamics.

### 3.1 Gravity constants (κ, μ_g, α_grav)

**Sim‑native measurement.**  Place a compact source region with known structural debt `ρ` by setting a cluster of nodes with `q_i > 0`.  Solve for the gravitational potential `Φ` via the local Poisson equation `(L_σ Φ)_i = −κ ρ_i`【398607633750785†L307-L319】 and allow only gravitational flux `J^(grav)` to be active.  Surround the source with a dilute test cloud of low‑resource nodes and track their radial drift velocities.  Two observables determine the constants:

1. **Potential profile:** Fit the radial dependence of `Φ(r)` to the discrete Green’s function.  The amplitude of `Φ` scales with `κ`, so `κ` is determined by matching the predicted potential to the measured `Φ` at multiple radii.
2. **Drift mobility:** For the test cloud, measure radial velocity versus local potential gradient.  From `v_r ≈ μ_g ∇Φ`【398607633750785†L315-L319】 the slope yields `μ_g`.  The screening parameter `α_grav` can be fitted by measuring the baseline field `b_i` around the source and solving `(L_σ b)_i − α b_i = −α q_i`【398607633750785†L297-L299】.

**Real‑world analogue.**  Construct an electrical‑network analogue using resistors as conduits (σ), capacitors as resource storage (F) and memristive elements to emulate structural debt.  Drive a current source at one node to emulate `ρ`.  The voltage field corresponds to `Φ`; the admittance network ensures local interactions.  Measure the voltage profile and test current drift to extract `κ` and `μ_g`.  Alternatively, build a shallow water or granular fluid channel where height differences create gravitational analogues; measure the drift of floating test particles.

### 3.2 Momentum module constants (α_π, λ_π, μ_π)

**Sim‑native measurement.**  Prepare a uniform lattice with low resource and enable diffusive flux only.  Inject a brief flux pulse through a narrow channel for a fixed number of time steps, then stop.  Track the bond momentum `π` and the resulting momentum‑driven drift `J^(mom)`:

* **Charging gain (`α_π`):** The rise of `π` during the pulse is proportional to `α_π`【398607633750785†L213-L223】.  Measure the peak momentum per unit integrated diffusive flux to determine `α_π`.
* **Decay rate (`λ_π`):** After the pulse, momentum decays exponentially as `π(t) ≈ π_0 e^{−λ_π τ}`【398607633750785†L213-L215】.  Fit an exponential to the decay curve to extract `λ_π`.
* **Mobility (`μ_π`):** When `π` is non‑zero, observe the additional transport of resource over distance.  The drift flux is `J^(mom) = μ_π σ π (F_i+F_j)/2`【398607633750785†L221-L223】.  Measuring the resulting displacement per unit momentum yields `μ_π`.

**Real‑world analogue.**  Build a mechanical model using carts on a low‑friction track linked by springs (σ).  Apply a brief push (diffusive pulse) to one cart and observe how momentum is stored and transferred.  Alternatively, create a microfluidic device where a transient pressure gradient charges momentum in a channel; monitor flow after the gradient is removed.

### 3.3 Angular‑momentum constants (α_L, λ_L, μ_L)

**Sim‑native measurement.**  In 2D or 3D colliders with the angular‑momentum module enabled, excite a localized rotational pulse: create a ring of diffusive flux around a plaquette for a short time.  Measure the resulting plaquette momentum `L` and its effect on resource transport:

* **Charging gain (`α_L`):** Angular momentum is generated by the discrete curl of `π`【398607633750785†L235-L244】.  The rise of `L` per unit curl gives `α_L`.
* **Decay rate (`λ_L`):** After the pulse, `L` decays exponentially `(1 − λ_L Δτ)`【398607633750785†L235-L244】; fit the decay to extract `λ_L`.
* **Mobility (`μ_L`):** Measure the rotational flux `J_rot` proportional to `μ_L σ F_avg ∇^⊥ L`【398607633750785†L242-L244】.  By observing the circulation induced around the plaquette, compute `μ_L`.

**Real‑world analogue.**  Use a shallow water tank or rotating table where local vortex injection can be controlled (e.g., by stirring a small region).  Measure how the induced vortex decays and transports fluid sideways; fit the same constants.

### 3.4 Floor/stiffness parameters (η_f, F_core, p)

**Sim‑native measurement.**  Treat the medium like a compressible material.  Two complementary assays measure the equation of state:

1. **Quasi‑static compression:** Slowly squeeze a blob of high‐resource nodes between two boundaries while measuring the flux needed to maintain compression.  Determine the onset density where repulsion activates (`F_core`), the exponent `p` by fitting how pressure grows with compression, and the overall strength `η_f`【398607633750785†L253-L263】.
2. **Controlled collisions:** Collide two high‑resource blobs at known velocities.  Observe maximum compression and rebound coefficient.  Fit the same parameters by matching simulation results to the observed compression curve.

**Real‑world analogue.**  Use a granular material or soft foam as a proxy for resource density.  Compress it slowly and record force–displacement curves to determine onset density and stiffness exponent.  Alternatively, collide two balls of modeling clay; measure rebound and deformation.

### 3.5 Coherence dynamics (α_C, λ_C, λ_M)

**Sim‑native measurement.**  Connect two regions by a bridge (a narrow set of bonds).  Impose a steady flow of resource through the bridge for a fixed duration, then stop.  Monitor the growth of bond coherence `C_{ij}` during the flow and its relaxation after the flow stops.

* **Growth rate (`α_C`):** The rate at which coherence increases with flux is proportional to `α_C`【398607633750785†L355-L363】.  Measure the initial slope of `C` versus integrated flow.
* **Decay rate (`λ_C`):** After the flow stops, coherence decays exponentially; fit `λ_C` from the decay time.【398607633750785†L355-L363】
* **Detector coupling (`λ_M`):** If detectors (measurement devices) are present, coherence decay is accelerated by a term `−λ_M m_{ij} g^{(a)}_{ij} √C`【398607633750785†L355-L363】.  Vary detector coupling and measure the change in decay rate to determine `λ_M`.

**Real‑world analogue.**  Implement an optical interferometer or electrical resonator where coherence (interference visibility or phase correlation) can be monitored.  Driving a steady current or light flux through a coupling region and then turning it off mimics the simulation.  Measure coherence decay under different noise levels (analogous to detector coupling).

### 3.6 Structural‑debt (q‑locking) parameters (α_q and thresholds)

**Sim‑native measurement.**  Apply cyclic stress to a region: alternate between high diffusive flow and rest.  Record how structural debt `q` accumulates under stress and relaxes during rest.  The rate of accumulation gives `α_q`.  Plot the hysteresis curve (q vs stress) to identify threshold behaviour.  Determine critical transitions (where debt locks in) to classify regimes.  A family of q‑locking laws can be falsified if the fitted parameters fail to predict memory in other scenarios.

**Real‑world analogue.**  Use a physical system with memory (e.g., a plastic material that accumulates permanent deformation when cyclically loaded).  Apply cyclic loads and measure permanent strain vs load amplitude.  Fit `α_q` and thresholds to match the hysteresis area and transition points.

### 3.7 Agency parameters (λ_a, β) and pointer records (α_r, η_r)

**Agency measurement.**  Prepare a region with structural debt `q_i` and measure how agency `a_i` adapts.  According to the target‑tracking rule【398607633750785†L346-L347】, the target agency is `a_target = 1/(1+λ_a q_i^2)`, and actual agency relaxes toward it at rate `β`.  Vary `q_i` in controlled experiments (e.g., by injecting structural debt) and fit the relationship between `a_i` and `q_i` to determine `λ_a`.  Measure the time constant of adaptation to obtain `β`.

**Pointer‑record measurement.**  Activate a detector in a region and monitor the record variable `r_i`.  The record accumulates as `r_i^{+} = r_i + α_r m_i D_i Δτ`【398607633750785†L374-L377】, where `D_i` is local dissipation.  Measure how `r` grows with detector strength and flux to determine `α_r`.  Then examine how increased records modify effective conductivity through `σ_eff,ij = σ_{ij}(1 + η_r \bar{r}_{ij}/(1+ \bar{r}_{ij}))`【398607633750785†L381-L384】; fit `η_r` from conductivity versus average record.

**Real‑world analogues.**  For agency, one could design a robot or agent in a networked environment whose responsiveness (`a`) decreases with accumulated load (`q`).  By measuring reaction speed under increasing load, one can fit `λ_a` and `β`.  For pointer records, use an adaptive synapse in neuromorphic hardware; the synaptic weight increases with activity (`α_r`) and saturates with a nonlinear function (`η_r`).

### 3.8 Boundary and healing constants (η_heal, F_MIN_grace)

**Grace injection threshold (`F_MIN_grace`).**  Determine the minimum resource that boundary agents should inject by examining when a region becomes operationally inert.  In simulation, slowly reduce `F_i` and monitor when diffusive flux ceases; choose `F_MIN_grace` just above this threshold.  In physical analogues, identify the minimum energy or mass per agent necessary to sustain dynamics.

**Bond‑healing rate (`η_heal`).**  In boundary modules, bond healing restores connectivity after damage.  Set up a broken bond and measure how quickly conductivity returns.  Fit the rate constant `η_heal` by comparing the time to heal to the total potential range.  In real systems, this could correspond to the healing of chemical bonds or reconnection of neural axons; measure the time constant of recovery.


## 4 Promoting constants to local constitutive functions

While the above parameters are treated as global constants, DET notes that many may be *derived* from local state rather than externally tuned.  For example, instead of a single `μ_g`, one can define a local gravity mobility `μ_g(i) = f(a_i, C_neighbourhood, σ_neighbourhood)`, making gravitational coupling depend on local agency or coherence.  Similarly, the floor stiffness `η_f(i)` could be a function of local packing disorder or coherence fragmentation.  To reduce parameter count:

1. **Identify state variables that correlate with the parameter’s role** (e.g., local disorder with stiffness, local coordination load with gravity mobility).
2. **Fit an empirical function** from these variables to measured fluxes in calibration experiments.
3. **Use the function as a constitutive relation** across simulations, eliminating the need to tune a global constant.

This approach turns adjustable knobs into measurable relationships, strengthening the predictive power of DET.


## 5 Ensuring independence from discretization and enforcing falsifiers

For all predictions, the numeric parameters in bucket C must not influence outcomes beyond error bars.  To verify this:

1. **Convergence tests.**  Rerun simulations with half the time step (`Δt/2`) and half the lattice spacing (`Δx/2`), or with larger domain sizes and different boundary conditions.  If results change significantly, refine until convergence or identify an unphysical parameter dependence.
2. **Boundary operator tests.**  Falsifiers F2 (coercion) and F3 (boundary redundancy) ensure that boundary operations do not modify agency directly and that enabling boundaries changes dynamics only through grace and healing【398607633750785†L425-L427】.  The boundary modules in `det_v6_2` enforce these conditions.
3. **Transferability falsifier.**  Introduce a formal falsifier *F_param* such that a parameter set measured in the metrology suite must predict outcomes across at least two distinct test classes (e.g., free fall and collision).  Failure indicates the module is a heuristic rather than a physical law.  This enforces the “no retuning” principle and turns parameter tuning into a scientific falsifier.


## 6 Conclusion

A systematic survey of the `det_v6_2` folder reveals a well‑defined set of tunable parameters.  When classified according to unit/scale setters (A), physical‑law constants (B) and numerical artefacts (C), it becomes clear that only a handful of constants truly define DET’s physics.  Their functions—momentum charging, angular‑momentum coupling, floor stiffness, gravitational coupling, coherence dynamics, structural debt formation, agency suppression and pointer reinforcement—are encoded in the theory card equations, providing explicit links between constants and observables.  The measurement rigs described above show how each constant can be fitted from controlled experiments in simulation or, where possible, physical analogues, and how these fits transfer to predictions in colliders.  The guiding principle is that **no retuning** is allowed: once measured, the constants should be universally valid within DET’s domain.  By promoting some constants to derived functions of local state and enforcing numerical convergence and falsifiers, the theory reduces parameter smell and strengthens its falsifiability.  This analysis establishes a practical metrology suite for DET v6.2 and lays a foundation for applying the theory to real‑world systems.