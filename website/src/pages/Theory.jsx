import { motion } from 'framer-motion';
import { useState } from 'react';

function Theory() {
  const [activeSection, setActiveSection] = useState('ontology');

  const sections = [
    { id: 'ontology', label: 'Ontology' },
    { id: 'variables', label: 'State Variables' },
    { id: 'presence', label: 'Presence & Time' },
    { id: 'flow', label: 'Flow Dynamics' },
    { id: 'gravity', label: 'Gravity' },
    { id: 'agency', label: 'Agency' },
    { id: 'quantum', label: 'Quantum' },
    { id: 'parameters', label: 'Parameters' },
  ];

  return (
    <div className="theory-page">
      <motion.header
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>Deep Existence Theory</h1>
        <p>Unified Canonical Formulation - Strictly Local, Law-Bound Boundary Action</p>
      </motion.header>

      <div className="theory-layout">
        <nav className="theory-nav">
          {sections.map(section => (
            <button
              key={section.id}
              className={`nav-item ${activeSection === section.id ? 'active' : ''}`}
              onClick={() => setActiveSection(section.id)}
            >
              {section.label}
            </button>
          ))}
        </nav>

        <div className="theory-content">
          {activeSection === 'ontology' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Ontological Commitments</h2>

              <div className="concept-card">
                <h3>The Scope Axiom</h3>
                <p>
                  DET begins from the assertion that present-moment participation requires
                  three minimal structural capacities:
                </p>
                <div className="triad-display">
                  <div className="triad-item">
                    <span className="symbol">I</span>
                    <span className="name">Information</span>
                    <span className="desc">Pattern continuity</span>
                  </div>
                  <div className="triad-item">
                    <span className="symbol">A</span>
                    <span className="name">Agency</span>
                    <span className="desc">Non-coercive choice</span>
                  </div>
                  <div className="triad-item">
                    <span className="symbol">K</span>
                    <span className="name">Movement</span>
                    <span className="desc">Time through events</span>
                  </div>
                </div>
              </div>

              <div className="entity-cards">
                <div className="entity-card">
                  <h4>Creatures (Agents)</h4>
                  <p>Constrained entities that:</p>
                  <ul>
                    <li>Store resource (F)</li>
                    <li>Participate in time</li>
                    <li>Form relations</li>
                    <li>Act through agency</li>
                  </ul>
                </div>

                <div className="entity-card">
                  <h4>Relations (Bonds)</h4>
                  <p>Local links that:</p>
                  <ul>
                    <li>Connect neighboring nodes</li>
                    <li>Carry coherence (C)</li>
                    <li>Store momentum (π)</li>
                    <li>Enable resource flow</li>
                  </ul>
                </div>

                <div className="entity-card">
                  <h4>Boundary Agent</h4>
                  <p>An unconstrained agent that:</p>
                  <ul>
                    <li>Does not accumulate past</li>
                    <li>Does not hoard</li>
                    <li>Is not subject to clocks, mass, or gravity</li>
                    <li>Acts only through law-bound operators</li>
                  </ul>
                </div>
              </div>

              <div className="key-principle">
                <h4>Strict Locality Principle</h4>
                <p>
                  All summations, averages, and normalizations are local unless explicitly bond-scoped.
                  There is no global state accessible to local dynamics. Disconnected components cannot
                  influence one another.
                </p>
                <div className="formula-display">
                  <code>∑(·) ≡ ∑_{'{k ∈ N_R(i)}'} (·)</code>
                </div>
              </div>
            </motion.section>
          )}

          {activeSection === 'variables' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>State Variables</h2>

              <div className="variable-group">
                <h3>Per-Creature Variables (node i)</h3>
                <table className="variable-table">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Range</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="var-symbol">F_i</td>
                      <td>≥ 0</td>
                      <td>Stored resource (F = F^op + F^locked)</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">q_i</td>
                      <td>[0, 1]</td>
                      <td>Structural debt (retained past)</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">τ_i</td>
                      <td>≥ 0</td>
                      <td>Proper time</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">σ_i</td>
                      <td>&gt; 0</td>
                      <td>Processing rate</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">a_i</td>
                      <td>[0, 1]</td>
                      <td>Agency (inviolable)</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">θ_i</td>
                      <td>S¹</td>
                      <td>Phase</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">k_i</td>
                      <td>N</td>
                      <td>Event count</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">r_i</td>
                      <td>≥ 0</td>
                      <td>Pointer record (measurement)</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="variable-group">
                <h3>Per-Bond Variables (edge i↔j)</h3>
                <table className="variable-table">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Range</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="var-symbol">σ_ij</td>
                      <td>&gt; 0</td>
                      <td>Bond conductivity</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">C_ij</td>
                      <td>[0, 1]</td>
                      <td>Coherence</td>
                    </tr>
                    <tr>
                      <td className="var-symbol">π_ij</td>
                      <td>R</td>
                      <td>Directed bond momentum</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="variable-group">
                <h3>Per-Plaquette Variables (face i,j,k,l)</h3>
                <table className="variable-table">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Range</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="var-symbol">L_ijkl</td>
                      <td>R</td>
                      <td>Plaquette angular momentum</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="concept-card">
                <h4>Q-Locking Contract</h4>
                <p>
                  DET treats q_i as retained past ("structural debt"). The default q-locking law
                  accumulates debt from net resource loss:
                </p>
                <div className="formula-display">
                  <code>q_i^+ = clip(q_i + α_q · max(0, -ΔF_i), 0, 1)</code>
                </div>
              </div>
            </motion.section>
          )}

          {activeSection === 'presence' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Presence and Time</h2>

              <div className="concept-card highlight">
                <h3>The Presence Formula</h3>
                <p>Presence P is the local clock rate - how fast proper time accumulates:</p>
                <div className="formula-display large">
                  <code>P_i = a_i · σ_i · (1 + F_i^op)⁻¹ · (1 + H_i)⁻¹ · γ_v⁻¹</code>
                </div>
                <div className="formula-terms">
                  <div className="term">
                    <span className="term-var">a_i</span>
                    <span className="term-desc">Agency - decision-making capacity</span>
                  </div>
                  <div className="term">
                    <span className="term-var">σ_i</span>
                    <span className="term-desc">Processing rate - intrinsic speed</span>
                  </div>
                  <div className="term">
                    <span className="term-var">(1+F)⁻¹</span>
                    <span className="term-desc">Resource load - gravitational dilation</span>
                  </div>
                  <div className="term">
                    <span className="term-var">(1+H)⁻¹</span>
                    <span className="term-desc">Coordination load - relational overhead</span>
                  </div>
                  <div className="term">
                    <span className="term-var">γ_v⁻¹</span>
                    <span className="term-desc">Kinematic factor - velocity dilation</span>
                  </div>
                </div>
              </div>

              <div className="concept-card">
                <h3>Gravitational Time Dilation</h3>
                <p>
                  The presence formula does not contain Φ explicitly. Gravitational time dilation
                  emerges through F-redistribution: gravitational flux accumulates F in potential
                  wells, which reduces P via the (1+F)⁻¹ factor.
                </p>
                <div className="formula-display">
                  <code>P/P_∞ = (1 + F_∞)/(1 + F)</code>
                </div>
                <div className="verification">
                  <span className="check">✓</span>
                  Verified to 0.43% against GPS satellite data
                </div>
              </div>

              <div className="concept-card">
                <h3>Kinematic Time Dilation</h3>
                <p>Moving clocks run slower due to the Lorentz factor:</p>
                <div className="formula-display">
                  <code>γ_v⁻¹ = √(1 - v²/c²)</code>
                </div>
                <p>In DET, this maps to bond momentum:</p>
                <div className="formula-display">
                  <code>γ_v⁻¹ ≈ (1 + π²/π_max²)^{'{-1/2}'}</code>
                </div>
                <div className="verification">
                  <span className="check">✓</span>
                  GPS: -7.21 μs/day (0.17% error)
                </div>
              </div>

              <div className="concept-card">
                <h3>Mass as Coordination Debt</h3>
                <p>Mass emerges as the reciprocal of presence:</p>
                <div className="formula-display">
                  <code>M_i ≡ P_i⁻¹</code>
                </div>
                <p>
                  Interpretation: mass is total coordination resistance to present-moment
                  participation under the DET clock law.
                </p>
              </div>
            </motion.section>
          )}

          {activeSection === 'flow' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Flow and Resource Dynamics</h2>

              <div className="concept-card">
                <h3>Total Flow Decomposition</h3>
                <p>Resource flow between nodes has five components:</p>
                <div className="formula-display large">
                  <code>J_{'{i→j}'} = J^diff + J^grav + J^mom + J^rot + J^floor</code>
                </div>
              </div>

              <div className="flow-components">
                <div className="flow-card">
                  <h4>Diffusive Flux</h4>
                  <p>Agency-gated transport from pressure/phase gradients:</p>
                  <div className="formula-display small">
                    <code>J^diff = g^a · σ · [√C · Im(ψ*ψ) + (1-√C)(F_i - F_j)]</code>
                  </div>
                  <p className="note">Quantum-classical interpolation via coherence C</p>
                </div>

                <div className="flow-card">
                  <h4>Gravitational Flux</h4>
                  <p>Drift down potential gradient:</p>
                  <div className="formula-display small">
                    <code>J^grav = μ_g · σ · (F_avg) · (Φ_i - Φ_j)</code>
                  </div>
                  <p className="note">Causes F accumulation in potential wells</p>
                </div>

                <div className="flow-card">
                  <h4>Momentum Flux</h4>
                  <p>Persistent directed transport:</p>
                  <div className="formula-display small">
                    <code>J^mom = μ_π · σ · π · (F_avg)</code>
                  </div>
                  <p className="note">Enables collision and approach dynamics</p>
                </div>

                <div className="flow-card">
                  <h4>Rotational Flux</h4>
                  <p>Divergence-free circulation:</p>
                  <div className="formula-display small">
                    <code>J^rot = μ_L · σ · F_avg · ∇⊥L</code>
                  </div>
                  <p className="note">Creates stable orbital motion</p>
                </div>

                <div className="flow-card">
                  <h4>Floor Repulsion</h4>
                  <p>Prevents infinite compression:</p>
                  <div className="formula-display small">
                    <code>J^floor = η_f · σ · (s_i + s_j) · (F_i - F_j)</code>
                  </div>
                  <p className="note">Activates when F &gt; F_core</p>
                </div>
              </div>

              <div className="concept-card">
                <h3>Momentum Dynamics</h3>
                <p>Bond momentum charges from diffusive flow and gravity:</p>
                <div className="formula-display">
                  <code>π^+ = (1 - λ_π·Δτ)π + α_π·J^diff·Δτ + β_g·g·Δτ</code>
                </div>
              </div>

              <div className="concept-card">
                <h3>Angular Momentum</h3>
                <p>Plaquette-based, charging from momentum curl:</p>
                <div className="formula-display">
                  <code>L^+ = (1 - λ_L·Δτ)L + α_L·curl(π)·Δτ</code>
                </div>
              </div>
            </motion.section>
          )}

          {activeSection === 'gravity' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Gravity Module</h2>

              <div className="concept-card highlight">
                <h3>Baseline-Referenced Gravity</h3>
                <p>
                  DET gravity is not an intrinsic force but an emergent potential field
                  sourced by the <strong>imbalance</strong> between local structural debt
                  and a dynamically computed baseline.
                </p>
                <div className="formula-display">
                  <code>ρ_i ≡ q_i - b_i</code>
                </div>
              </div>

              <div className="concept-card">
                <h3>Baseline Field</h3>
                <p>The baseline is a low-pass-filtered version of structure via screened Poisson:</p>
                <div className="formula-display">
                  <code>(L_σ · b)_i - α·b_i = -α·q_i</code>
                </div>
                <p className="note">This allows gravity to respond to structure variations, not absolute values.</p>
              </div>

              <div className="concept-card">
                <h3>Gravitational Potential</h3>
                <div className="formula-display large">
                  <code>(L_σ · Φ)_i = +κ · ρ_i</code>
                </div>
                <p>
                  With discrete Laplacian eigenvalues L_k &lt; 0 and ρ &gt; 0 near mass,
                  this yields Φ &lt; 0 (attractive gravity).
                </p>
              </div>

              <div className="concept-card">
                <h3>Lattice Correction Factor</h3>
                <p>The discrete Laplacian requires a correction η for continuum mapping:</p>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Lattice Size N</th>
                      <th>η</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>32</td><td>0.901</td></tr>
                    <tr><td>64</td><td>0.955</td></tr>
                    <tr><td>96</td><td>0.968</td></tr>
                    <tr><td>128</td><td>0.975</td></tr>
                  </tbody>
                </table>
              </div>

              <div className="concept-card">
                <h3>Effective G</h3>
                <div className="formula-display">
                  <code>G_eff = η·κ / (4π)</code>
                </div>
                <div className="verification">
                  <span className="check">✓</span>
                  Kepler's Third Law verified: T²/r³ = 0.4308 ± 1.2%
                </div>
              </div>
            </motion.section>
          )}

          {activeSection === 'agency' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Agency Dynamics (v6.4)</h2>

              <div className="concept-card highlight">
                <h3>Two-Component Agency Model</h3>
                <p>Agency dynamics reflect the matter/life duality:</p>
              </div>

              <div className="agency-components">
                <div className="agency-card">
                  <h4>A. Structural Ceiling (Matter Law)</h4>
                  <div className="formula-display">
                    <code>a_max = 1 / (1 + λ_a · q²)</code>
                  </div>
                  <p>
                    This is the <strong>ceiling</strong> that structure permits - what matter allows.
                    High structural debt (q→1) forces a_max→0 regardless of other factors.
                  </p>
                  <p className="emphasis">This is an upper bound, not a target.</p>
                </div>

                <div className="agency-card">
                  <h4>B. Relational Drive (Life Law)</h4>
                  <div className="formula-display">
                    <code>Δa_drive = γ(C) · (P_i - P̄_neighbors)</code>
                  </div>
                  <p>Where the coherence-gated drive coefficient is:</p>
                  <div className="formula-display small">
                    <code>γ(C) = γ_max · C^n (n ≥ 2)</code>
                  </div>
                  <p>
                    This is the <strong>will</strong> of the creature - where life chooses to go
                    within structural constraints. Only high-C entities can actively steer.
                  </p>
                </div>
              </div>

              <div className="concept-card">
                <h3>Unified Update</h3>
                <div className="formula-display large">
                  <code>a^+ = clip(a + β_a·(a_max - a) + Δa_drive, 0, a_max)</code>
                </div>
              </div>

              <div className="properties-grid">
                <div className="property">
                  <span className="check">✓</span>
                  <span>Structural ceiling inviolable: a ≤ a_max always</span>
                </div>
                <div className="property">
                  <span className="check">✓</span>
                  <span>High-q constrained regardless of coherence (Zombie Test)</span>
                </div>
                <div className="property">
                  <span className="check">✓</span>
                  <span>Only high-C entities can exercise relational drive</span>
                </div>
                <div className="property">
                  <span className="check">✓</span>
                  <span>Low-C entities passively relax toward a_max</span>
                </div>
                <div className="property">
                  <span className="check">✓</span>
                  <span>With C=0: pure target-tracking (a → a_max)</span>
                </div>
              </div>

              <div className="concept-card">
                <h3>The Zombie Test</h3>
                <p>
                  Verifies that "gravity trumps will" - structural debt imposes an inviolable
                  ceiling regardless of coherence. A high-debt entity (q=0.8) forced to
                  maximum coherence (C=1.0) cannot exceed a_max ≈ 0.05.
                </p>
                <div className="verification">
                  <span className="check">✓</span>
                  F_A1 PASSED: Matter/life duality verified
                </div>
              </div>
            </motion.section>
          )}

          {activeSection === 'quantum' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Quantum Behavior</h2>

              <div className="concept-card highlight">
                <h3>Retrocausal Locality</h3>
                <p>
                  DET resolves Bell's theorem by changing the ontology of measurement.
                  Measurement settings are future boundary conditions, not state readouts.
                </p>
              </div>

              <div className="comparison-table">
                <div className="comparison-row header">
                  <span>Standard HV</span>
                  <span>DET Retrocausal</span>
                </div>
                <div className="comparison-row">
                  <span>Measurement reads pre-existing state</span>
                  <span>Measurement is a boundary condition</span>
                </div>
                <div className="comparison-row">
                  <span>Forward time only</span>
                  <span>Block universe with variational principle</span>
                </div>
                <div className="comparison-row">
                  <span>Source determines outcomes</span>
                  <span>Source + measurements select history</span>
                </div>
              </div>

              <div className="concept-card">
                <h3>The Reconciliation Algorithm</h3>
                <ol className="algorithm-steps">
                  <li>
                    <strong>Preparation (t=0):</strong> Source creates entangled pair with shared θ, C
                  </li>
                  <li>
                    <strong>Selection (t=1):</strong> Detectors freely choose settings α and β
                  </li>
                  <li>
                    <strong>Reconciliation:</strong> Find history minimizing total Action
                    <div className="formula-display small">
                      <code>S = S_source + S_meas_A + S_meas_B + S_bond</code>
                    </div>
                  </li>
                  <li>
                    <strong>Measurement:</strong> Read outcomes from reconciled state
                  </li>
                </ol>
              </div>

              <div className="concept-card">
                <h3>Bell/CHSH Results</h3>
                <div className="results-grid">
                  <div className="result">
                    <span className="result-label">CHSH Value</span>
                    <span className="result-value">|S| = 2.41 ± 0.03</span>
                  </div>
                  <div className="result">
                    <span className="result-label">Classical Bound</span>
                    <span className="result-value">|S| ≤ 2</span>
                  </div>
                  <div className="result">
                    <span className="result-label">Quantum Maximum</span>
                    <span className="result-value">|S| ≤ 2√2 ≈ 2.83</span>
                  </div>
                </div>
                <div className="verification large">
                  <span className="check">✓</span>
                  BELL INEQUALITY VIOLATED
                </div>
              </div>

              <div className="concept-card">
                <h3>Coherence and Decoherence</h3>
                <p>Coherence C interpolates between quantum and classical:</p>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Coherence C</th>
                      <th>CHSH |S|</th>
                      <th>Regime</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>1.0</td><td>~2.4</td><td>Quantum</td></tr>
                    <tr><td>0.5</td><td>~1.2</td><td>Transition</td></tr>
                    <tr><td>0.0</td><td>~0</td><td>Classical</td></tr>
                  </tbody>
                </table>
              </div>

              <div className="properties-grid">
                <div className="property">
                  <span className="check">✓</span>
                  <span>No superluminal signaling: marginals independent</span>
                </div>
                <div className="property">
                  <span className="check">✓</span>
                  <span>No conspiracy: detector settings freely chosen</span>
                </div>
                <div className="property">
                  <span className="check">✓</span>
                  <span>Locality preserved: all dynamics strictly local</span>
                </div>
              </div>
            </motion.section>
          )}

          {activeSection === 'parameters' && (
            <motion.section
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="content-section"
            >
              <h2>Unified Parameters</h2>

              <div className="concept-card highlight">
                <h3>Parameter Unification</h3>
                <p>
                  The ~25 physical parameters reduce to <strong>12 base parameters</strong> via
                  recognized symmetries and derivation rules.
                </p>
              </div>

              <div className="param-table-container">
                <h4>Base Parameters (12)</h4>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Symbol</th>
                      <th>Default</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr><td>τ_base</td><td>0.02</td><td>Time/screening scale</td></tr>
                    <tr><td>σ_base</td><td>0.12</td><td>Charging rate scale</td></tr>
                    <tr><td>λ_base</td><td>0.008</td><td>Decay rate scale</td></tr>
                    <tr><td>μ_base</td><td>2.0</td><td>Mobility/power scale</td></tr>
                    <tr><td>κ_base</td><td>5.0</td><td>Coupling scale</td></tr>
                    <tr><td>C_0</td><td>0.15</td><td>Coherence scale</td></tr>
                    <tr><td>φ_L</td><td>0.5</td><td>Angular/momentum ratio</td></tr>
                    <tr><td>λ_a</td><td>30.0</td><td>Structural ceiling coupling</td></tr>
                    <tr><td>τ_eq_C</td><td>20.0</td><td>Coherence equilibration ratio</td></tr>
                    <tr><td>π_max</td><td>3.0</td><td>Maximum momentum</td></tr>
                    <tr><td>μ_π_factor</td><td>0.175</td><td>Momentum mobility factor</td></tr>
                    <tr><td>λ_L_factor</td><td>0.625</td><td>Angular decay factor</td></tr>
                  </tbody>
                </table>
              </div>

              <div className="concept-card">
                <h3>Identified Symmetries</h3>
                <div className="symmetries-grid">
                  <div className="symmetry">
                    <h5>Equal Values</h5>
                    <ul>
                      <li>α_grav = DT = τ_base = 0.02</li>
                      <li>α_π = η_floor = σ_base = 0.12</li>
                      <li>γ_a_max = C_init = C_0 = 0.15</li>
                      <li>floor_power = γ_a_power = μ_grav = 2.0</li>
                    </ul>
                  </div>
                  <div className="symmetry">
                    <h5>Factor Relationships</h5>
                    <ul>
                      <li>α_L = α_π / 2</li>
                      <li>β_g = 5 × μ_grav</li>
                      <li>β_a = 10 × τ_base</li>
                      <li>α_q = σ_base / 10</li>
                    </ul>
                  </div>
                  <div className="symmetry">
                    <h5>Golden Ratio (~φ ≈ 1.618)</h5>
                    <ul>
                      <li>λ_π / λ_L ≈ 1.6</li>
                      <li>L_max / π_max ≈ φ</li>
                    </ul>
                  </div>
                </div>
              </div>

              <div className="concept-card">
                <h3>Parameter Classification</h3>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Bucket</th>
                      <th>Definition</th>
                      <th>Examples</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td className="bucket-a">A (Unit/Scale)</td>
                      <td>Define units or baseline scale</td>
                      <td>DT, N, R, F_VAC, π_max</td>
                    </tr>
                    <tr>
                      <td className="bucket-b">B (Physical Law)</td>
                      <td>Control physical couplings</td>
                      <td>α_π, λ_π, κ, μ_g, α_C</td>
                    </tr>
                    <tr>
                      <td className="bucket-c">C (Numerical)</td>
                      <td>Control stability</td>
                      <td>outflow_limit, R_boundary</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </motion.section>
          )}
        </div>
      </div>
    </div>
  );
}

export default Theory;
