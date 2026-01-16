import { motion } from 'framer-motion';

function Novelty() {
  const novelties = [
    {
      category: 'Foundational',
      items: [
        {
          title: 'Time Emerges from Events',
          description: 'Time is not a pre-existing dimension but emerges from event counts and local interactions. Proper time τ accumulates through the Presence formula.',
          impact: 'Resolves the "problem of time" in quantum gravity by making time fundamentally relational.',
          formula: 'P = a·σ·(1+F)⁻¹·(1+H)⁻¹·γᵥ⁻¹'
        },
        {
          title: 'Mass as Coordination Debt',
          description: 'Mass M = P⁻¹ is not a fundamental property but the total coordination resistance to present-moment participation.',
          impact: 'Unifies inertial and gravitational mass through a single mechanism.',
          formula: 'M ≡ P⁻¹'
        },
        {
          title: 'Agency as Fundamental',
          description: 'Agency a ∈ [0,1] is a primitive state variable, not derived from other quantities. It represents decision-making capacity.',
          impact: 'Provides a foundation for consciousness-compatible physics without dualism.',
          formula: 'a_max = 1/(1 + λₐq²)'
        }
      ]
    },
    {
      category: 'Gravity',
      items: [
        {
          title: 'Baseline-Referenced Gravity',
          description: 'Gravity is sourced by the imbalance ρ = q - b between structural debt and a dynamically computed baseline, not absolute mass.',
          impact: 'Explains why empty space has no gravitational effect without fine-tuning.',
          formula: 'ρᵢ ≡ qᵢ - bᵢ'
        },
        {
          title: 'Gravitational Time Dilation via F-Redistribution',
          description: 'The Presence formula contains no explicit Φ. Time dilation emerges because gravitational flux accumulates resource F in potential wells.',
          impact: 'Unifies gravity and time dilation through a single mechanism.',
          formula: 'P/P∞ = (1 + F∞)/(1 + F)'
        },
        {
          title: 'Lattice Correction Factor',
          description: 'Derivable η ≈ 0.965 corrects the discrete Laplacian to match continuum physics. Not a free parameter.',
          impact: 'Enables precise mapping between discrete simulation and continuum predictions.',
          formula: 'G_eff = η·κ/(4π)'
        }
      ]
    },
    {
      category: 'Quantum',
      items: [
        {
          title: 'Retrocausal Bell Resolution',
          description: 'Bell violations through block-universe reconciliation, not hidden variables. Measurement settings are future boundary conditions.',
          impact: 'Resolves Bell\'s theorem while maintaining strict locality and no-signaling.',
          formula: 'S = S_source + S_meas_A + S_meas_B + S_bond'
        },
        {
          title: 'Coherence as Quantum-Classical Interpolator',
          description: 'Coherence C ∈ [0,1] continuously interpolates between quantum (high C) and classical (low C) behavior.',
          impact: 'Provides a mechanism for decoherence and the quantum-classical transition.',
          formula: 'J^diff = √C·Im(ψ*ψ) + (1-√C)(F_i - F_j)'
        },
        {
          title: 'No Wavefunction Collapse',
          description: 'Coherence dynamics replace collapse. The "collapse" is gradual decoherence via λ_C decay and measurement coupling λ_M.',
          impact: 'Eliminates the measurement problem by making collapse an emergent process.',
          formula: 'C⁺ = C + αC|J|Δτ - λCC Δτ - λMm√C Δτ'
        }
      ]
    },
    {
      category: 'Agency & Life',
      items: [
        {
          title: 'Two-Component Agency Model',
          description: 'Agency has two components: structural ceiling (what matter permits) and relational drive (where life chooses).',
          impact: 'Provides a physics-compatible framework for will and choice.',
          formula: 'a⁺ = clip(a + βₐ(a_max - a) + Δa_drive, 0, a_max)'
        },
        {
          title: 'Coherence-Gated Will',
          description: 'Only high-coherence entities can exercise relational drive. The drive coefficient scales as γ(C) = γ_max·C^n.',
          impact: 'Explains why classical objects don\'t exhibit "choice" while quantum systems might.',
          formula: 'Δa_drive = γ(C)·(P - P̄)'
        },
        {
          title: 'Grace as Constrained Action',
          description: 'The boundary agent can inject resource but only through law-bound, local, non-coercive operators. Agency-gated: a=0 blocks grace.',
          impact: 'Provides a mechanism for boundary action that respects local physics.',
          formula: 'G_{i→j} = η_g·g^a·Q·(d_i·r_j/Σr - d_j·r_i/Σr)'
        }
      ]
    },
    {
      category: 'Parameter Unification',
      items: [
        {
          title: '12 Base Parameters',
          description: '~25 physical parameters reduce to 12 base parameters through recognized symmetries and derivation rules.',
          impact: 'Reduces overfitting risk and suggests deeper structure.',
          formula: '25+ → 12 base'
        },
        {
          title: 'Golden Ratio Hints',
          description: 'Near-golden ratio relationships: λ_π/λ_L ≈ 1.6 ≈ φ and L_max/π_max ≈ φ.',
          impact: 'Suggests possible geometric origin of parameters.',
          formula: 'L_max/π_max ≈ φ ≈ 1.618'
        },
        {
          title: 'Verified Derivations',
          description: 'All 15/15 falsifiers pass with unified parameter derivation. No loss of physics from unification.',
          impact: 'Demonstrates the derivations are physically meaningful, not just coincidental.',
          formula: '15/15 PASS'
        }
      ]
    }
  ];

  const comparisons = [
    {
      aspect: 'Time',
      standard: 'Fundamental dimension (GR) or parameter (QM)',
      det: 'Emergent from event counts and interactions'
    },
    {
      aspect: 'Mass',
      standard: 'Fundamental property (rest mass)',
      det: 'Coordination resistance: M = P⁻¹'
    },
    {
      aspect: 'Gravity',
      standard: 'Spacetime curvature (GR)',
      det: 'Emergent from structural debt imbalance'
    },
    {
      aspect: 'Bell Violations',
      standard: 'Non-locality or superdeterminism required',
      det: 'Retrocausal reconciliation with strict locality'
    },
    {
      aspect: 'Wavefunction',
      standard: 'Fundamental object that collapses',
      det: 'Local wavefunction with continuous decoherence'
    },
    {
      aspect: 'Measurement',
      standard: 'Special process (collapse)',
      det: 'Future boundary condition in block universe'
    },
    {
      aspect: 'Locality',
      standard: 'Violated by entanglement',
      det: 'Strictly preserved (1 cell/step max)'
    },
    {
      aspect: 'Agency',
      standard: 'Not part of physics',
      det: 'Fundamental state variable'
    }
  ];

  return (
    <div className="novelty-page">
      <motion.header
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>Novel Contributions</h1>
        <p>What DET brings to physics that other frameworks don't</p>
      </motion.header>

      {/* Key Differentiators */}
      <section className="differentiators-section">
        <h2>Key Differentiators</h2>
        <div className="diff-grid">
          <motion.div
            className="diff-card"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
          >
            <div className="diff-icon">⏱</div>
            <h3>Emergent Time</h3>
            <p>Time doesn't exist independently - it emerges from local event counting.</p>
          </motion.div>
          <motion.div
            className="diff-card"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            <div className="diff-icon">📍</div>
            <h3>Strict Locality</h3>
            <p>All dynamics are nearest-neighbor. No action at a distance, ever.</p>
          </motion.div>
          <motion.div
            className="diff-card"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            <div className="diff-icon">🎯</div>
            <h3>Fundamental Agency</h3>
            <p>Decision-making capacity is a primitive, not derived from other physics.</p>
          </motion.div>
          <motion.div
            className="diff-card"
            initial={{ opacity: 0, scale: 0.9 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
          >
            <div className="diff-icon">🔄</div>
            <h3>Retrocausal QM</h3>
            <p>Bell violations via block-universe reconciliation, not non-locality.</p>
          </motion.div>
        </div>
      </section>

      {/* Comparison Table */}
      <section className="comparison-section">
        <h2>DET vs Standard Physics</h2>
        <div className="comparison-table-container">
          <table className="comparison-table">
            <thead>
              <tr>
                <th>Aspect</th>
                <th>Standard Physics</th>
                <th>DET Approach</th>
              </tr>
            </thead>
            <tbody>
              {comparisons.map((row, index) => (
                <motion.tr
                  key={row.aspect}
                  initial={{ opacity: 0, x: -20 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: index * 0.05 }}
                >
                  <td className="aspect-cell">{row.aspect}</td>
                  <td className="standard-cell">{row.standard}</td>
                  <td className="det-cell">{row.det}</td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Detailed Novelties */}
      {novelties.map((category, catIndex) => (
        <section key={category.category} className="novelty-section">
          <motion.h2
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            {category.category}
          </motion.h2>
          <div className="novelty-grid">
            {category.items.map((item, index) => (
              <motion.div
                key={item.title}
                className="novelty-card"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: index * 0.1 }}
              >
                <h3>{item.title}</h3>
                <p className="novelty-description">{item.description}</p>
                <div className="novelty-formula">
                  <code>{item.formula}</code>
                </div>
                <div className="novelty-impact">
                  <span className="impact-label">Impact:</span>
                  <span className="impact-text">{item.impact}</span>
                </div>
              </motion.div>
            ))}
          </div>
        </section>
      ))}

      {/* What DET Does NOT Claim */}
      <section className="non-claims-section">
        <h2>What DET Does NOT Claim</h2>
        <div className="non-claims-grid">
          <div className="non-claim">
            <span className="x-mark">✗</span>
            <p>DET does not claim to be a "theory of everything" - it's a framework for emergent physics.</p>
          </div>
          <div className="non-claim">
            <span className="x-mark">✗</span>
            <p>DET does not claim to explain consciousness - only to provide physics compatible with agency.</p>
          </div>
          <div className="non-claim">
            <span className="x-mark">✗</span>
            <p>DET does not claim to replace QFT - it offers an alternative ontology for quantum behavior.</p>
          </div>
          <div className="non-claim">
            <span className="x-mark">✗</span>
            <p>DET does not claim perfect agreement with all experiments - only with those tested so far.</p>
          </div>
        </div>
      </section>

      {/* Future Directions */}
      <section className="future-section">
        <h2>Future Directions</h2>
        <div className="future-grid">
          <div className="future-card">
            <h3>Completed in v6.4</h3>
            <ul>
              <li>✓ External G calibration methods</li>
              <li>✓ Galaxy rotation curve fitting</li>
              <li>✓ Gravitational lensing ray-tracing</li>
              <li>✓ Cosmological scaling analysis</li>
              <li>✓ Black hole thermodynamics</li>
              <li>✓ Quantum-classical transition</li>
            </ul>
          </div>
          <div className="future-card">
            <h3>Under Investigation</h3>
            <ul>
              <li>Dark matter implications</li>
              <li>Early universe cosmology</li>
              <li>Particle physics connections</li>
              <li>Further parameter unification</li>
              <li>Experimental proposals</li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Novelty;
