import { motion } from 'framer-motion';
import { useState } from 'react';

function Falsifiers() {
  const [filter, setFilter] = useState('all');

  const falsifiers = [
    // Core Falsifiers
    { id: 'F1', name: 'Locality Violation', category: 'core', status: 'pass',
      description: 'Adding causally disconnected nodes changes dynamics within a subgraph',
      result: 'Max propagation speed: 1.0 cells/step' },
    { id: 'F2', name: 'Coercion', category: 'core', status: 'pass',
      description: 'A node with a_i=0 receives grace injection or bond healing',
      result: 'Sentinel grace: 0.00e+00' },
    { id: 'F3', name: 'Boundary Redundancy', category: 'core', status: 'pass',
      description: 'Boundary-enabled and disabled systems are qualitatively indistinguishable',
      result: 'Grace ON differs from OFF' },
    { id: 'F4', name: 'No Regime Transition', category: 'core', status: 'pass',
      description: 'Increasing ⟨a⟩ fails to transition from low- to high-coherence regimes',
      result: 'Smooth regime transitions' },
    { id: 'F5', name: 'Hidden Global Aggregates', category: 'core', status: 'pass',
      description: 'Dynamics depend on sums/averages outside the local neighborhood',
      result: 'Max regional difference: 7.63e-17' },
    { id: 'F6', name: 'Binding Failure', category: 'core', status: 'pass',
      description: 'With gravity enabled, two bodies with q>0 fail to form a bound state',
      result: 'Binding achieved, min separation: 0.0' },
    { id: 'F7', name: 'Mass Non-Conservation', category: 'core', status: 'pass',
      description: 'Total mass ΣF_i drifts by >10% in a closed system',
      result: 'Mass drift: 0.00%' },
    { id: 'F8', name: 'Momentum Pushes Vacuum', category: 'core', status: 'pass',
      description: 'Non-zero momentum π in a zero-resource F≈0 region produces sustained transport',
      result: 'Vacuum momentum: no transport' },
    { id: 'F9', name: 'Spontaneous Drift', category: 'core', status: 'pass',
      description: 'A symmetric system develops net COM drift without stochastic input',
      result: 'Max COM drift: 0.04 cells' },
    { id: 'F10', name: 'Regime Discontinuity', category: 'core', status: 'pass',
      description: 'Scanning λ_π produces discontinuous jumps in collision outcomes',
      result: 'No discontinuities' },

    // Angular Momentum
    { id: 'F_L1', name: 'Rotational Conservation', category: 'angular', status: 'pass',
      description: 'With only rotational flux active, total mass is not conserved',
      result: 'Mass error: 1.20e-16' },
    { id: 'F_L2', name: 'Vacuum Spin Transport', category: 'angular', status: 'pass',
      description: 'Rotational flux does not vanish in vacuum',
      result: 'Linear scaling with F_VAC' },
    { id: 'F_L3', name: 'Orbital Capture Failure', category: 'angular', status: 'pass',
      description: 'With angular momentum enabled, non-head-on collisions fail to produce stable orbits',
      result: '15.16 revolutions achieved' },

    // Gravitational Time Dilation
    { id: 'F_GTD1', name: 'Presence Formula', category: 'gtd', status: 'pass',
      description: 'P ≠ a·σ/(1+F)/(1+H) to numerical precision',
      result: 'Formula correctly implemented' },
    { id: 'F_GTD2', name: 'Clock Rate Scaling', category: 'gtd', status: 'pass',
      description: 'P/P_∞ ≠ (1+F_∞)/(1+F) by >0.5%',
      result: 'Dilation factor: 205.7' },

    // Kinematic Time Dilation
    { id: 'F_KTD1', name: 'Kinematic Formula', category: 'ktd', status: 'pass',
      description: 'P_moving/P_rest ≠ γ_v^(-1) by >2%',
      result: 'Formula: γ_v^(-1) = √(1-v²/c²)' },
    { id: 'F_KTD2', name: 'GPS Kinematic Effect', category: 'ktd', status: 'pass',
      description: 'Kinematic dilation ≠ -7.2 μs/day by >5%',
      result: '-7.21 μs/day (0.17% error)' },
    { id: 'F_KTD3', name: 'Hafele-Keating Eastward', category: 'ktd', status: 'pass',
      description: 'Eastward flight doesn\'t lose time',
      result: '-63 ns (obs: -59±10 ns)' },
    { id: 'F_KTD4', name: 'Hafele-Keating Westward', category: 'ktd', status: 'pass',
      description: 'Westward flight doesn\'t gain time',
      result: '+296 ns (obs: +273±7 ns)' },
    { id: 'F_KTD5', name: 'Combined GPS Effect', category: 'ktd', status: 'pass',
      description: 'Net relativistic effect ≠ +38.65 μs/day by >5%',
      result: '+38.52 μs/day (0.35% error)' },

    // Agency
    { id: 'F_A1', name: 'Zombie Test', category: 'agency', status: 'pass',
      description: 'High-debt node with forced high-C exceeds structural ceiling',
      result: 'a < a_max with C=1.0, q=0.8' },

    // Kepler
    { id: 'F_K1', name: 'Kepler\'s Third Law', category: 'kepler', status: 'pass',
      description: 'T²/r³ ratio varies by more than 20% across orbital radii',
      result: 'T²/r³ = 0.4308 ± 1.2%' },

    // Bell
    { id: 'F_Bell', name: 'Bell Violation', category: 'bell', status: 'pass',
      description: 'CHSH value |S| ≤ 2 for entangled pairs',
      result: '|S| = 2.41 > 2.0' },
  ];

  const categories = [
    { id: 'all', label: 'All (22)' },
    { id: 'core', label: 'Core (10)' },
    { id: 'angular', label: 'Angular (3)' },
    { id: 'gtd', label: 'Grav. Dilation (2)' },
    { id: 'ktd', label: 'Kin. Dilation (5)' },
    { id: 'agency', label: 'Agency (1)' },
    { id: 'kepler', label: 'Kepler (1)' },
    { id: 'bell', label: 'Bell (1)' },
  ];

  const filteredFalsifiers = filter === 'all'
    ? falsifiers
    : falsifiers.filter(f => f.category === filter);

  const passCount = falsifiers.filter(f => f.status === 'pass').length;

  return (
    <div className="falsifiers-page">
      <motion.header
        className="page-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>Falsifiers</h1>
        <p>The theory is false if any condition below holds under canonical rules</p>
      </motion.header>

      <div className="falsifier-stats">
        <div className="stat-large">
          <span className="stat-number">{passCount}/{falsifiers.length}</span>
          <span className="stat-label">Tests Passed</span>
        </div>
        <div className="stat-bar">
          <div className="stat-bar-fill" style={{ width: `${(passCount / falsifiers.length) * 100}%` }}></div>
        </div>
      </div>

      <div className="filter-buttons">
        {categories.map(cat => (
          <button
            key={cat.id}
            className={`filter-btn ${filter === cat.id ? 'active' : ''}`}
            onClick={() => setFilter(cat.id)}
          >
            {cat.label}
          </button>
        ))}
      </div>

      <div className="falsifiers-grid">
        {filteredFalsifiers.map((f, index) => (
          <motion.div
            key={f.id}
            className={`falsifier-card ${f.status}`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
          >
            <div className="falsifier-header">
              <span className="falsifier-id">{f.id}</span>
              <span className={`falsifier-status ${f.status}`}>
                {f.status === 'pass' ? '✓ PASS' : '✗ FAIL'}
              </span>
            </div>
            <h3 className="falsifier-name">{f.name}</h3>
            <p className="falsifier-description">{f.description}</p>
            <div className="falsifier-result">
              <span className="result-label">Result:</span>
              <span className="result-value">{f.result}</span>
            </div>
          </motion.div>
        ))}
      </div>

      <section className="falsifier-explanation">
        <h2>What Makes DET Falsifiable?</h2>
        <div className="explanation-grid">
          <div className="explanation-card">
            <h3>Strict Locality</h3>
            <p>
              All interactions must be nearest-neighbor only. If any test shows information
              propagating faster than 1 cell/timestep, the theory is falsified.
            </p>
          </div>
          <div className="explanation-card">
            <h3>Conservation Laws</h3>
            <p>
              Total resource F must be conserved (except at boundaries). Any unexplained
              creation or destruction of resource falsifies the theory.
            </p>
          </div>
          <div className="explanation-card">
            <h3>Agency Inviolability</h3>
            <p>
              No external force can modify agency a_i directly. Grace must respect a=0
              boundaries. Any coercion falsifies the theory.
            </p>
          </div>
          <div className="explanation-card">
            <h3>Experimental Agreement</h3>
            <p>
              DET must reproduce known physics: GPS time dilation, Kepler's laws,
              Bell violations. Disagreement beyond error bounds falsifies the theory.
            </p>
          </div>
        </div>
      </section>

      <section className="critical-tests">
        <h2>Critical Tests</h2>
        <div className="critical-grid">
          <div className="critical-card">
            <h3>The Zombie Test (F_A1)</h3>
            <p>
              Verifies that "gravity trumps will" - structural debt imposes an inviolable
              ceiling on agency regardless of coherence. A high-debt entity (q=0.8) with
              forced maximum coherence (C=1.0) cannot exceed a_max ≈ 0.05.
            </p>
            <div className="critical-result">
              <span className="check">✓</span>
              Matter/life duality verified
            </div>
          </div>
          <div className="critical-card">
            <h3>Bell/CHSH (F_Bell)</h3>
            <p>
              Tests whether DET can produce quantum correlations via retrocausal reconciliation.
              Classical hidden variable theories are bounded by |S| ≤ 2. DET achieves |S| = 2.41.
            </p>
            <div className="critical-result">
              <span className="check">✓</span>
              Bell inequality violated
            </div>
          </div>
          <div className="critical-card">
            <h3>Kepler Standard Candle (F_K1)</h3>
            <p>
              Verifies DET produces physically correct Newtonian gravity. Orbital periods
              must satisfy T² ∝ r³. Measured coefficient of variation: 1.2%.
            </p>
            <div className="critical-result">
              <span className="check">✓</span>
              Kepler's Third Law satisfied
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Falsifiers;
