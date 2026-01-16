import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

function Home() {
  const fadeIn = {
    initial: { opacity: 0, y: 20 },
    animate: { opacity: 1, y: 0 },
    transition: { duration: 0.6 }
  };

  const stats = [
    { value: '22', label: 'Falsifiers Passed' },
    { value: '0.35%', label: 'GPS Accuracy' },
    { value: '2.41', label: 'Bell CHSH Value' },
    { value: '12', label: 'Base Parameters' },
  ];

  const features = [
    {
      title: 'Emergent Time',
      description: 'Time is not fundamental. It emerges from event counts and local interactions among constrained agents.',
      icon: '⏱',
    },
    {
      title: 'Relational Gravity',
      description: 'Gravity arises from structural debt imbalances, not intrinsic force. Verified against Kepler\'s laws.',
      icon: '🌍',
    },
    {
      title: 'Quantum Coherence',
      description: 'Quantum behavior through coherence C and retrocausal reconciliation. Bell violations achieved.',
      icon: '⚛',
    },
    {
      title: 'Agency Dynamics',
      description: 'Two-component agency: structural ceiling from matter, relational drive from coherent choice.',
      icon: '🎯',
    },
    {
      title: 'Strict Locality',
      description: 'All interactions nearest-neighbor only. Maximum propagation: 1 cell/timestep. No action at a distance.',
      icon: '📍',
    },
    {
      title: 'Unified Parameters',
      description: '25+ physical parameters reduced to 12 base parameters through recognized symmetries.',
      icon: '🔧',
    },
  ];

  return (
    <div className="home">
      {/* Hero Section */}
      <section className="hero">
        <motion.div className="hero-content" {...fadeIn}>
          <div className="hero-badge">v6.3 Unified Canonical Formulation</div>
          <h1 className="hero-title">
            Deep Existence<br />
            <span className="gradient-text">Theory</span>
          </h1>
          <p className="hero-subtitle">
            A unified framework where time, mass, gravity, and quantum behavior emerge
            from local interactions among constrained agents.
          </p>
          <div className="hero-buttons">
            <Link to="/simulator" className="btn btn-primary">
              Launch 3D Simulator
            </Link>
            <Link to="/theory" className="btn btn-secondary">
              Explore Theory
            </Link>
          </div>
        </motion.div>

        <motion.div
          className="hero-visual"
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.8, delay: 0.2 }}
        >
          <div className="hero-orb">
            <div className="orb-ring orb-ring-1"></div>
            <div className="orb-ring orb-ring-2"></div>
            <div className="orb-ring orb-ring-3"></div>
            <div className="orb-core"></div>
          </div>
        </motion.div>
      </section>

      {/* Stats Section */}
      <section className="stats-section">
        <div className="stats-grid">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              className="stat-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <span className="stat-value">{stat.value}</span>
              <span className="stat-label">{stat.label}</span>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Core Thesis */}
      <section className="thesis-section">
        <motion.div
          className="thesis-card"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2>Core Thesis</h2>
          <blockquote>
            "Time, mass, gravity, and quantum behavior emerge from local interactions
            among constrained agents. Any boundary action manifests locally and non-coercively."
          </blockquote>
          <div className="thesis-principles">
            <div className="principle">
              <strong>I</strong>
              <span>Information</span>
              <p>Pattern continuity</p>
            </div>
            <div className="principle">
              <strong>A</strong>
              <span>Agency</span>
              <p>Non-coercive choice</p>
            </div>
            <div className="principle">
              <strong>K</strong>
              <span>Movement</span>
              <p>Time through events</p>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Features Grid */}
      <section className="features-section">
        <h2 className="section-title">Key Concepts</h2>
        <div className="features-grid">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="feature-card"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <span className="feature-icon">{feature.icon}</span>
              <h3>{feature.title}</h3>
              <p>{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* Presence Formula */}
      <section className="formula-section">
        <motion.div
          className="formula-card"
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
        >
          <h2>The Presence Formula</h2>
          <p className="formula-intro">Local clock rate emerges from agency, processing, resource, and coordination:</p>
          <div className="formula">
            <code>P = a · σ · (1 + F)⁻¹ · (1 + H)⁻¹ · γᵥ⁻¹</code>
          </div>
          <div className="formula-breakdown">
            <div className="formula-term">
              <span className="term-symbol">a</span>
              <span className="term-name">Agency</span>
              <span className="term-range">[0, 1]</span>
            </div>
            <div className="formula-term">
              <span className="term-symbol">σ</span>
              <span className="term-name">Processing Rate</span>
              <span className="term-range">&gt; 0</span>
            </div>
            <div className="formula-term">
              <span className="term-symbol">F</span>
              <span className="term-name">Resource</span>
              <span className="term-range">≥ 0</span>
            </div>
            <div className="formula-term">
              <span className="term-symbol">H</span>
              <span className="term-name">Coordination Load</span>
              <span className="term-range">≥ 0</span>
            </div>
            <div className="formula-term">
              <span className="term-symbol">γᵥ</span>
              <span className="term-name">Kinematic Factor</span>
              <span className="term-range">≥ 1</span>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Validation Highlights */}
      <section className="validation-section">
        <h2 className="section-title">Validated Against Real Physics</h2>
        <div className="validation-grid">
          <motion.div
            className="validation-card"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h3>GPS Time Dilation</h3>
            <div className="validation-result">
              <span className="result-value">+38.52 μs/day</span>
              <span className="result-label">Net relativistic effect</span>
            </div>
            <div className="validation-breakdown">
              <div className="breakdown-item">
                <span>Gravitational:</span>
                <span>+45.85 μs/day</span>
              </div>
              <div className="breakdown-item">
                <span>Kinematic:</span>
                <span>-7.21 μs/day</span>
              </div>
              <div className="breakdown-item accuracy">
                <span>Accuracy:</span>
                <span>0.35%</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="validation-card"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
          >
            <h3>Hafele-Keating Experiment</h3>
            <div className="validation-result">
              <span className="result-value">1972 Atomic Clocks</span>
              <span className="result-label">Circumnavigation test</span>
            </div>
            <div className="validation-breakdown">
              <div className="breakdown-item">
                <span>Eastward:</span>
                <span>-63 ns (obs: -59±10)</span>
              </div>
              <div className="breakdown-item">
                <span>Westward:</span>
                <span>+296 ns (obs: +273±7)</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="validation-card"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            <h3>Bell/CHSH Violation</h3>
            <div className="validation-result">
              <span className="result-value">|S| = 2.41</span>
              <span className="result-label">Exceeds classical bound of 2</span>
            </div>
            <div className="validation-breakdown">
              <div className="breakdown-item">
                <span>Classical limit:</span>
                <span>|S| ≤ 2</span>
              </div>
              <div className="breakdown-item">
                <span>Quantum max:</span>
                <span>|S| ≤ 2√2 ≈ 2.83</span>
              </div>
              <div className="breakdown-item">
                <span>Mechanism:</span>
                <span>Retrocausal</span>
              </div>
            </div>
          </motion.div>

          <motion.div
            className="validation-card"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.1 }}
          >
            <h3>Kepler's Third Law</h3>
            <div className="validation-result">
              <span className="result-value">T²/r³ = 0.4308</span>
              <span className="result-label">Constant across orbits</span>
            </div>
            <div className="validation-breakdown">
              <div className="breakdown-item">
                <span>Coefficient of Variation:</span>
                <span>1.2%</span>
              </div>
              <div className="breakdown-item">
                <span>Eccentricity:</span>
                <span>&lt; 0.03</span>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="cta-section">
        <motion.div
          className="cta-card"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <h2>Experience the Universe Evolving</h2>
          <p>Watch DET dynamics in real-time with our interactive 3D simulator. See particles, gravity, coherence, and emergent time unfold.</p>
          <Link to="/simulator" className="btn btn-primary btn-large">
            Launch 3D Simulator
          </Link>
        </motion.div>
      </section>
    </div>
  );
}

export default Home;
