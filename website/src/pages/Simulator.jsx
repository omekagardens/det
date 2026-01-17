import { useRef, useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import DETParticleUniverse from '../simulator/DETParticleUniverse';

/**
 * Beautiful DET Universe Simulator
 * Visualizes Deep Existence Theory particle dynamics with:
 * - Glowing particles sized by Resource (F)
 * - Colors based on Structural Debt (q) and Agency (a)
 * - Coherence bonds between phase-aligned particles
 * - Trails showing particle history
 * - Time dilation effects from Presence (P)
 */

function Simulator() {
  const canvasRef = useRef(null);
  const universeRef = useRef(null);
  const animationRef = useRef(null);

  const [running, setRunning] = useState(true);
  const [scenario, setScenario] = useState('galaxy');
  const [showTrails, setShowTrails] = useState(true);
  const [showBonds, setShowBonds] = useState(true);
  const [showGlow, setShowGlow] = useState(true);
  const [particleCount, setParticleCount] = useState(200);
  const [stats, setStats] = useState({});
  const [colorMode, setColorMode] = useState('agency'); // agency, structure, presence, phase
  const [drawMode, setDrawMode] = useState(false);
  const [boundaryEnabled, setBoundaryEnabled] = useState(true);
  const [graceEnabled, setGraceEnabled] = useState(true);
  const [replicationEnabled, setReplicationEnabled] = useState(false);

  // Drawing state
  const isDrawing = useRef(false);
  const lastDrawPos = useRef({ x: 0, y: 0 });
  const lastDrawTime = useRef(0);

  // Initialize universe
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;

    universeRef.current = new DETParticleUniverse({
      numParticles: particleCount,
      width: canvas.width,
      height: canvas.height,
      boundaryEnabled,
      graceEnabled,
      replicationEnabled,
    });

    universeRef.current.setupScenario(scenario);
  }, [particleCount]);

  // Handle scenario change
  const changeScenario = useCallback((newScenario) => {
    if (universeRef.current) {
      universeRef.current.setupScenario(newScenario);
      setScenario(newScenario);
    }
  }, []);

  // Get particle color based on DET variables
  const getParticleColor = useCallback((particle, mode) => {
    switch (mode) {
      case 'agency': {
        // Agency: cyan (low) -> magenta (high)
        const h = 180 + particle.a * 120;
        const s = 80 + particle.a * 20;
        const l = 50 + particle.a * 30;
        return `hsl(${h}, ${s}%, ${l}%)`;
      }
      case 'structure': {
        // Structural debt: blue (low q) -> red (high q)
        const h = 240 - particle.q * 240;
        const s = 70 + particle.q * 30;
        const l = 50 + (1 - particle.q) * 30;
        return `hsl(${h}, ${s}%, ${l}%)`;
      }
      case 'presence': {
        // Presence: dim red (low P) -> bright white (high P)
        const l = 30 + particle.P * 70;
        const s = 100 - particle.P * 50;
        return `hsl(60, ${s}%, ${l}%)`;
      }
      case 'phase': {
        // Phase: rainbow cycle
        const h = (particle.theta / (Math.PI * 2)) * 360;
        return `hsl(${h}, 90%, 60%)`;
      }
      case 'resource': {
        // Resource F: dark purple (low) -> bright gold (high)
        const normalized = Math.min(1, particle.F / 2);
        const h = 280 - normalized * 230;
        const l = 40 + normalized * 40;
        return `hsl(${h}, 85%, ${l}%)`;
      }
      default:
        return '#fff';
    }
  }, []);

  // Render loop
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    const render = () => {
      const universe = universeRef.current;
      if (!universe) return;

      // Run simulation
      if (running) {
        universe.step_simulation();
      }

      const width = canvas.width;
      const height = canvas.height;

      // Clear with fade effect for trails
      if (showTrails) {
        ctx.fillStyle = 'rgba(5, 5, 15, 0.15)';
      } else {
        ctx.fillStyle = 'rgba(5, 5, 15, 1)';
      }
      ctx.fillRect(0, 0, width, height);

      // Draw coherence bonds
      if (showBonds) {
        const bonds = universe.getBonds();
        for (const bond of bonds) {
          const alpha = Math.min(0.9, bond.C * 0.8);
          ctx.beginPath();
          ctx.moveTo(bond.p1.x, bond.p1.y);
          ctx.lineTo(bond.p2.x, bond.p2.y);

          // Gradient based on phase alignment
          const gradient = ctx.createLinearGradient(
            bond.p1.x, bond.p1.y, bond.p2.x, bond.p2.y
          );
          const color1 = getParticleColor(bond.p1, colorMode);
          const color2 = getParticleColor(bond.p2, colorMode);
          gradient.addColorStop(0, color1.replace(')', `, ${alpha})`).replace('hsl', 'hsla'));
          gradient.addColorStop(1, color2.replace(')', `, ${alpha})`).replace('hsl', 'hsla'));

          ctx.strokeStyle = gradient;
          ctx.lineWidth = 1 + bond.C * 3; // Thicker bonds for higher coherence
          ctx.stroke();

          // Show angular momentum as curved lines for strong bonds
          if (Math.abs(bond.L) > 0.5 && bond.C > 0.2) {
            const midX = (bond.p1.x + bond.p2.x) / 2;
            const midY = (bond.p1.y + bond.p2.y) / 2;
            const radius = 5 + Math.abs(bond.L) * 2;
            ctx.beginPath();
            ctx.arc(midX, midY, radius, 0, Math.PI * (bond.L > 0 ? 1 : -1));
            ctx.strokeStyle = `rgba(255, 200, 100, ${bond.C * 0.5})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        }
      }

      // Draw particles
      for (const p of universe.particles) {
        if (!p.alive) continue; // Skip dead particles
        if (p.isDormant) continue; // Skip dormant particles (they're not visible)

        // Boundary particles rendered differently
        if (p.isBoundary) {
          const size = 4 + p.F * 3;
          ctx.beginPath();
          ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(200, 80%, 60%, 0.7)`;
          ctx.fill();
          ctx.strokeStyle = `hsla(200, 90%, 70%, 0.9)`;
          ctx.lineWidth = 2;
          ctx.stroke();
          continue;
        }

        const color = getParticleColor(p, colorMode);
        const size = 3 + p.F * 8; // Size based on resource

        // Glow effect
        if (showGlow) {
          const glowSize = size * 3;
          const gradient = ctx.createRadialGradient(
            p.x, p.y, 0,
            p.x, p.y, glowSize
          );
          gradient.addColorStop(0, color.replace(')', ', 0.8)').replace('hsl', 'hsla'));
          gradient.addColorStop(0.4, color.replace(')', ', 0.3)').replace('hsl', 'hsla'));
          gradient.addColorStop(1, color.replace(')', ', 0)').replace('hsl', 'hsla'));

          ctx.beginPath();
          ctx.arc(p.x, p.y, glowSize, 0, Math.PI * 2);
          ctx.fillStyle = gradient;
          ctx.fill();
        }

        // Core particle
        ctx.beginPath();
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();

        // Bright center
        ctx.beginPath();
        ctx.arc(p.x, p.y, size * 0.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255, 255, 255, ${0.5 + p.P * 0.5})`;
        ctx.fill();
      }

      // Update stats periodically
      if (universe.step % 10 === 0) {
        setStats(universe.getStats());
      }

      animationRef.current = requestAnimationFrame(render);
    };

    animationRef.current = requestAnimationFrame(render);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [running, showTrails, showBonds, showGlow, colorMode, getParticleColor]);

  // Handle resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;

      if (universeRef.current) {
        universeRef.current.width = canvas.width;
        universeRef.current.height = canvas.height;
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Handle drawing particles
  const getCanvasCoords = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
      x: (clientX - rect.left) * scaleX,
      y: (clientY - rect.top) * scaleY
    };
  }, []);

  const handlePointerDown = useCallback((e) => {
    if (!drawMode) return;

    isDrawing.current = true;
    const pos = getCanvasCoords(e);
    lastDrawPos.current = pos;
    lastDrawTime.current = performance.now();

    // Add first particle
    if (universeRef.current) {
      universeRef.current.addParticle(pos.x, pos.y, 0, 0);
    }
  }, [drawMode, getCanvasCoords]);

  const handlePointerMove = useCallback((e) => {
    if (!drawMode || !isDrawing.current) return;

    const pos = getCanvasCoords(e);
    const now = performance.now();
    const dt = Math.max(1, now - lastDrawTime.current) / 1000;

    // Calculate velocity from drag
    const vx = (pos.x - lastDrawPos.current.x) / dt * 0.01;
    const vy = (pos.y - lastDrawPos.current.y) / dt * 0.01;

    // Add particle with velocity based on drag direction
    if (universeRef.current) {
      universeRef.current.addParticle(pos.x, pos.y, vx, vy);
    }

    lastDrawPos.current = pos;
    lastDrawTime.current = now;
  }, [drawMode, getCanvasCoords]);

  const handlePointerUp = useCallback(() => {
    isDrawing.current = false;
  }, []);

  const scenarios = [
    { id: 'galaxy', label: 'Spiral Galaxy', desc: 'Rotating spiral with central mass' },
    { id: 'orbiting', label: 'Orbital System', desc: 'Particles orbiting central mass' },
    { id: 'collision', label: 'Cluster Collision', desc: 'Two clusters colliding' },
    { id: 'expansion', label: 'Big Bang', desc: 'Expansion from singularity' },
    { id: 'quantum', label: 'Quantum Coherent', desc: 'High coherence, phase-aligned' },
    { id: 'grace-demo', label: 'Grace Demo', desc: 'Resource redistribution via grace' },
    { id: 'replication-demo', label: 'Replication', desc: 'Node division via fork mechanism' },
    { id: 'random', label: 'Random', desc: 'Random initial conditions' },
  ];

  const colorModes = [
    { id: 'agency', label: 'Agency (a)', desc: 'Decision-making capacity' },
    { id: 'structure', label: 'Structure (q)', desc: 'Structural debt / mass' },
    { id: 'presence', label: 'Presence (P)', desc: 'Local clock rate' },
    { id: 'phase', label: 'Phase (θ)', desc: 'Quantum phase' },
    { id: 'resource', label: 'Resource (F)', desc: 'Stored energy' },
  ];

  return (
    <div className="simulator-page fullscreen">
      <canvas
        ref={canvasRef}
        className="simulator-canvas"
        style={{ cursor: drawMode ? 'crosshair' : 'default' }}
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={handlePointerUp}
        onTouchStart={handlePointerDown}
        onTouchMove={handlePointerMove}
        onTouchEnd={handlePointerUp}
      />

      {/* Stats Overlay */}
      <div className="sim-stats-overlay">
        <div className="sim-title">
          <span className="det-logo">DET</span>
          <span className="version">v6.3 Universe</span>
        </div>
        <div className="sim-stat">
          <span className="label">Time</span>
          <span className="value">{stats.time?.toFixed(1) || 0}</span>
        </div>
        <div className="sim-stat">
          <span className="label">Particles</span>
          <span className="value">{stats.particles || 0}</span>
        </div>
        <div className="sim-stat">
          <span className="label">Bonds</span>
          <span className="value">{stats.bonds || 0}</span>
        </div>
        <div className="sim-stat">
          <span className="label">Avg F</span>
          <span className="value">{stats.avgF?.toFixed(2) || 0}</span>
        </div>
        <div className="sim-stat">
          <span className="label">Max q</span>
          <span className="value">{stats.maxQ?.toFixed(2) || 0}</span>
        </div>
        <div className="sim-stat">
          <span className="label">Avg P</span>
          <span className="value">{stats.avgP?.toFixed(3) || 0}</span>
        </div>
        <div className="sim-stat">
          <span className="label">Fusions</span>
          <span className="value" style={{ color: stats.fusions > 0 ? '#f0a' : '#0ff' }}>{stats.fusions || 0}</span>
        </div>
        {boundaryEnabled && (
          <div className="sim-stat">
            <span className="label">Boundary</span>
            <span className="value" style={{ color: '#6af' }}>{stats.boundary || 0}</span>
          </div>
        )}
        {graceEnabled && (
          <div className="sim-stat">
            <span className="label">Grace</span>
            <span className="value" style={{ color: stats.graceFlow > 0.1 ? '#0f8' : '#0ff' }}>{stats.graceFlow?.toFixed(2) || 0}</span>
          </div>
        )}
        {replicationEnabled && (
          <>
            <div className="sim-stat">
              <span className="label">Replications</span>
              <span className="value" style={{ color: stats.replications > 0 ? '#f80' : '#0ff' }}>{stats.replications || 0}</span>
            </div>
            <div className="sim-stat">
              <span className="label">Dormant</span>
              <span className="value" style={{ color: '#888' }}>{stats.dormant || 0}</span>
            </div>
          </>
        )}
      </div>

      {/* Formula Display */}
      <div className="sim-formula-overlay">
        <div className="formula-line">P = a·σ·(1+F)⁻¹·(1+H)⁻¹</div>
        <div className="formula-desc">Presence (local clock rate)</div>
      </div>

      {/* Controls Panel */}
      <motion.div
        className="sim-controls-panel"
        initial={{ x: 100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ delay: 0.3 }}
      >
        {/* Playback */}
        <div className="control-group">
          <button
            className={`sim-btn primary ${running ? 'active' : ''}`}
            onClick={() => setRunning(!running)}
          >
            {running ? '⏸ Pause' : '▶ Play'}
          </button>
          <button
            className="sim-btn"
            onClick={() => universeRef.current?.step_simulation()}
            disabled={running}
          >
            Step
          </button>
          <button
            className={`sim-btn ${drawMode ? 'active' : ''}`}
            onClick={() => setDrawMode(!drawMode)}
            style={{ marginTop: '8px' }}
          >
            {drawMode ? '✏ Drawing ON' : '✏ Draw Particles'}
          </button>
        </div>

        {/* Scenarios */}
        <div className="control-group">
          <h4>Scenario</h4>
          <div className="scenario-grid">
            {scenarios.map(s => (
              <button
                key={s.id}
                className={`sim-btn scenario ${scenario === s.id ? 'active' : ''}`}
                onClick={() => changeScenario(s.id)}
                title={s.desc}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        {/* Color Mode */}
        <div className="control-group">
          <h4>Color by</h4>
          <div className="color-mode-grid">
            {colorModes.map(m => (
              <button
                key={m.id}
                className={`sim-btn color-mode ${colorMode === m.id ? 'active' : ''}`}
                onClick={() => setColorMode(m.id)}
                title={m.desc}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Visual Options */}
        <div className="control-group">
          <h4>Visuals</h4>
          <label className="sim-checkbox">
            <input
              type="checkbox"
              checked={showTrails}
              onChange={(e) => setShowTrails(e.target.checked)}
            />
            <span>Trails</span>
          </label>
          <label className="sim-checkbox">
            <input
              type="checkbox"
              checked={showBonds}
              onChange={(e) => setShowBonds(e.target.checked)}
            />
            <span>Coherence Bonds</span>
          </label>
          <label className="sim-checkbox">
            <input
              type="checkbox"
              checked={showGlow}
              onChange={(e) => setShowGlow(e.target.checked)}
            />
            <span>Glow Effect</span>
          </label>
        </div>

        {/* DET Physics */}
        <div className="control-group">
          <h4>DET Physics</h4>
          <label className="sim-checkbox">
            <input
              type="checkbox"
              checked={boundaryEnabled}
              onChange={(e) => {
                setBoundaryEnabled(e.target.checked);
                universeRef.current?.setBoundaryEnabled(e.target.checked);
              }}
            />
            <span>Boundary Agents</span>
          </label>
          <label className="sim-checkbox">
            <input
              type="checkbox"
              checked={graceEnabled}
              onChange={(e) => {
                setGraceEnabled(e.target.checked);
                universeRef.current?.setGraceEnabled(e.target.checked);
              }}
            />
            <span>Grace Injection</span>
          </label>
          <label className="sim-checkbox">
            <input
              type="checkbox"
              checked={replicationEnabled}
              onChange={(e) => {
                setReplicationEnabled(e.target.checked);
                universeRef.current?.setReplicationEnabled(e.target.checked);
              }}
            />
            <span>Node Replication</span>
          </label>
        </div>

        {/* Particle Count */}
        <div className="control-group">
          <h4>Particles: {particleCount}</h4>
          <input
            type="range"
            min="50"
            max="500"
            value={particleCount}
            onChange={(e) => setParticleCount(parseInt(e.target.value))}
            className="sim-slider"
          />
        </div>

        {/* Legend */}
        <div className="control-group legend">
          <h4>DET Variables</h4>
          <div className="legend-item">
            <span className="dot" style={{ background: '#f0f' }}></span>
            <span><strong>a</strong> Agency [0,1]</span>
          </div>
          <div className="legend-item">
            <span className="dot" style={{ background: '#f55' }}></span>
            <span><strong>q</strong> Structure [0,1]</span>
          </div>
          <div className="legend-item">
            <span className="dot" style={{ background: '#ff0' }}></span>
            <span><strong>F</strong> Resource ≥0</span>
          </div>
          <div className="legend-item">
            <span className="dot" style={{ background: '#0ff' }}></span>
            <span><strong>P</strong> Presence</span>
          </div>
          <div className="legend-item">
            <span className="dot" style={{ background: 'linear-gradient(90deg, red, yellow, green, cyan, blue, magenta)' }}></span>
            <span><strong>θ</strong> Phase</span>
          </div>
          <div className="legend-item">
            <span className="line"></span>
            <span><strong>C</strong> Coherence bonds</span>
          </div>
        </div>
      </motion.div>

      {/* Info Tooltip */}
      <div className="sim-info-tooltip">
        <div className="info-title">Deep Existence Theory</div>
        <div className="info-text">
          Each particle is a "creature" with resource F, structural debt q,
          agency a, and phase θ. Gravity emerges from q, time dilation from F,
          and quantum correlations from coherence C between phase-aligned particles.
        </div>
        {drawMode && (
          <div className="info-text" style={{ marginTop: '8px', color: '#a78bfa' }}>
            Click and drag on the canvas to draw new particles!
            Drag direction sets initial velocity.
          </div>
        )}
      </div>
    </div>
  );
}

export default Simulator;
