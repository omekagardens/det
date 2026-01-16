import { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Stars, Html } from '@react-three/drei';
import * as THREE from 'three';
import { motion } from 'framer-motion';
import DETEngine from '../simulator/DETEngine';

// Particle visualization component
function Particles({ engine, displayField, threshold }) {
  const meshRef = useRef();
  const [positions, setPositions] = useState([]);
  const [colors, setColors] = useState([]);
  const [scales, setScales] = useState([]);

  const updateParticles = useCallback(() => {
    if (!engine) return;

    const data = engine.getFieldData(displayField, threshold);
    const newPositions = [];
    const newColors = [];
    const newScales = [];

    data.forEach(point => {
      newPositions.push(point.x, point.y, point.z);

      // Color based on field type
      let r, g, b;
      if (displayField === 'F') {
        // Resource: yellow-orange-red gradient
        r = Math.min(1, point.normalized * 2);
        g = Math.max(0, 1 - point.normalized);
        b = 0.1;
      } else if (displayField === 'q') {
        // Structure: blue-purple gradient
        r = point.normalized * 0.5;
        g = 0.2;
        b = 0.5 + point.normalized * 0.5;
      } else if (displayField === 'P') {
        // Presence: green-white gradient
        r = point.normalized;
        g = 0.5 + point.normalized * 0.5;
        b = point.normalized;
      } else if (displayField === 'Phi') {
        // Potential: cyan-blue for negative
        const absNorm = Math.abs(point.normalized);
        r = 0.1;
        g = absNorm * 0.8;
        b = 0.5 + absNorm * 0.5;
      }
      newColors.push(r, g, b);

      // Scale based on value
      newScales.push(0.3 + Math.abs(point.normalized) * 0.7);
    });

    setPositions(new Float32Array(newPositions));
    setColors(new Float32Array(newColors));
    setScales(new Float32Array(newScales.map(s => s)));
  }, [engine, displayField, threshold]);

  useFrame(() => {
    updateParticles();
  });

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    if (positions.length > 0) {
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    }
    return geo;
  }, [positions, colors]);

  if (positions.length === 0) return null;

  return (
    <points ref={meshRef} geometry={geometry}>
      <pointsMaterial
        size={0.5}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
        blending={THREE.AdditiveBlending}
      />
    </points>
  );
}

// Velocity field visualization
function VelocityField({ engine }) {
  const linesRef = useRef();
  const [lines, setLines] = useState([]);

  useFrame(() => {
    if (!engine) return;

    const N = engine.N;
    const newLines = [];
    const step = 4; // Sample every 4 cells

    for (let z = 0; z < N; z += step) {
      for (let y = 0; y < N; y += step) {
        for (let x = 0; x < N; x += step) {
          const i = engine.idx(x, y, z);
          const px = engine.pi_x[i];
          const py = engine.pi_y[i];
          const pz = engine.pi_z[i];
          const mag = Math.sqrt(px * px + py * py + pz * pz);

          if (mag > 0.01) {
            const scale = 2;
            newLines.push({
              start: [x - N / 2, y - N / 2, z - N / 2],
              end: [
                x - N / 2 + px * scale,
                y - N / 2 + py * scale,
                z - N / 2 + pz * scale
              ],
              mag
            });
          }
        }
      }
    }

    setLines(newLines);
  });

  return (
    <group ref={linesRef}>
      {lines.map((line, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([...line.start, ...line.end])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={new THREE.Color().setHSL(0.6 - line.mag * 0.3, 1, 0.5)} />
        </line>
      ))}
    </group>
  );
}

// Grid helper
function GridHelper() {
  return (
    <group>
      <gridHelper args={[32, 32, '#333', '#222']} rotation={[0, 0, 0]} />
      <gridHelper args={[32, 32, '#333', '#222']} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -16]} />
      <gridHelper args={[32, 32, '#333', '#222']} rotation={[0, 0, Math.PI / 2]} position={[-16, 0, 0]} />
    </group>
  );
}

// Stats overlay
function StatsOverlay({ stats }) {
  return (
    <div className="stats-overlay">
      <div className="stat-row">
        <span className="stat-name">Step:</span>
        <span className="stat-val">{stats.step}</span>
      </div>
      <div className="stat-row">
        <span className="stat-name">Time:</span>
        <span className="stat-val">{stats.time.toFixed(2)}</span>
      </div>
      <div className="stat-row">
        <span className="stat-name">Total F:</span>
        <span className="stat-val">{stats.totalF.toFixed(2)}</span>
      </div>
      <div className="stat-row">
        <span className="stat-name">Max F:</span>
        <span className="stat-val">{stats.maxF.toFixed(2)}</span>
      </div>
      <div className="stat-row">
        <span className="stat-name">Avg P:</span>
        <span className="stat-val">{stats.avgP.toFixed(4)}</span>
      </div>
      <div className="stat-row">
        <span className="stat-name">Max q:</span>
        <span className="stat-val">{stats.maxQ.toFixed(4)}</span>
      </div>
    </div>
  );
}

// Main scene
function Scene({ engine, displayField, threshold, showVelocity, showGrid }) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.5} />
      <Stars radius={100} depth={50} count={3000} factor={4} fade />

      {showGrid && <GridHelper />}

      <Particles engine={engine} displayField={displayField} threshold={threshold} />

      {showVelocity && <VelocityField engine={engine} />}

      <OrbitControls
        enablePan
        enableZoom
        enableRotate
        autoRotate
        autoRotateSpeed={0.5}
      />
    </>
  );
}

// Simulation runner
function SimulationRunner({ engine, running, speed }) {
  useFrame(() => {
    if (running && engine) {
      for (let i = 0; i < speed; i++) {
        engine.step_simulation();
      }
    }
  });
  return null;
}

function Simulator() {
  const [engine, setEngine] = useState(null);
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [displayField, setDisplayField] = useState('F');
  const [threshold, setThreshold] = useState(0.05);
  const [showVelocity, setShowVelocity] = useState(false);
  const [showGrid, setShowGrid] = useState(true);
  const [stats, setStats] = useState({
    step: 0, time: 0, totalF: 0, maxF: 0, avgF: 0, totalQ: 0, maxQ: 0, avgP: 1, minP: 1
  });
  const [scenario, setScenario] = useState('binary');

  // Initialize engine
  useEffect(() => {
    const newEngine = new DETEngine({ N: 32 });
    setEngine(newEngine);
  }, []);

  // Setup scenarios
  const setupScenario = useCallback((name) => {
    if (!engine) return;

    // Reset
    engine.initializeFields();

    const N = engine.N;
    const center = N / 2;

    switch (name) {
      case 'binary':
        // Two orbiting masses
        engine.addMass(center - 6, center, center, 5, 3);
        engine.addMass(center + 6, center, center, 5, 3);
        engine.addVelocity(center - 6, center, center, 0, 0.3, 0, 3);
        engine.addVelocity(center + 6, center, center, 0, -0.3, 0, 3);
        break;

      case 'collision':
        // Head-on collision
        engine.addMass(center - 10, center, center, 8, 4);
        engine.addMass(center + 10, center, center, 8, 4);
        engine.addVelocity(center - 10, center, center, 0.5, 0, 0, 4);
        engine.addVelocity(center + 10, center, center, -0.5, 0, 0, 4);
        break;

      case 'galaxy':
        // Spiral galaxy-like structure
        for (let i = 0; i < 8; i++) {
          const angle = (i / 8) * Math.PI * 2;
          const r = 8;
          const x = center + Math.cos(angle) * r;
          const y = center + Math.sin(angle) * r;
          engine.addMass(x, y, center, 3, 2);
          // Tangential velocity
          engine.addVelocity(x, y, center, -Math.sin(angle) * 0.2, Math.cos(angle) * 0.2, 0, 2);
        }
        // Central mass
        engine.addMass(center, center, center, 10, 4);
        break;

      case 'cluster':
        // Random cluster
        for (let i = 0; i < 15; i++) {
          const x = center + (Math.random() - 0.5) * 20;
          const y = center + (Math.random() - 0.5) * 20;
          const z = center + (Math.random() - 0.5) * 20;
          engine.addMass(x, y, z, 2 + Math.random() * 3, 2);
        }
        break;

      case 'expansion':
        // Big bang-like expansion
        engine.addMass(center, center, center, 50, 6);
        for (let z = -5; z <= 5; z++) {
          for (let y = -5; y <= 5; y++) {
            for (let x = -5; x <= 5; x++) {
              const r = Math.sqrt(x * x + y * y + z * z);
              if (r > 0 && r <= 5) {
                const i = engine.idx(center + x, center + y, center + z);
                engine.pi_x[i] = x * 0.3 / r;
                engine.pi_y[i] = y * 0.3 / r;
                engine.pi_z[i] = z * 0.3 / r;
              }
            }
          }
        }
        break;

      default:
        break;
    }

    setScenario(name);
  }, [engine]);

  // Initial setup
  useEffect(() => {
    if (engine) {
      setupScenario('binary');
    }
  }, [engine, setupScenario]);

  // Update stats
  useEffect(() => {
    const interval = setInterval(() => {
      if (engine) {
        setStats(engine.getStats());
      }
    }, 100);
    return () => clearInterval(interval);
  }, [engine]);

  return (
    <div className="simulator-page">
      <motion.header
        className="page-header compact"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1>3D Universe Simulator</h1>
        <p>Watch DET dynamics in real-time</p>
      </motion.header>

      <div className="simulator-layout">
        <div className="simulator-canvas-container">
          <Canvas camera={{ position: [30, 20, 30], fov: 60 }}>
            <Scene
              engine={engine}
              displayField={displayField}
              threshold={threshold}
              showVelocity={showVelocity}
              showGrid={showGrid}
            />
            <SimulationRunner engine={engine} running={running} speed={speed} />
          </Canvas>

          <StatsOverlay stats={stats} />

          <div className="playback-controls">
            <button
              className={`control-btn ${running ? 'active' : ''}`}
              onClick={() => setRunning(!running)}
            >
              {running ? '⏸ Pause' : '▶ Play'}
            </button>
            <button
              className="control-btn"
              onClick={() => engine && engine.step_simulation()}
              disabled={running}
            >
              ⏭ Step
            </button>
            <div className="speed-control">
              <label>Speed:</label>
              <input
                type="range"
                min="1"
                max="10"
                value={speed}
                onChange={(e) => setSpeed(parseInt(e.target.value))}
              />
              <span>{speed}x</span>
            </div>
          </div>
        </div>

        <div className="simulator-controls">
          <div className="control-section">
            <h3>Scenario</h3>
            <div className="scenario-buttons">
              {[
                { id: 'binary', label: 'Binary Orbit' },
                { id: 'collision', label: 'Collision' },
                { id: 'galaxy', label: 'Galaxy' },
                { id: 'cluster', label: 'Cluster' },
                { id: 'expansion', label: 'Expansion' },
              ].map(s => (
                <button
                  key={s.id}
                  className={`scenario-btn ${scenario === s.id ? 'active' : ''}`}
                  onClick={() => setupScenario(s.id)}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="control-section">
            <h3>Display Field</h3>
            <div className="field-buttons">
              {[
                { id: 'F', label: 'Resource (F)', color: '#f59e0b' },
                { id: 'q', label: 'Structure (q)', color: '#8b5cf6' },
                { id: 'P', label: 'Presence (P)', color: '#22c55e' },
                { id: 'Phi', label: 'Potential (Φ)', color: '#06b6d4' },
              ].map(f => (
                <button
                  key={f.id}
                  className={`field-btn ${displayField === f.id ? 'active' : ''}`}
                  style={{ '--field-color': f.color }}
                  onClick={() => setDisplayField(f.id)}
                >
                  {f.label}
                </button>
              ))}
            </div>
          </div>

          <div className="control-section">
            <h3>Visualization</h3>
            <div className="viz-options">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={showVelocity}
                  onChange={(e) => setShowVelocity(e.target.checked)}
                />
                Show Momentum Field
              </label>
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={showGrid}
                  onChange={(e) => setShowGrid(e.target.checked)}
                />
                Show Grid
              </label>
            </div>
            <div className="threshold-control">
              <label>Threshold: {threshold.toFixed(2)}</label>
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
            </div>
          </div>

          <div className="control-section">
            <h3>Field Legend</h3>
            <div className="legend">
              <div className="legend-item">
                <span className="legend-color" style={{ background: 'linear-gradient(to right, #fcd34d, #f97316, #ef4444)' }}></span>
                <span>Resource F: Low → High</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ background: 'linear-gradient(to right, #6366f1, #a855f7)' }}></span>
                <span>Structure q: Mass accumulation</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ background: 'linear-gradient(to right, #22c55e, #ffffff)' }}></span>
                <span>Presence P: Clock rate</span>
              </div>
              <div className="legend-item">
                <span className="legend-color" style={{ background: 'linear-gradient(to right, #0891b2, #06b6d4)' }}></span>
                <span>Potential Φ: Gravity wells</span>
              </div>
            </div>
          </div>

          <div className="control-section info">
            <h3>DET Dynamics</h3>
            <p>
              This simulator shows DET physics in action:
            </p>
            <ul>
              <li><strong>F (Resource)</strong>: Flows between cells, accumulates in gravity wells</li>
              <li><strong>q (Structure)</strong>: Increases from F loss, creates gravity</li>
              <li><strong>P (Presence)</strong>: Local clock rate, slows near mass</li>
              <li><strong>Φ (Potential)</strong>: Gravitational potential field</li>
            </ul>
            <p className="formula">
              P = a·σ·(1+F)⁻¹·(1+H)⁻¹
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Simulator;
