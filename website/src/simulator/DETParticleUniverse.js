/**
 * Deep Existence Theory (DET) v6.3 Particle Universe Engine
 * Beautiful particle-based simulation using strict DET mathematics
 *
 * Each particle is a "creature" with:
 * - F: Resource (stored energy)
 * - q: Structural debt (retained past, creates gravity)
 * - a: Agency (decision-making capacity)
 * - σ: Processing rate
 * - θ: Phase (for quantum-like behavior)
 * - π: Momentum vector
 * - C: Coherence with neighbors
 */

export class DETParticleUniverse {
  constructor(config = {}) {
    this.numParticles = config.numParticles || 200;
    this.width = config.width || 800;
    this.height = config.height || 800;
    this.DT = config.DT || 0.016;

    // DET Parameters (from unified schema)
    this.params = {
      // Gravity
      kappa: 0.5,           // Gravitational coupling
      mu_g: 0.3,            // Gravity mobility
      alpha_grav: 0.02,     // Screening parameter

      // Momentum
      alpha_pi: 0.12,       // Momentum charging
      lambda_pi: 0.005,     // Momentum decay (reduced for persistence)
      mu_pi: 2.5,           // Momentum mobility (increased for visibility)
      pi_max: 8.0,          // Maximum momentum (increased)
      beta_g: 3.0,          // Gravity-momentum coupling (increased)

      // Agency
      lambda_a: 30.0,       // Structural ceiling coupling
      beta_a: 0.2,          // Agency relaxation

      // Coherence
      C_init: 0.5,          // Initial coherence
      alpha_C: 0.04,        // Coherence growth
      lambda_C: 0.002,      // Coherence decay
      C_range: 80,          // Coherence interaction range

      // Structure
      alpha_q: 0.001,       // Q-locking rate

      // Floor repulsion
      eta_f: 2.0,           // Floor stiffness
      F_core: 0.3,          // Core repulsion distance factor

      // Phase
      omega_0: 0.1,         // Base phase evolution rate

      // Visuals
      trailLength: 20,
      maxBondDistance: 100,
    };

    Object.assign(this.params, config.params || {});

    this.particles = [];
    this.bonds = [];
    this.step = 0;
    this.time = 0;

    this.initializeParticles();
  }

  initializeParticles() {
    this.particles = [];
    const cx = this.width / 2;
    const cy = this.height / 2;

    for (let i = 0; i < this.numParticles; i++) {
      // Random position in a disc
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * Math.min(this.width, this.height) * 0.4;

      this.particles.push({
        id: i,
        // Position
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,

        // DET State Variables
        F: 0.5 + Math.random() * 0.5,      // Resource [0, ∞)
        q: Math.random() * 0.3,             // Structural debt [0, 1]
        a: 0.8 + Math.random() * 0.2,       // Agency [0, 1]
        sigma: 0.8 + Math.random() * 0.4,   // Processing rate
        theta: Math.random() * Math.PI * 2, // Phase

        // Momentum (larger initial values for visible movement)
        px: (Math.random() - 0.5) * 3,
        py: (Math.random() - 0.5) * 3,

        // Derived
        P: 1.0,  // Presence (computed)

        // Visual trail
        trail: [],

        // Coherence bonds (computed each frame)
        coherence: new Map(),
      });
    }
  }

  // DET Presence formula: P = a·σ·(1+F)⁻¹·(1+H)⁻¹
  computePresence(particle) {
    const { a, sigma, F } = particle;
    // H = coordination load (simplified as proportional to number of coherent bonds)
    const H = particle.coherence.size * 0.1;
    return (a * sigma) / (1 + F) / (1 + H);
  }

  // Agency ceiling: a_max = 1/(1 + λ_a·q²)
  computeAgencyCeiling(q) {
    return 1 / (1 + this.params.lambda_a * q * q);
  }

  // Gravitational potential from structural debt
  computeGravity(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const r2 = dx * dx + dy * dy;
    const r = Math.sqrt(r2) + 1;

    // DET: gravity sourced by structural debt q
    // Φ = -κ·q/r (attractive)
    const q_effective = p2.q - this.params.alpha_grav; // Baseline-referenced
    if (q_effective <= 0) return { fx: 0, fy: 0 };

    const strength = this.params.kappa * q_effective * p1.F / (r * r);

    return {
      fx: strength * dx / r,
      fy: strength * dy / r
    };
  }

  // Floor repulsion to prevent overlap
  computeFloorRepulsion(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const r = Math.sqrt(dx * dx + dy * dy) + 0.1;

    const minDist = (p1.F + p2.F) * 10 * this.params.F_core;
    if (r > minDist) return { fx: 0, fy: 0 };

    // DET floor repulsion: J^floor = η_f·(s_i + s_j)·(F_i - F_j)
    const overlap = (minDist - r) / minDist;
    const strength = this.params.eta_f * overlap * overlap * (p1.F + p2.F);

    return {
      fx: -strength * dx / r,
      fy: -strength * dy / r
    };
  }

  // Coherence between particles (quantum-like correlation)
  computeCoherence(p1, p2) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const r = Math.sqrt(dx * dx + dy * dy);

    if (r > this.params.C_range) return 0;

    // Coherence based on phase alignment and distance
    const phaseDiff = Math.abs(Math.cos(p1.theta - p2.theta));
    const distFactor = 1 - r / this.params.C_range;

    // Agency gate: g^(a) = √(a_i·a_j)
    const agencyGate = Math.sqrt(p1.a * p2.a);

    return phaseDiff * distFactor * agencyGate * this.params.C_init;
  }

  // Diffusive flux between particles (DET flow)
  computeDiffusiveFlow(p1, p2, C) {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    const r = Math.sqrt(dx * dx + dy * dy) + 0.1;

    // Agency gate
    const g_a = Math.sqrt(p1.a * p2.a);

    // Quantum-classical interpolation via coherence
    // J^diff = g^a·[√C·Im(ψ*ψ) + (1-√C)·(F_i - F_j)]
    const sqrtC = Math.sqrt(C);
    const phaseTerm = Math.sin(p1.theta - p2.theta); // Im(ψ*ψ)
    const pressureTerm = p1.F - p2.F;

    const flowMag = g_a * (sqrtC * phaseTerm * 0.1 + (1 - sqrtC) * pressureTerm * 0.05);

    return {
      fx: flowMag * dx / r,
      fy: flowMag * dy / r,
      flowMag
    };
  }

  step_simulation() {
    const dt = this.DT;
    const particles = this.particles;
    const n = particles.length;

    // Reset coherence bonds
    for (const p of particles) {
      p.coherence.clear();
    }

    // Compute all pairwise interactions
    const forces = particles.map(() => ({ fx: 0, fy: 0 }));
    const flowIn = particles.map(() => 0);

    for (let i = 0; i < n; i++) {
      const p1 = particles[i];

      for (let j = i + 1; j < n; j++) {
        const p2 = particles[j];

        // Compute coherence
        const C = this.computeCoherence(p1, p2);
        if (C > 0.1) {
          p1.coherence.set(j, C);
          p2.coherence.set(i, C);
        }

        // Gravity
        const grav1 = this.computeGravity(p1, p2);
        const grav2 = this.computeGravity(p2, p1);
        forces[i].fx += grav1.fx;
        forces[i].fy += grav1.fy;
        forces[j].fx += grav2.fx;
        forces[j].fy += grav2.fy;

        // Floor repulsion
        const floor = this.computeFloorRepulsion(p1, p2);
        forces[i].fx += floor.fx;
        forces[i].fy += floor.fy;
        forces[j].fx -= floor.fx;
        forces[j].fy -= floor.fy;

        // Diffusive flow
        if (C > 0) {
          const flow = this.computeDiffusiveFlow(p1, p2, C);
          flowIn[i] -= flow.flowMag;
          flowIn[j] += flow.flowMag;
        }
      }
    }

    // Update each particle
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      const f = forces[i];

      // Compute presence (local clock rate)
      p.P = this.computePresence(p);
      const localDt = dt * p.P;

      // Update momentum with gravity coupling
      // π⁺ = (1 - λ_π·Δτ)·π + β_g·g·Δτ
      p.px = (1 - this.params.lambda_pi * localDt) * p.px + this.params.beta_g * f.fx * localDt;
      p.py = (1 - this.params.lambda_pi * localDt) * p.py + this.params.beta_g * f.fy * localDt;

      // Clamp momentum
      const pMag = Math.sqrt(p.px * p.px + p.py * p.py);
      if (pMag > this.params.pi_max) {
        p.px *= this.params.pi_max / pMag;
        p.py *= this.params.pi_max / pMag;
      }

      // Update position from momentum
      p.x += p.px * this.params.mu_pi * localDt * 10;
      p.y += p.py * this.params.mu_pi * localDt * 10;

      // Boundary wrapping
      if (p.x < 0) p.x += this.width;
      if (p.x > this.width) p.x -= this.width;
      if (p.y < 0) p.y += this.height;
      if (p.y > this.height) p.y -= this.height;

      // Update resource F from flow
      p.F = Math.max(0.1, p.F + flowIn[i] * localDt);

      // Update structural debt q (q-locking from resource loss)
      if (flowIn[i] < 0) {
        p.q = Math.min(1, p.q + this.params.alpha_q * Math.abs(flowIn[i]));
      }

      // Update agency toward ceiling
      const a_max = this.computeAgencyCeiling(p.q);
      p.a = p.a + this.params.beta_a * (a_max - p.a) * localDt;
      p.a = Math.max(0, Math.min(a_max, p.a));

      // Update phase
      p.theta += this.params.omega_0 * localDt;
      if (p.theta > Math.PI * 2) p.theta -= Math.PI * 2;

      // Update trail
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > this.params.trailLength) {
        p.trail.shift();
      }
    }

    this.step++;
    this.time += dt;
  }

  // Get bonds for visualization
  getBonds() {
    const bonds = [];
    const seen = new Set();

    for (const p of this.particles) {
      for (const [j, C] of p.coherence) {
        const key = p.id < j ? `${p.id}-${j}` : `${j}-${p.id}`;
        if (!seen.has(key) && C > 0.15) {
          seen.add(key);
          bonds.push({
            p1: p,
            p2: this.particles[j],
            C
          });
        }
      }
    }

    return bonds;
  }

  // Get statistics
  getStats() {
    let totalF = 0, totalQ = 0, totalP = 0;
    let maxF = 0, maxQ = 0, minP = Infinity;

    for (const p of this.particles) {
      totalF += p.F;
      totalQ += p.q;
      totalP += p.P;
      maxF = Math.max(maxF, p.F);
      maxQ = Math.max(maxQ, p.q);
      minP = Math.min(minP, p.P);
    }

    const n = this.particles.length;
    return {
      step: this.step,
      time: this.time,
      particles: n,
      totalF,
      avgF: totalF / n,
      maxF,
      avgQ: totalQ / n,
      maxQ,
      avgP: totalP / n,
      minP,
      bonds: this.getBonds().length
    };
  }

  // Add a new particle at specified position with optional velocity
  addParticle(x, y, vx = 0, vy = 0) {
    const newId = this.particles.length;
    this.particles.push({
      id: newId,
      x: x,
      y: y,
      F: 0.5 + Math.random() * 0.5,
      q: Math.random() * 0.3,
      a: 0.8 + Math.random() * 0.2,
      sigma: 0.8 + Math.random() * 0.4,
      theta: Math.random() * Math.PI * 2,
      px: vx,
      py: vy,
      P: 1.0,
      trail: [],
      coherence: new Map(),
    });
    return this.particles[newId];
  }

  // Scenario setups
  setupScenario(name) {
    const cx = this.width / 2;
    const cy = this.height / 2;

    switch (name) {
      case 'orbiting':
        this.initializeParticles();
        // Create a central massive particle
        this.particles[0].x = cx;
        this.particles[0].y = cy;
        this.particles[0].q = 0.9;
        this.particles[0].F = 2.0;
        this.particles[0].px = 0;
        this.particles[0].py = 0;

        // Others orbit around it
        for (let i = 1; i < this.particles.length; i++) {
          const angle = (i / this.particles.length) * Math.PI * 2;
          const r = 100 + Math.random() * 200;
          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.1 + Math.random() * 0.2;
          // Tangential velocity for orbit
          const v = 1.5 / Math.sqrt(r / 100);
          this.particles[i].px = -Math.sin(angle) * v;
          this.particles[i].py = Math.cos(angle) * v;
        }
        break;

      case 'collision':
        this.initializeParticles();
        const half = Math.floor(this.numParticles / 2);
        // Two clusters approaching each other
        for (let i = 0; i < half; i++) {
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 80;
          this.particles[i].x = cx - 200 + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].px = 2;
          this.particles[i].py = 0;
          this.particles[i].q = 0.3;
        }
        for (let i = half; i < this.numParticles; i++) {
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 80;
          this.particles[i].x = cx + 200 + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].px = -2;
          this.particles[i].py = 0;
          this.particles[i].q = 0.3;
        }
        break;

      case 'expansion':
        this.initializeParticles();
        // Big bang - everything from center with outward velocity
        for (let i = 0; i < this.numParticles; i++) {
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 50;
          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.5 + Math.random() * 0.3;
          this.particles[i].F = 1 + Math.random();
          // Outward velocity proportional to distance
          const v = 0.5 + r / 30;
          this.particles[i].px = Math.cos(angle) * v;
          this.particles[i].py = Math.sin(angle) * v;
        }
        break;

      case 'quantum':
        this.initializeParticles();
        // High coherence scenario - synchronized phases
        for (let i = 0; i < this.numParticles; i++) {
          this.particles[i].theta = (i / this.numParticles) * Math.PI * 2;
          this.particles[i].a = 0.95;
          this.particles[i].q = 0.1;
          this.particles[i].px *= 0.3;
          this.particles[i].py *= 0.3;
        }
        this.params.C_init = 0.8;
        this.params.C_range = 150;
        break;

      case 'galaxy':
        this.initializeParticles();
        // Spiral galaxy structure
        this.particles[0].x = cx;
        this.particles[0].y = cy;
        this.particles[0].q = 0.95;
        this.particles[0].F = 3.0;
        this.particles[0].px = 0;
        this.particles[0].py = 0;

        for (let i = 1; i < this.numParticles; i++) {
          // Spiral arm
          const arm = i % 2;
          const t = (i / this.numParticles) * 4;
          const r = 50 + t * 80;
          const angle = t * Math.PI + arm * Math.PI + (Math.random() - 0.5) * 0.5;

          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.2 + Math.random() * 0.1;

          // Circular orbit velocity
          const v = 2.0 / Math.sqrt(r / 50);
          this.particles[i].px = -Math.sin(angle) * v;
          this.particles[i].py = Math.cos(angle) * v;
        }
        break;

      default:
        this.initializeParticles();
    }

    // Reset trails
    for (const p of this.particles) {
      p.trail = [];
    }

    this.step = 0;
    this.time = 0;
  }
}

export default DETParticleUniverse;
