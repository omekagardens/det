/**
 * Deep Existence Theory (DET) v6.3 Particle Universe Engine
 * Proper implementation of DET physics with bonding, collision, and fusion
 *
 * Each particle ("creature") has:
 * - F: Resource (stored energy/mass)
 * - q: Structural debt (retained past, creates gravity, enables binding)
 * - a: Agency (decision-making capacity)
 * - σ: Processing rate
 * - θ: Phase (for quantum coherence)
 * - π: Momentum vector
 * - L: Angular momentum (for orbital dynamics)
 *
 * Bonds between particles have:
 * - C: Coherence (quantum correlation strength)
 * - π_bond: Bond momentum (memory of flow)
 */

export class DETParticleUniverse {
  constructor(config = {}) {
    this.numParticles = config.numParticles || 200;
    this.width = config.width || 800;
    this.height = config.height || 800;
    this.DT = config.DT || 0.02;

    // DET Parameters (from unified schema v6.3)
    this.params = {
      // Gravity
      kappa: 5.0,           // Gravitational coupling (κ)
      mu_g: 2.0,            // Gravity mobility
      alpha_grav: 0.02,     // Screening parameter
      beta_g: 10.0,         // Gravity-momentum coupling (5 × μ_g)

      // Momentum
      alpha_pi: 0.12,       // Momentum charging gain
      lambda_pi: 0.008,     // Momentum decay rate
      mu_pi: 0.35,          // Momentum mobility
      pi_max: 3.0,          // Maximum momentum

      // Angular Momentum (for orbital capture)
      alpha_L: 0.06,        // Angular momentum charging
      lambda_L: 0.005,      // Angular momentum decay
      mu_L: 0.18,           // Rotational mobility
      L_max: 5.0,           // Maximum angular momentum

      // Agency
      lambda_a: 30.0,       // Structural ceiling coupling
      beta_a: 0.2,          // Agency relaxation rate

      // Coherence (quantum bonds)
      C_init: 0.15,         // Initial coherence for new bonds
      C_min: 0.01,          // Minimum coherence
      alpha_C: 0.04,        // Coherence growth rate (from flow)
      lambda_C: 0.002,      // Coherence decay rate

      // Structure (q-locking for binding)
      alpha_q: 0.012,       // Q-locking rate (binding during collision)
      q_bind_threshold: 0.3, // q threshold for stable binding

      // Floor repulsion (prevent infinite compression)
      eta_f: 0.12,          // Floor stiffness
      F_core: 5.0,          // Core onset density
      floor_power: 2.0,     // Floor exponent

      // Phase
      omega_0: 0.1,         // Base phase evolution rate

      // Fusion
      fusion_distance: 15,  // Distance for potential fusion
      fusion_C_threshold: 0.6, // Coherence needed for fusion

      // Visuals
      trailLength: 20,
      bond_display_threshold: 0.05, // Show bonds above this C
      interaction_range: 150, // Max interaction distance
    };

    Object.assign(this.params, config.params || {});

    this.particles = [];
    this.bonds = new Map(); // Persistent bond states: "i-j" -> {C, pi, L}
    this.step = 0;
    this.time = 0;

    this.initializeParticles();
  }

  initializeParticles() {
    this.particles = [];
    this.bonds.clear();
    const cx = this.width / 2;
    const cy = this.height / 2;

    for (let i = 0; i < this.numParticles; i++) {
      const angle = Math.random() * Math.PI * 2;
      const r = Math.random() * Math.min(this.width, this.height) * 0.35;

      this.particles.push({
        id: i,
        x: cx + Math.cos(angle) * r,
        y: cy + Math.sin(angle) * r,
        F: 0.5 + Math.random() * 1.5,     // Resource [0.5, 2]
        q: Math.random() * 0.4,            // Structural debt [0, 0.4]
        a: 0.7 + Math.random() * 0.3,      // Agency [0.7, 1]
        sigma: 0.8 + Math.random() * 0.4,  // Processing rate
        theta: Math.random() * Math.PI * 2, // Phase
        px: (Math.random() - 0.5) * 2,     // Momentum x
        py: (Math.random() - 0.5) * 2,     // Momentum y
        L: 0,                               // Angular momentum
        P: 1.0,                             // Presence (computed)
        trail: [],
        alive: true,
      });
    }
  }

  // Bond key helper
  bondKey(i, j) {
    return i < j ? `${i}-${j}` : `${j}-${i}`;
  }

  // Get or create bond state
  getBond(i, j) {
    const key = this.bondKey(i, j);
    if (!this.bonds.has(key)) {
      this.bonds.set(key, {
        C: 0,      // Coherence
        pi: 0,     // Bond momentum
        L: 0,      // Bond angular momentum
      });
    }
    return this.bonds.get(key);
  }

  // DET Presence formula: P = a·σ·(1+F)⁻¹·(1+H)⁻¹
  computePresence(particle, numBonds) {
    const { a, sigma, F } = particle;
    const H = numBonds * 0.1; // Coordination load
    return (a * sigma) / (1 + F) / (1 + H);
  }

  // Agency ceiling: a_max = 1/(1 + λ_a·q²)
  computeAgencyCeiling(q) {
    return 1 / (1 + this.params.lambda_a * q * q);
  }

  // Gravitational force between particles (sourced by q)
  computeGravity(p1, p2, r, dx, dy) {
    // DET: gravity sourced by structural debt q
    // Attractive force toward higher-q particles
    const q_source = p2.q - this.params.alpha_grav;
    if (q_source <= 0) return { fx: 0, fy: 0 };

    // Φ = -κ·q/r → F = -∇Φ = -κ·q/r²·r̂
    const strength = this.params.kappa * q_source * p1.F / (r * r + 10);

    return {
      fx: strength * dx / r,
      fy: strength * dy / r
    };
  }

  // Floor repulsion to prevent overlap
  computeFloorRepulsion(p1, p2, r, dx, dy) {
    const minDist = Math.sqrt(p1.F + p2.F) * 8;
    if (r > minDist) return { fx: 0, fy: 0 };

    // DET floor: J^floor = η_f·(s_i + s_j)·(F_i - F_j)
    const overlap = Math.pow((minDist - r) / minDist, this.params.floor_power);
    const strength = this.params.eta_f * overlap * (p1.F + p2.F) * 50;

    return {
      fx: -strength * dx / (r + 0.1),
      fy: -strength * dy / (r + 0.1)
    };
  }

  // Compute diffusive flow (DET quantum-classical interpolation)
  computeDiffusiveFlow(p1, p2, bond, r) {
    const g_a = Math.sqrt(p1.a * p2.a); // Agency gate
    const sqrtC = Math.sqrt(bond.C);

    // J^diff = g^a·[√C·Im(ψ*ψ) + (1-√C)·(F_i - F_j)]
    const phaseTerm = Math.sin(p1.theta - p2.theta);
    const pressureTerm = p1.F - p2.F;

    return g_a * (sqrtC * phaseTerm * 0.5 + (1 - sqrtC) * pressureTerm * 0.1);
  }

  step_simulation() {
    const dt = this.DT;
    const particles = this.particles.filter(p => p.alive);
    const n = particles.length;

    if (n === 0) return;

    // Track bond counts for presence calculation
    const bondCounts = new Array(n).fill(0);
    const forces = particles.map(() => ({ fx: 0, fy: 0, torque: 0 }));
    const flowIn = particles.map(() => 0);
    const qLockIn = particles.map(() => 0);
    const activeBonds = new Set();

    // Pairwise interactions
    for (let i = 0; i < n; i++) {
      const p1 = particles[i];

      for (let j = i + 1; j < n; j++) {
        const p2 = particles[j];

        const dx = p2.x - p1.x;
        const dy = p2.y - p1.y;
        const r = Math.sqrt(dx * dx + dy * dy);

        if (r > this.params.interaction_range) continue;

        const bond = this.getBond(p1.id, p2.id);
        activeBonds.add(this.bondKey(p1.id, p2.id));

        // Update coherence (DET coherence dynamics)
        // Phase alignment factor
        const phaseAlign = 0.5 + 0.5 * Math.cos(p1.theta - p2.theta);
        const distFactor = Math.max(0, 1 - r / this.params.interaction_range);
        const agencyGate = Math.sqrt(p1.a * p2.a);

        // Coherence growth from proximity and phase alignment
        const C_growth = this.params.alpha_C * phaseAlign * distFactor * agencyGate * dt;
        // Coherence decay
        const C_decay = this.params.lambda_C * bond.C * dt;

        // Update coherence
        bond.C = Math.min(1, Math.max(this.params.C_min, bond.C + C_growth - C_decay));

        // Count significant bonds for presence
        if (bond.C > this.params.bond_display_threshold) {
          bondCounts[i]++;
          bondCounts[j]++;
        }

        // Compute diffusive flow
        const J_diff = this.computeDiffusiveFlow(p1, p2, bond, r);

        // Update bond momentum: π⁺ = (1-λΔτ)π + αJ·Δτ + β_g·g·Δτ
        const avgDt = dt * 0.5 * (p1.P + p2.P);
        bond.pi = (1 - this.params.lambda_pi * avgDt) * bond.pi
                  + this.params.alpha_pi * J_diff * avgDt;

        // Clamp bond momentum
        bond.pi = Math.max(-this.params.pi_max, Math.min(this.params.pi_max, bond.pi));

        // Momentum-driven drift flux
        const J_mom = this.params.mu_pi * bond.pi * (p1.F + p2.F) / 2;

        // Total flow
        const J_total = J_diff + J_mom;
        flowIn[i] -= J_total;
        flowIn[j] += J_total;

        // Gravity
        const grav1 = this.computeGravity(p1, p2, r, dx, dy);
        const grav2 = this.computeGravity(p2, p1, r, -dx, -dy);
        forces[i].fx += grav1.fx;
        forces[i].fy += grav1.fy;
        forces[j].fx += grav2.fx;
        forces[j].fy += grav2.fy;

        // Floor repulsion
        const floor = this.computeFloorRepulsion(p1, p2, r, dx, dy);
        forces[i].fx += floor.fx;
        forces[i].fy += floor.fy;
        forces[j].fx -= floor.fx;
        forces[j].fy -= floor.fy;

        // Q-locking during close approach (binding mechanism)
        const collisionDist = Math.sqrt(p1.F + p2.F) * 6;
        if (r < collisionDist && bond.C > 0.1) {
          // Strong interaction = structural debt increases (binding)
          const qLock = this.params.alpha_q * bond.C * (collisionDist - r) / collisionDist;
          qLockIn[i] += qLock;
          qLockIn[j] += qLock;
        }

        // Angular momentum for orbital dynamics
        // Cross product r × J gives torque
        const cross = dx * p1.py - dy * p1.px;
        const torque = this.params.alpha_L * cross * bond.C * 0.01;
        bond.L = (1 - this.params.lambda_L * avgDt) * bond.L + torque * avgDt;
        bond.L = Math.max(-this.params.L_max, Math.min(this.params.L_max, bond.L));

        // Rotational influence on velocity
        if (Math.abs(bond.L) > 0.01 && r > 10) {
          const rotStrength = this.params.mu_L * bond.L / (r + 10);
          forces[i].fx += -dy * rotStrength * 0.5;
          forces[i].fy += dx * rotStrength * 0.5;
          forces[j].fx += dy * rotStrength * 0.5;
          forces[j].fy += -dx * rotStrength * 0.5;
        }

        // Fusion check: high coherence + very close = merge
        if (r < this.params.fusion_distance && bond.C > this.params.fusion_C_threshold) {
          this.fuseParticles(p1, p2, i, j, particles);
        }
      }
    }

    // Update each particle
    for (let i = 0; i < n; i++) {
      const p = particles[i];
      if (!p.alive) continue;

      const f = forces[i];

      // Compute presence (local clock rate)
      p.P = this.computePresence(p, bondCounts[i]);
      const localDt = dt * p.P;

      // Update momentum with forces
      p.px = (1 - this.params.lambda_pi * localDt) * p.px + this.params.beta_g * f.fx * localDt;
      p.py = (1 - this.params.lambda_pi * localDt) * p.py + this.params.beta_g * f.fy * localDt;

      // Clamp momentum
      const pMag = Math.sqrt(p.px * p.px + p.py * p.py);
      if (pMag > this.params.pi_max * 3) {
        p.px *= this.params.pi_max * 3 / pMag;
        p.py *= this.params.pi_max * 3 / pMag;
      }

      // Update position
      p.x += p.px * this.params.mu_pi * localDt * 30;
      p.y += p.py * this.params.mu_pi * localDt * 30;

      // Boundary wrapping
      if (p.x < 0) p.x += this.width;
      if (p.x > this.width) p.x -= this.width;
      if (p.y < 0) p.y += this.height;
      if (p.y > this.height) p.y -= this.height;

      // Update resource F
      p.F = Math.max(0.1, p.F + flowIn[i] * localDt);

      // Update structural debt q (q-locking)
      p.q = Math.min(1, p.q + qLockIn[i] * localDt);

      // Update agency toward ceiling
      const a_max = this.computeAgencyCeiling(p.q);
      p.a = p.a + this.params.beta_a * (a_max - p.a) * localDt;
      p.a = Math.max(0.01, Math.min(a_max, p.a));

      // Update phase
      p.theta = (p.theta + this.params.omega_0 * localDt) % (Math.PI * 2);

      // Update trail
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > this.params.trailLength) {
        p.trail.shift();
      }
    }

    // Clean up bonds for dead particles or distant pairs
    for (const key of this.bonds.keys()) {
      if (!activeBonds.has(key)) {
        const bond = this.bonds.get(key);
        bond.C *= 0.95; // Decay inactive bonds
        if (bond.C < this.params.C_min) {
          this.bonds.delete(key);
        }
      }
    }

    this.step++;
    this.time += dt;
  }

  // Fuse two particles into one
  fuseParticles(p1, p2, i1, i2, particles) {
    // Conservation of mass/resource
    const totalF = p1.F + p2.F;
    const totalQ = p1.q + p2.q;

    // Momentum conservation
    const totalPx = p1.px * p1.F + p2.px * p2.F;
    const totalPy = p1.py * p1.F + p2.py * p2.F;

    // Merge into p1 (larger F survives)
    const survivor = p1.F >= p2.F ? p1 : p2;
    const absorbed = p1.F >= p2.F ? p2 : p1;

    survivor.F = totalF;
    survivor.q = Math.min(1, (totalQ / 2) + 0.1); // Fusion increases structure
    survivor.px = totalPx / totalF;
    survivor.py = totalPy / totalF;
    survivor.x = (p1.x * p1.F + p2.x * p2.F) / totalF;
    survivor.y = (p1.y * p1.F + p2.y * p2.F) / totalF;

    // Mark absorbed particle as dead
    absorbed.alive = false;
    absorbed.F = 0;

    // Clean up bonds involving absorbed particle
    const absorbedId = absorbed.id;
    for (const key of this.bonds.keys()) {
      if (key.includes(`${absorbedId}-`) || key.includes(`-${absorbedId}`)) {
        this.bonds.delete(key);
      }
    }
  }

  // Get bonds for visualization
  getBonds() {
    const bonds = [];
    for (const [key, bond] of this.bonds) {
      if (bond.C > this.params.bond_display_threshold) {
        const [i, j] = key.split('-').map(Number);
        const p1 = this.particles.find(p => p.id === i && p.alive);
        const p2 = this.particles.find(p => p.id === j && p.alive);
        if (p1 && p2) {
          bonds.push({ p1, p2, C: bond.C, L: bond.L });
        }
      }
    }
    return bonds;
  }

  // Get statistics
  getStats() {
    const alive = this.particles.filter(p => p.alive);
    let totalF = 0, totalQ = 0, totalP = 0;
    let maxF = 0, maxQ = 0;

    for (const p of alive) {
      totalF += p.F;
      totalQ += p.q;
      totalP += p.P;
      maxF = Math.max(maxF, p.F);
      maxQ = Math.max(maxQ, p.q);
    }

    const n = alive.length;
    return {
      step: this.step,
      time: this.time,
      particles: n,
      totalF,
      avgF: n > 0 ? totalF / n : 0,
      maxF,
      avgQ: n > 0 ? totalQ / n : 0,
      maxQ,
      avgP: n > 0 ? totalP / n : 0,
      bonds: this.getBonds().length,
      fusions: this.numParticles - n
    };
  }

  // Add a new particle
  addParticle(x, y, vx = 0, vy = 0) {
    const newId = this.particles.length;
    const p = {
      id: newId,
      x, y,
      F: 0.5 + Math.random() * 0.5,
      q: Math.random() * 0.2,
      a: 0.8 + Math.random() * 0.2,
      sigma: 0.8 + Math.random() * 0.4,
      theta: Math.random() * Math.PI * 2,
      px: vx,
      py: vy,
      L: 0,
      P: 1.0,
      trail: [],
      alive: true,
    };
    this.particles.push(p);
    return p;
  }

  // Scenario setups
  setupScenario(name) {
    const cx = this.width / 2;
    const cy = this.height / 2;

    // Reset
    this.bonds.clear();

    switch (name) {
      case 'orbiting':
        this.initializeParticles();
        // Central massive particle
        this.particles[0].x = cx;
        this.particles[0].y = cy;
        this.particles[0].q = 0.9;
        this.particles[0].F = 5.0;
        this.particles[0].px = 0;
        this.particles[0].py = 0;

        for (let i = 1; i < this.particles.length; i++) {
          const angle = (i / this.particles.length) * Math.PI * 2;
          const r = 80 + Math.random() * 150;
          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.15 + Math.random() * 0.15;
          this.particles[i].theta = angle; // Phase aligned with position
          const v = 2.5 / Math.sqrt(r / 80);
          this.particles[i].px = -Math.sin(angle) * v;
          this.particles[i].py = Math.cos(angle) * v;
        }
        break;

      case 'collision':
        this.initializeParticles();
        const half = Math.floor(this.numParticles / 2);
        for (let i = 0; i < half; i++) {
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 60;
          this.particles[i].x = cx - 180 + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].px = 3;
          this.particles[i].py = (Math.random() - 0.5) * 0.5;
          this.particles[i].q = 0.35;
          this.particles[i].theta = 0; // Same phase for cluster coherence
        }
        for (let i = half; i < this.numParticles; i++) {
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 60;
          this.particles[i].x = cx + 180 + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].px = -3;
          this.particles[i].py = (Math.random() - 0.5) * 0.5;
          this.particles[i].q = 0.35;
          this.particles[i].theta = Math.PI; // Opposite phase
        }
        break;

      case 'expansion':
        this.initializeParticles();
        for (let i = 0; i < this.numParticles; i++) {
          const angle = Math.random() * Math.PI * 2;
          const r = Math.random() * 40;
          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.5 + Math.random() * 0.4;
          this.particles[i].F = 1.5 + Math.random();
          this.particles[i].theta = angle;
          const v = 1.0 + r / 20;
          this.particles[i].px = Math.cos(angle) * v;
          this.particles[i].py = Math.sin(angle) * v;
        }
        break;

      case 'quantum':
        this.initializeParticles();
        // High coherence - all phases synchronized
        for (let i = 0; i < this.numParticles; i++) {
          this.particles[i].theta = 0; // All same phase = max coherence
          this.particles[i].a = 0.95;
          this.particles[i].q = 0.1;
          this.particles[i].px *= 0.2;
          this.particles[i].py *= 0.2;
        }
        this.params.alpha_C = 0.1; // Boost coherence growth
        break;

      case 'galaxy':
        this.initializeParticles();
        this.particles[0].x = cx;
        this.particles[0].y = cy;
        this.particles[0].q = 0.95;
        this.particles[0].F = 8.0;
        this.particles[0].px = 0;
        this.particles[0].py = 0;

        for (let i = 1; i < this.numParticles; i++) {
          const arm = i % 2;
          const t = (i / this.numParticles) * 3;
          const r = 60 + t * 100;
          const angle = t * Math.PI * 0.8 + arm * Math.PI + (Math.random() - 0.5) * 0.3;

          this.particles[i].x = cx + Math.cos(angle) * r;
          this.particles[i].y = cy + Math.sin(angle) * r;
          this.particles[i].q = 0.2 + Math.random() * 0.15;
          this.particles[i].theta = angle;

          const v = 2.5 / Math.sqrt(r / 60);
          this.particles[i].px = -Math.sin(angle) * v;
          this.particles[i].py = Math.cos(angle) * v;
        }
        break;

      default:
        this.initializeParticles();
    }

    for (const p of this.particles) {
      p.trail = [];
      p.alive = true;
    }

    this.step = 0;
    this.time = 0;
  }
}

export default DETParticleUniverse;
