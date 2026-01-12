import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Settings, Activity, Zap, Wind, TrendingUp } from 'lucide-react';

// --- MATH UTILITIES FOR 1D PHYSICS ---

const N = 400; // Grid size

// Helper: Modulo that handles negative numbers correctly
const mod = (n: number, m: number) => ((n % m) + m) % m;

// Helper: 1D Periodic Distance
const periodicDistance = (x1: number, x2: number, size: number) => {
  let dx = x1 - x2;
  if (dx > size / 2) dx -= size;
  if (dx < -size / 2) dx += size;
  return dx;
};

// --- SIMULATION ENGINE (PORTED FROM PYTHON) ---

class DETEngine {
  // Parameters
  params = {
    DT: 0.02,
    F_VAC: 0.01,
    R: 10,
    nu: 0.1,
    omega_0: 0.0,
    momentum: {
      enabled: true,
      alpha_pi: 0.1,
      lambda_pi: 0.01,
      mu_pi: 0.3,
      pi_max: 3.0
    },
    coherence: {
      alpha_c: 0.05,
      gamma_c: 2.0,
      lambda_c: 0.002,
      c_min: 0.05
    },
    q_locking: {
      enabled: true,
      alpha_q: 0.02
    },
    outflow_limit: 0.25
  };

  // State Arrays (Float32 for performance)
  F: Float32Array;
  q: Float32Array;
  theta: Float32Array;
  a: Float32Array;
  C_right: Float32Array;
  C_left: Float32Array;
  sigma: Float32Array;
  pi_right: Float32Array;
  P: Float32Array;
  Delta_tau: Float32Array;
  
  // Metrics
  time: number = 0;
  stepCount: number = 0;

  constructor() {
    this.F = new Float32Array(N).fill(this.params.F_VAC);
    this.q = new Float32Array(N).fill(0);
    this.theta = new Float32Array(N).fill(0);
    this.a = new Float32Array(N).fill(1);
    this.C_right = new Float32Array(N).fill(this.params.coherence.c_min);
    this.C_left = new Float32Array(N).fill(this.params.coherence.c_min);
    this.sigma = new Float32Array(N).fill(1);
    this.pi_right = new Float32Array(N).fill(0);
    this.P = new Float32Array(N).fill(1);
    this.Delta_tau = new Float32Array(N).fill(this.params.DT);
  }

  // --- INITIALIZATION ---

  reset() {
    this.time = 0;
    this.stepCount = 0;
    this.F.fill(this.params.F_VAC);
    this.q.fill(0);
    this.theta.fill(0);
    this.pi_right.fill(0);
    this.C_right.fill(this.params.coherence.c_min);
    this.C_left.fill(this.params.coherence.c_min);
  }

  addGaussian(center: number, amp: number, width: number, cBoost: number = 0.7) {
    for (let i = 0; i < N; i++) {
      const dx = periodicDistance(i, center, N);
      const envelope = Math.exp(-0.5 * (dx / width) ** 2);
      this.F[i] += amp * envelope;
      this.C_right[i] = Math.min(1.0, this.C_right[i] + cBoost * envelope);
      this.C_left[i] = Math.min(1.0, this.C_left[i] + cBoost * envelope);
    }
  }

  addMomentumImpulse(center: number, momentum: number, width: number) {
    for (let i = 0; i < N; i++) {
      const dx = periodicDistance(i, center, N);
      const envelope = Math.exp(-0.5 * (dx / width) ** 2);
      this.pi_right[i] += momentum * envelope;
      // Clip
      if (this.pi_right[i] > this.params.momentum.pi_max) this.pi_right[i] = this.params.momentum.pi_max;
      if (this.pi_right[i] < -this.params.momentum.pi_max) this.pi_right[i] = -this.params.momentum.pi_max;
    }
  }

  // FIX #2: Physical Phase Gradient Initialization
  addMomentumViaPhase(center: number, velocity: number, width: number) {
    for (let i = 0; i < N; i++) {
      const dx = periodicDistance(i, center, N);
      const envelope = Math.exp(-0.5 * (dx / width) ** 2);
      // theta(x) ~ v * x
      this.theta[i] += velocity * dx * envelope;
      this.theta[i] = mod(this.theta[i], 2 * Math.PI);
    }
  }

  initDipoleCollision(separation: number, initialMomentum: number, usePhase: boolean) {
    this.reset();
    const center = N / 2;
    const left = center - separation / 2;
    const right = center + separation / 2;
    const width = 5.0;
    const amp = 6.0;

    // Add Resources
    this.addGaussian(left, amp, width);
    this.addGaussian(right, amp, width);

    // Add Momentum
    if (initialMomentum !== 0) {
      if (usePhase) {
        this.addMomentumViaPhase(left, initialMomentum, 25.0);
        this.addMomentumViaPhase(right, -initialMomentum, 25.0);
      } else {
        this.addMomentumImpulse(left, initialMomentum, 25.0);
        this.addMomentumImpulse(right, -initialMomentum, 25.0);
      }
    }
  }

  // --- CORE STEP (FIXES APPLIED) ---

  step() {
    const dt = this.params.DT;
    const p = this.params;
    
    // Arrays for next step flow
    const J_right = new Float32Array(N);
    const J_left = new Float32Array(N);
    const J_diff_R = new Float32Array(N);
    const J_diff_L = new Float32Array(N); // Needed for momentum update
    const Delta_tau_right = new Float32Array(N);

    // 1. Presence & Local Time
    for (let i = 0; i < N; i++) {
      this.P[i] = this.a[i] / (1.0 + this.F[i]);
      this.Delta_tau[i] = this.P[i] * dt;
    }

    // 2. Wavefunction & Flow
    // FIX #1: Periodic local sum
    const F_local = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        let sum = 0;
        for (let offset = -p.R; offset <= p.R; offset++) {
            sum += this.F[mod(i + offset, N)];
        }
        F_local[i] = sum + 1e-9;
    }

    const amp = new Float32Array(N);
    for(let i=0; i<N; i++) {
        amp[i] = Math.sqrt(Math.max(0, Math.min(1, this.F[i] / F_local[i])));
    }

    for (let i = 0; i < N; i++) {
      const ip1 = mod(i + 1, N);
      const im1 = mod(i - 1, N);

      // Bond times
      Delta_tau_right[i] = 0.5 * (this.Delta_tau[i] + this.Delta_tau[ip1]);

      // Quantum Flow (Imag part of conjugate product)
      // A*exp(-it1) * B*exp(it2) = AB exp(i(t2-t1)) -> Imag = AB sin(t2-t1)
      const quantum_R = amp[i] * amp[ip1] * Math.sin(this.theta[ip1] - this.theta[i]);
      const quantum_L = amp[i] * amp[im1] * Math.sin(this.theta[im1] - this.theta[i]);

      const classical_R = this.F[i] - this.F[ip1];
      const classical_L = this.F[i] - this.F[im1];

      const sqrt_C_R = Math.sqrt(this.C_right[i]);
      const sqrt_C_L = Math.sqrt(this.C_left[i]);

      const drive_R = sqrt_C_R * quantum_R + (1 - sqrt_C_R) * classical_R;
      const drive_L = sqrt_C_L * quantum_L + (1 - sqrt_C_L) * classical_L;

      const cond_R = this.sigma[i] * (this.C_right[i] + 1e-4);
      const cond_L = this.sigma[i] * (this.C_left[i] + 1e-4);

      J_diff_R[i] = cond_R * drive_R;
      J_diff_L[i] = cond_L * drive_L; // Store for momentum update

      // Momentum Flow
      let J_mom_R = 0;
      let J_mom_L = 0;
      if (p.momentum.enabled) {
        const F_avg_R = 0.5 * (this.F[i] + this.F[ip1]);
        const F_avg_L = 0.5 * (this.F[i] + this.F[im1]);
        
        // J_mom_L comes from neighbor's pi_right rolled
        const pi_prev = this.pi_right[im1];

        J_mom_R = p.momentum.mu_pi * this.sigma[i] * this.pi_right[i] * F_avg_R;
        J_mom_L = -p.momentum.mu_pi * this.sigma[i] * pi_prev * F_avg_L;
      }

      J_right[i] = J_diff_R[i] + J_mom_R;
      J_left[i] = J_diff_L[i] + J_mom_L;
    }

    // FIX #5: Generalized Conservative Limiter
    // Calculate total outflow per node and scale if needed
    for (let i = 0; i < N; i++) {
        // Outflow R is J_right[i] > 0
        // Outflow L is J_left[i] > 0
        const out_R = Math.max(0, J_right[i]);
        const out_L = Math.max(0, J_left[i]);
        const total_out = out_R + out_L;
        const max_out = p.outflow_limit * this.F[i] / dt;

        if (total_out > max_out && total_out > 1e-9) {
            const scale = max_out / total_out;
            if (J_right[i] > 0) {
                J_right[i] *= scale;
                // Also scale diff component for momentum consistency
                 if (J_diff_R[i] > 0) J_diff_R[i] *= scale;
            }
            if (J_left[i] > 0) {
                J_left[i] *= scale;
                 if (J_diff_L[i] > 0) J_diff_L[i] *= scale;
            }
        }
    }

    // Updates
    const dF = new Float32Array(N);
    const next_F = new Float32Array(N);

    // 3. Resource Update
    for (let i = 0; i < N; i++) {
      const im1 = mod(i - 1, N);
      const ip1 = mod(i + 1, N);
      
      // Inflow from left neighbor (their J_right) + Inflow from right neighbor (their J_left)
      // Minus Outflow right + Outflow left
      const flow_in = J_right[im1] + J_left[ip1];
      const flow_out = J_right[i] + J_left[i];
      
      dF[i] = flow_in - flow_out;
      next_F[i] = Math.max(p.F_VAC, Math.min(1000, this.F[i] + dF[i] * dt));
    }

    // 4. Momentum Update (FIX #3)
    if (p.momentum.enabled) {
        for (let i = 0; i < N; i++) {
            // FIX #3: Clamp decay
            const decay = Math.max(0.0, 1.0 - p.momentum.lambda_pi * Delta_tau_right[i]);
            const drive = p.momentum.alpha_pi * J_diff_R[i] * Delta_tau_right[i];
            
            this.pi_right[i] = decay * this.pi_right[i] + drive;
            
            // Clip
            if (this.pi_right[i] > p.momentum.pi_max) this.pi_right[i] = p.momentum.pi_max;
            if (this.pi_right[i] < -p.momentum.pi_max) this.pi_right[i] = -p.momentum.pi_max;
        }
    }

    // 5. q-locking
    if (p.q_locking.enabled) {
        for (let i = 0; i < N; i++) {
            const delta_F = next_F[i] - this.F[i];
            const dq = p.q_locking.alpha_q * Math.max(0, -delta_F);
            this.q[i] = Math.min(1, Math.max(0, this.q[i] + dq));
        }
    }

    // Apply F
    for (let i = 0; i < N; i++) this.F[i] = next_F[i];

    // 6. Phase Update
    const next_theta = new Float32Array(N);
    for (let i = 0; i < N; i++) {
        const ip1 = mod(i + 1, N);
        const im1 = mod(i - 1, N);
        
        // Laplacian on unit circle
        const d_fwd = Math.atan2(Math.sin(this.theta[ip1] - this.theta[i]), Math.cos(this.theta[ip1] - this.theta[i]));
        const d_bwd = Math.atan2(Math.sin(this.theta[im1] - this.theta[i]), Math.cos(this.theta[im1] - this.theta[i]));
        
        next_theta[i] = this.theta[i] + p.nu * (d_fwd + d_bwd) * dt + p.omega_0 * this.Delta_tau[i];
        next_theta[i] = mod(next_theta[i], 2 * Math.PI);
    }
    for (let i = 0; i < N; i++) this.theta[i] = next_theta[i];

    // 7. Coherence Update
    for (let i = 0; i < N; i++) {
        const compression = Math.max(0, dF[i]);
        const alpha_eff = p.coherence.alpha_c * (1.0 + 20.0 * compression);
        
        // Alignment terms
        // J_R_align includes neighbor flow
        const ip1 = mod(i + 1, N);
        const im1 = mod(i - 1, N);
        
        const J_R_self = Math.max(0, J_right[i]);
        const J_R_neighbor = Math.max(0, J_right[ip1]); // approximate roll
        
        const J_L_self = Math.max(0, J_left[i]);
        const J_L_neighbor = Math.max(0, J_left[im1]);

        const drive_R = alpha_eff * (J_R_self + p.coherence.gamma_c * J_R_neighbor);
        const drive_L = alpha_eff * (J_L_self + p.coherence.gamma_c * J_L_neighbor);
        
        this.C_right[i] += (drive_R - p.coherence.lambda_c * this.C_right[i]) * dt;
        this.C_left[i] += (drive_L - p.coherence.lambda_c * this.C_left[i]) * dt;
        
        this.C_right[i] = Math.max(p.coherence.c_min, Math.min(1, this.C_right[i]));
        this.C_left[i] = Math.max(p.coherence.c_min, Math.min(1, this.C_left[i]));
    }

    // 8. Conductivity
    for (let i = 0; i < N; i++) {
        const absFlow = Math.abs(J_right[i]) + Math.abs(J_left[i]);
        this.sigma[i] = 1.0 + 0.1 * Math.log(1.0 + absFlow);
    }

    this.time += dt;
    this.stepCount++;
  }

  // FIX #4: Robust Separation Calculation (Blob based)
  getSeparation() {
    // 1. Identify blobs (F > threshold)
    const threshold = this.params.F_VAC * 10;
    const isOver = new Uint8Array(N);
    let overCount = 0;
    
    for(let i=0; i<N; i++) {
        if(this.F[i] > threshold) {
            isOver[i] = 1;
            overCount++;
        } else {
            isOver[i] = 0;
        }
    }

    if (overCount < 2) return { sep: 0, blobs: 0 };

    // Simple 1D Clustering
    const blobs: {mass: number, com: number}[] = [];
    let currentBlob = { positions: [] as number[], mass: 0 };
    
    // Handle wrap-around by rotating array until we find a gap?
    // Easier: Just do 2-pass labeling. 
    // Pass 1: standard segments
    const segments = [];
    let inSegment = false;
    let currentSeg = [];
    
    for(let i=0; i<N; i++) {
        if(isOver[i]) {
            if(!inSegment) { inSegment = true; currentSeg = []; }
            currentSeg.push(i);
        } else {
            if(inSegment) { segments.push(currentSeg); inSegment = false; }
        }
    }
    if(inSegment) segments.push(currentSeg);

    // Merge wrap-around (first and last segments)
    if(segments.length > 1 && isOver[0] && isOver[N-1]) {
        const first = segments.shift();
        const last = segments.pop();
        if (first && last) {
            const merged = last.concat(first); // Note: indices will be [N-k... N-1, 0... k]
            segments.push(merged);
        }
    }

    // Calculate COM for each segment
    for(const seg of segments) {
        let mass = 0;
        let cosSum = 0;
        let sinSum = 0;
        
        for(const idx of seg) {
            const val = this.F[idx];
            mass += val;
            const angle = 2 * Math.PI * idx / N;
            cosSum += val * Math.cos(angle);
            sinSum += val * Math.sin(angle);
        }
        
        if (mass > 0) {
            const avgCos = cosSum / mass;
            const avgSin = sinSum / mass;
            let avgAngle = Math.atan2(avgSin, avgCos);
            if (avgAngle < 0) avgAngle += 2 * Math.PI;
            const com = avgAngle * N / (2 * Math.PI);
            blobs.push({ mass, com });
        }
    }
    
    // Sort by mass
    blobs.sort((a,b) => b.mass - a.mass);
    
    if (blobs.length < 2) return { sep: 0, blobs: blobs.length };
    
    const d = periodicDistance(blobs[1].com, blobs[0].com, N);
    return { sep: Math.abs(d), blobs: blobs.length };
  }

  getTotalEnergy() {
    let sum = 0;
    for(let i=0; i<N; i++) sum += this.F[i];
    return sum;
  }
}

// --- VISUALIZATION COMPONENT ---

const DETColliderV2: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const engine = useRef(new DETEngine());
  const requestRef = useRef<number>();
  
  const [isPlaying, setIsPlaying] = useState(false);
  const [separation, setSeparation] = useState(0);
  const [energy, setEnergy] = useState(0);
  const [blobCount, setBlobCount] = useState(0);
  const [step, setStep] = useState(0);
  
  // Settings
  const [initSep, setInitSep] = useState(100);
  const [initMom, setInitMom] = useState(0.5);
  const [friction, setFriction] = useState(0.01);
  const [usePhase, setUsePhase] = useState(false); // Fix #2 Toggle
  const [useQLocking, setUseQLocking] = useState(true);

  // Initialize
  useEffect(() => {
    resetSim();
    return () => stopLoop();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current && canvasRef.current) {
        canvasRef.current.width = containerRef.current.clientWidth;
        canvasRef.current.height = 300;
        draw();
      }
    };
    window.addEventListener('resize', handleResize);
    handleResize();
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const resetSim = () => {
    stopLoop();
    engine.current.params.momentum.lambda_pi = friction;
    engine.current.params.q_locking.enabled = useQLocking;
    engine.current.initDipoleCollision(initSep, initMom, usePhase);
    draw();
    updateStats();
  };

  const updateStats = () => {
      const metrics = engine.current.getSeparation();
      setSeparation(metrics.sep);
      setBlobCount(metrics.blobs);
      setEnergy(engine.current.getTotalEnergy());
      setStep(engine.current.stepCount);
  };

  const loop = () => {
    // Run multiple substeps per frame for speed
    for(let i=0; i<8; i++) {
        engine.current.step();
    }
    draw();
    updateStats();
    if (isPlaying) requestRef.current = requestAnimationFrame(loop);
  };

  const startLoop = () => {
    if (!isPlaying) {
      setIsPlaying(true);
      requestRef.current = requestAnimationFrame(loop);
    }
  };

  const stopLoop = () => {
    setIsPlaying(false);
    if (requestRef.current) cancelAnimationFrame(requestRef.current);
  };

  const togglePlay = () => isPlaying ? stopLoop() : startLoop();

  const draw = () => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext('2d');
    if (!ctx) return;
    
    const w = cvs.width;
    const h = cvs.height;
    const e = engine.current;
    
    // Clear
    ctx.fillStyle = '#111827'; // Dark bg
    ctx.fillRect(0, 0, w, h);
    
    // Grid
    ctx.strokeStyle = '#374151';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h/2);
    ctx.lineTo(w, h/2);
    ctx.stroke();

    const maxF = 12.0; // Fixed scale for stability visual

    // Draw F (Cyan Fill)
    ctx.fillStyle = 'rgba(34, 211, 238, 0.2)';
    ctx.strokeStyle = '#22d3ee';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < N; i++) {
        const x = (i / N) * w;
        const y = h - (e.F[i] / maxF) * h;
        ctx.lineTo(x, y);
    }
    ctx.lineTo(w, h);
    ctx.fill();
    ctx.stroke();

    // Draw Momentum (Magenta Line)
    ctx.strokeStyle = '#e879f9';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
        const x = (i / N) * w;
        // Scale momentum for visibility
        const piVal = e.pi_right[i];
        const y = h/2 - (piVal * 30); 
        if (i===0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw q (Yellow Line)
    if (useQLocking) {
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        for (let i = 0; i < N; i++) {
            const x = (i / N) * w;
            const y = h - (e.q[i] * 0.9 * h); 
            if (i===0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }
  };

  return (
    <div className="flex flex-col h-full bg-gray-950 text-gray-100 p-6 font-sans">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-blue-400 flex items-center gap-2">
            <Activity className="w-6 h-6" /> DET v5 Collider Suite v2
          </h1>
          <p className="text-gray-400 text-sm mt-1">
             Physics Engine: Periodic Local Sum • Conservative Limiter • Phase Gradient Init
          </p>
        </div>
        <div className="flex gap-4">
             <div className="flex flex-col items-end">
                <span className="text-xs text-gray-500 uppercase">Separation</span>
                <span className={`text-xl font-mono ${separation < 20 && separation > 0 ? 'text-red-500 animate-pulse' : 'text-green-400'}`}>
                    {separation.toFixed(1)}
                </span>
             </div>
             <div className="flex flex-col items-end">
                <span className="text-xs text-gray-500 uppercase">Energy (F)</span>
                <span className="text-xl font-mono text-cyan-400">{energy.toFixed(1)}</span>
             </div>
        </div>
      </div>

      {/* Main Display */}
      <div ref={containerRef} className="flex-1 min-h-0 bg-gray-900 rounded-xl border border-gray-800 overflow-hidden relative shadow-2xl mb-6">
        <canvas ref={canvasRef} className="block w-full h-full" />
        
        {/* Legend Overlay */}
        <div className="absolute top-4 right-4 bg-gray-950/80 backdrop-blur p-3 rounded-lg border border-gray-800 text-xs space-y-2">
            <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-cyan-400/50 border border-cyan-400 rounded-sm"></div>
                <span>Resource Field (F)</span>
            </div>
            <div className="flex items-center gap-2">
                <div className="w-3 h-0.5 bg-fuchsia-400"></div>
                <span>Momentum (π)</span>
            </div>
            {useQLocking && (
                <div className="flex items-center gap-2">
                    <div className="w-3 h-0.5 border-t-2 border-dashed border-yellow-400"></div>
                    <span>Structure (q)</span>
                </div>
            )}
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 bg-gray-900 p-4 rounded-xl border border-gray-800">
        
        {/* Playback */}
        <div className="space-y-4">
            <div className="flex gap-2">
                <button
                    onClick={togglePlay}
                    className={`flex-1 py-2 px-4 rounded-lg font-medium flex items-center justify-center gap-2 transition-colors ${
                    isPlaying 
                        ? 'bg-amber-500/20 text-amber-400 hover:bg-amber-500/30' 
                        : 'bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30'
                    }`}
                >
                    {isPlaying ? <><Pause size={18} /> Pause</> : <><Play size={18} /> Run</>}
                </button>
                <button
                    onClick={resetSim}
                    className="px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors"
                    title="Reset Simulation"
                >
                    <RotateCcw size={18} />
                </button>
            </div>
            <div className="text-xs text-gray-500 font-mono">
                Step: {step} | Blobs: {blobCount}
            </div>
        </div>

        {/* Physics Params */}
        <div className="space-y-4">
            <div className="flex items-center justify-between">
                <label className="text-sm text-gray-400 flex items-center gap-2">
                    <Wind size={14} /> Initial Momentum
                </label>
                <span className="text-xs font-mono text-cyan-400">{initMom.toFixed(1)}</span>
            </div>
            <input 
                type="range" min="0" max="1.5" step="0.1"
                value={initMom}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setInitMom(v);
                    // Live update if paused/reset logic permits, but simpler to wait for reset
                }}
                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-500"
            />
            
            <div className="flex items-center justify-between">
                <label className="text-sm text-gray-400 flex items-center gap-2">
                    <Settings size={14} /> Friction (λ)
                </label>
                <span className="text-xs font-mono text-fuchsia-400">{friction.toFixed(3)}</span>
            </div>
            <input 
                type="range" min="0.001" max="0.1" step="0.001"
                value={friction}
                onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setFriction(v);
                    if (engine.current) engine.current.params.momentum.lambda_pi = v;
                }}
                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-fuchsia-500"
            />
        </div>

        {/* Toggles */}
        <div className="space-y-3">
             <button
                onClick={() => setUsePhase(!usePhase)}
                className={`w-full py-2 px-3 rounded-lg text-sm flex items-center justify-between transition-colors border ${
                    usePhase 
                    ? 'bg-blue-900/30 border-blue-700 text-blue-300' 
                    : 'bg-gray-800 border-gray-700 text-gray-400'
                }`}
            >
                <span className="flex items-center gap-2"><Zap size={14} /> Phase Gradient Init</span>
                <span className="text-xs font-mono">{usePhase ? 'ON (Physical)' : 'OFF (Impulse)'}</span>
            </button>
            
            <button
                onClick={() => {
                    const newVal = !useQLocking;
                    setUseQLocking(newVal);
                    if (engine.current) engine.current.params.q_locking.enabled = newVal;
                }}
                className={`w-full py-2 px-3 rounded-lg text-sm flex items-center justify-between transition-colors border ${
                    useQLocking 
                    ? 'bg-yellow-900/30 border-yellow-700 text-yellow-300' 
                    : 'bg-gray-800 border-gray-700 text-gray-400'
                }`}
            >
                <span className="flex items-center gap-2"><TrendingUp size={14} /> Structure (q-lock)</span>
                <span className="text-xs font-mono">{useQLocking ? 'ENABLED' : 'DISABLED'}</span>
            </button>
        </div>

      </div>
    </div>
  );
};

export default DETColliderV2;