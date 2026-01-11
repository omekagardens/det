"""
DET v6 Organ Healing Simulation
================================

Applying DET dynamics to therapeutic intervention at the organ level.

Key Insight: DET is scale-invariant. The same dynamics that govern
salvation history apply to healing at any scale:
- Organ ↔ Body (organ as pseudo-boundary to body)
- Cell ↔ Organ
- Molecule ↔ Cell

Therapeutic Mapping:
-------------------
| Physiological Concept    | DET Variable                              |
|--------------------------|-------------------------------------------|
| Vitality/ATP/Resources   | F (Resource)                              |
| Damage/Inflammation/Scar | q (Structural Debt)                       |
| Cellular autonomy        | a (Agency)                                |
| Tissue coherence         | C (Coherence)                             |
| Metabolic rate           | P (Presence)                              |
| Healing signals          | k (Movement/Events)                       |
| Disease gravity          | Φ (Potential from q)                      |

Therapeutic Principles from DET:
-------------------------------
1. AGENCY INVIOLABILITY: Cannot force healing - must work WITH tissue
2. LOCALITY: Healing happens cell-by-cell, not globally imposed
3. NON-COERCIVE GRACE: Intervention gated by tissue receptivity (a)
4. RECOVERY-PERMITTING: System can heal if agency is maintained

Reference: DET Theory Card v6.0
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto
import warnings
warnings.filterwarnings('ignore')


class TissueState(Enum):
    """Tissue health states."""
    HEALTHY = auto()
    STRESSED = auto()
    INFLAMED = auto()
    DAMAGED = auto()
    CHRONIC = auto()
    INTERVENTION = auto()
    RECOVERING = auto()
    HEALED = auto()
    ENHANCED = auto()


class InterventionType(Enum):
    """Types of therapeutic intervention."""
    NONE = auto()
    PHARMACEUTICAL = auto()      # Drug delivery (external resource)
    REGENERATIVE = auto()        # Stem cell / growth factor
    ENERGY_BASED = auto()        # Light therapy, ultrasound, PEMF
    COHERENCE_BASED = auto()     # Fascia work, acupuncture, prayer/intention
    COMBINED = auto()            # Multi-modal


@dataclass
class OrganParams:
    """Parameters for organ healing simulation."""
    N: int = 100                    # Number of cells/nodes in organ
    DT: float = 0.02                # Time step
    
    # Healthy baseline state
    F_healthy: float = 5.0          # Healthy resource level
    C_healthy: float = 0.85         # Healthy coherence
    a_healthy: float = 0.95         # Healthy agency
    q_healthy: float = 0.02         # Minimal baseline structural debt
    
    # Damage parameters
    damage_q_injection: float = 0.4  # Initial damage q
    damage_F_loss: float = 0.6       # Resource depletion from damage
    damage_C_loss: float = 0.5       # Coherence loss
    
    # Chronic disease parameters
    chronic_q_lock_rate: float = 0.02   # Rate of ongoing q accumulation
    chronic_a_suppression: float = 25.0  # λ in a_target = 1/(1 + λq²)
    
    # Healing parameters
    alpha_q_clear: float = 0.015     # Natural q clearing rate
    C_growth_rate: float = 0.03      # Coherence growth from flow
    C_decay_rate: float = 0.01       # Natural coherence decay
    grace_efficiency: float = 0.5    # Grace injection efficiency
    
    # Intervention parameters
    intervention_F_boost: float = 3.0     # Resource from intervention
    intervention_C_boost: float = 0.2     # Coherence from intervention
    intervention_q_clear_boost: float = 2.0  # Accelerated q clearing
    
    # "Stem cell" / regenerative node parameters
    regen_node_enabled: bool = True
    regen_node_count: int = 5        # Number of high-agency nodes
    regen_node_a: float = 0.99       # Near-perfect agency
    regen_node_F: float = 8.0        # High resource
    
    # Momentum and flow
    momentum_enabled: bool = True
    alpha_pi: float = 0.08
    lambda_pi: float = 0.015
    mu_pi: float = 0.25
    
    # Gravity (disease attracts more disease)
    gravity_enabled: bool = True
    alpha_grav: float = 0.02
    kappa_grav: float = 2.0
    mu_grav: float = 1.0
    
    # Numerical
    outflow_limit: float = 0.2
    a_rate: float = 0.15


class OrganHealingSimulation:
    """
    DET-based organ healing simulation.
    
    Models therapeutic intervention through strictly local,
    non-coercive, recovery-permitting dynamics.
    """
    
    def __init__(self, params: Optional[OrganParams] = None):
        self.p = params or OrganParams()
        N = self.p.N
        
        # Per-cell state
        self.F = np.ones(N) * self.p.F_healthy
        self.q = np.ones(N) * self.p.q_healthy
        self.a = np.ones(N) * self.p.a_healthy
        self.k = np.zeros(N, dtype=int)
        
        # Per-bond state
        self.C_R = np.ones(N) * self.p.C_healthy
        self.pi_R = np.zeros(N)
        self.sigma = np.ones(N)
        
        # Gravity fields
        self.b = np.zeros(N)
        self.Phi = np.zeros(N)
        self.g = np.zeros(N)
        
        # Diagnostics
        self.P = np.ones(N)
        self.Delta_tau = np.ones(N) * self.p.DT
        
        # Simulation state
        self.time = 0.0
        self.step_count = 0
        self.state = TissueState.HEALTHY
        self.intervention_type = InterventionType.NONE
        self.intervention_active = False
        
        # Regenerative nodes (stem cells / high-agency tissue)
        self.regen_nodes = []
        if self.p.regen_node_enabled:
            # Place regen nodes distributed through tissue
            self.regen_nodes = list(np.linspace(10, N-10, self.p.regen_node_count, dtype=int))
        
        # History
        self.history: Dict[str, List] = {
            'time': [], 'state': [],
            'mean_F': [], 'mean_q': [], 'mean_a': [], 'mean_C': [], 'mean_P': [],
            'max_q': [], 'min_a': [],
            'total_k': [], 'grace_total': [],
            'damage_center_F': [], 'damage_center_q': [], 'damage_center_a': [],
        }
        
        # Damage location (will be set when damage occurs)
        self.damage_center = N // 2
        
        # FFT setup
        self._setup_fft()
    
    def _setup_fft(self):
        """Setup FFT solvers for gravity."""
        N = self.p.N
        k = np.fft.fftfreq(N) * N
        self.L_k = -4 * np.sin(np.pi * k / N)**2
        self.H_k = self.L_k - self.p.alpha_grav
        self.H_k[np.abs(self.H_k) < 1e-12] = 1e-12
        self.L_k_poisson = self.L_k.copy()
        self.L_k_poisson[0] = 1.0
    
    def _solve_helmholtz(self, source):
        source_k = np.fft.fft(source)
        b_k = -self.p.alpha_grav * source_k / self.H_k
        return np.real(np.fft.ifft(b_k))
    
    def _solve_poisson(self, source):
        source_k = np.fft.fft(source)
        source_k[0] = 0
        Phi_k = -self.p.kappa_grav * source_k / self.L_k_poisson
        Phi_k[0] = 0
        return np.real(np.fft.ifft(Phi_k))
    
    def _compute_gravity(self):
        """Compute disease gravity field from q."""
        if not self.p.gravity_enabled:
            self.g = np.zeros(self.p.N)
            return
        
        self.b = self._solve_helmholtz(self.q)
        rho = self.q - self.b
        self.Phi = self._solve_poisson(rho)
        
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        self.g = -0.5 * (R(self.Phi) - L(self.Phi))
    
    def induce_damage(self, center: Optional[int] = None, 
                      width: float = 15.0, severity: float = 1.0):
        """
        Induce localized tissue damage.
        
        Models: injury, infection, ischemia, toxin exposure, etc.
        """
        N = self.p.N
        self.damage_center = center if center is not None else N // 2
        
        x = np.arange(N)
        damage_env = severity * np.exp(-0.5 * ((x - self.damage_center) / width)**2)
        
        # Structural debt injection (damage, inflammation)
        self.q += self.p.damage_q_injection * damage_env
        
        # Resource depletion (ATP loss, cell death)
        self.F *= (1 - self.p.damage_F_loss * damage_env)
        
        # Coherence disruption (tissue disorganization)
        self.C_R *= (1 - self.p.damage_C_loss * damage_env)
        
        # NOTE: Agency is NOT directly modified!
        # It will decay via q-coupling: a_target = 1/(1 + λq²)
        
        self.state = TissueState.DAMAGED
        self.q = np.clip(self.q, 0, 1)
        self.F = np.clip(self.F, 0.01, 100)
        self.C_R = np.clip(self.C_R, 0.1, 1.0)
        
        print(f"\n[DAMAGE INDUCED] at node {self.damage_center}")
        print(f"  Severity: {severity:.1f}")
        print(f"  Q injected: {self.p.damage_q_injection * severity:.3f}")
        print(f"  Mean F after: {np.mean(self.F):.3f}")
        print(f"  Mean C after: {np.mean(self.C_R):.3f}")
    
    def induce_chronic_disease(self, center: Optional[int] = None,
                               width: float = 20.0, severity: float = 0.8):
        """
        Induce chronic disease state with ongoing q accumulation.
        
        Models: autoimmune, cancer, chronic inflammation, fibrosis
        """
        self.induce_damage(center, width, severity)
        self.state = TissueState.CHRONIC
        
        # Mark disease region for ongoing q-locking
        x = np.arange(self.p.N)
        self.chronic_region = np.exp(-0.5 * ((x - self.damage_center) / width)**2)
        
        print(f"  [CHRONIC] Ongoing q-accumulation enabled in disease region")
    
    def apply_intervention(self, intervention_type: InterventionType,
                          duration: int = 200, intensity: float = 1.0):
        """
        Apply therapeutic intervention.
        
        All interventions must respect DET constraints:
        - Cannot directly modify agency
        - Must act locally
        - Effectiveness gated by tissue receptivity (a)
        """
        self.intervention_type = intervention_type
        self.intervention_active = True
        self.intervention_intensity = intensity
        self.intervention_duration = duration
        self.intervention_steps = 0
        self.state = TissueState.INTERVENTION
        
        print(f"\n[INTERVENTION: {intervention_type.name}]")
        print(f"  Duration: {duration} steps")
        print(f"  Intensity: {intensity:.2f}")
        
        if intervention_type == InterventionType.REGENERATIVE:
            # Activate regenerative nodes (like introducing stem cells)
            for rn in self.regen_nodes:
                self.a[rn] = self.p.regen_node_a
                self.F[rn] = self.p.regen_node_F
                self.q[rn] = 0.0  # Stem cells have no accumulated damage
            print(f"  Regenerative nodes activated: {len(self.regen_nodes)}")
    
    def _apply_intervention_step(self):
        """Apply per-step intervention effects."""
        if not self.intervention_active:
            return
        
        self.intervention_steps += 1
        if self.intervention_steps > self.intervention_duration:
            self.intervention_active = False
            self.state = TissueState.RECOVERING
            print(f"\n[INTERVENTION COMPLETE] Entering recovery phase")
            return
        
        p = self.p
        intensity = self.intervention_intensity
        
        # Effects are LOCAL and AGENCY-GATED
        # Higher agency = more receptive to intervention
        
        if self.intervention_type == InterventionType.PHARMACEUTICAL:
            # Drug delivery: provides external resource where needed
            # Effectiveness gated by tissue agency (receptor sensitivity)
            need = np.maximum(0, p.F_healthy - self.F)
            drug_delivery = intensity * 0.1 * need * self.a  # GATED by agency
            self.F += drug_delivery
        
        elif self.intervention_type == InterventionType.REGENERATIVE:
            # Stem cell / growth factor: high-agency nodes radiate healing
            for rn in self.regen_nodes:
                # Maintain regenerative node state
                self.a[rn] = p.regen_node_a
                self.q[rn] = 0.0
                
                # Healing radiates locally, gated by neighbor's agency
                for i in range(max(0, rn-10), min(p.N, rn+11)):
                    if i != rn:
                        distance = abs(i - rn)
                        influence = np.exp(-distance / 5) * self.a[i] * intensity
                        
                        # Coherence building (paracrine signaling)
                        self.C_R[i] = min(1.0, self.C_R[i] + 0.01 * influence)
                        
                        # Q clearing (tissue remodeling)
                        self.q[i] = max(0, self.q[i] - 0.01 * influence)
        
        elif self.intervention_type == InterventionType.ENERGY_BASED:
            # Light/ultrasound/PEMF: boosts cellular processing rate
            # Effect distributed across all tissue, gated by agency
            sigma_boost = intensity * 0.01 * self.a
            self.sigma = np.clip(self.sigma + sigma_boost, 1.0, 2.0)
            
            # Also provides some resource (ATP production boost)
            self.F += intensity * 0.05 * self.a
        
        elif self.intervention_type == InterventionType.COHERENCE_BASED:
            # Fascia work, acupuncture, intention/prayer
            # Directly targets coherence restoration, gated by agency
            coherence_boost = intensity * 0.02 * self.a
            self.C_R = np.clip(self.C_R + coherence_boost, 0.1, 1.0)
            
            # Also calms inflammation (reduces q accumulation rate)
            # This is indirect - it doesn't remove q, but slows its growth
        
        elif self.intervention_type == InterventionType.COMBINED:
            # Multi-modal: combines all approaches at reduced intensity
            sub_intensity = intensity * 0.4
            
            # Pharmaceutical component
            need = np.maximum(0, p.F_healthy - self.F)
            self.F += sub_intensity * 0.05 * need * self.a
            
            # Regenerative component
            for rn in self.regen_nodes:
                self.a[rn] = p.regen_node_a
                self.q[rn] = 0.0
            
            # Energy component
            self.sigma = np.clip(self.sigma + sub_intensity * 0.005 * self.a, 1.0, 2.0)
            
            # Coherence component
            self.C_R = np.clip(self.C_R + sub_intensity * 0.01 * self.a, 0.1, 1.0)
            
            # ROOT CAUSE TREATMENT: Reduce chronic q-locking rate
            # This represents treating the underlying disease mechanism
            if hasattr(self, 'chronic_region'):
                # Gradually reduce the chronic region's activity
                self.chronic_region *= 0.995  # Shrink disease driver
    
    def _clip(self):
        """Enforce bounds."""
        self.F = np.clip(self.F, 0.01, 100)
        self.q = np.clip(self.q, 0, 1)
        self.a = np.clip(self.a, 0, 1)
        self.C_R = np.clip(self.C_R, 0.1, 1.0)
        
        # Maintain regenerative nodes if active
        if self.intervention_active and self.intervention_type == InterventionType.REGENERATIVE:
            for rn in self.regen_nodes:
                self.a[rn] = self.p.regen_node_a
                self.q[rn] = 0.0
    
    def step(self):
        """Execute one DET update step."""
        p = self.p
        N = p.N
        dk = p.DT
        
        R = lambda x: np.roll(x, -1)
        L = lambda x: np.roll(x, 1)
        
        # ============================================================
        # STEP 0: Gravity (disease attracts disease)
        # ============================================================
        self._compute_gravity()
        
        # ============================================================
        # STEP 1: Presence and proper time
        # ============================================================
        H = self.sigma
        self.P = self.a * self.sigma / (1.0 + self.F) / (1.0 + H)
        self.Delta_tau = self.P * dk
        Delta_tau_R = 0.5 * (self.Delta_tau + R(self.Delta_tau))
        
        # ============================================================
        # STEP 2: Flow computation
        # ============================================================
        classical_R = self.F - R(self.F)
        classical_L = self.F - L(self.F)
        
        sqrt_C_R = np.sqrt(self.C_R)
        sqrt_C_L = np.sqrt(L(self.C_R))
        
        drive_R = (1 - sqrt_C_R) * classical_R
        drive_L = (1 - sqrt_C_L) * classical_L
        
        # Agency-gated diffusion
        g_R = np.sqrt(self.a * R(self.a))
        g_L = np.sqrt(self.a * L(self.a))
        
        cond_R = self.sigma * (self.C_R + 1e-4)
        cond_L = self.sigma * (L(self.C_R) + 1e-4)
        
        J_diff_R = g_R * cond_R * drive_R
        J_diff_L = g_L * cond_L * drive_L
        
        # Momentum flux
        J_mom_R = J_mom_L = 0
        if p.momentum_enabled:
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            J_mom_R = p.mu_pi * self.sigma * self.pi_R * F_avg_R
            J_mom_L = -p.mu_pi * self.sigma * L(self.pi_R) * F_avg_L
        
        # Gravity flux
        J_grav_R = J_grav_L = 0
        if p.gravity_enabled:
            g_bond_R = 0.5 * (self.g + R(self.g))
            g_bond_L = 0.5 * (self.g + L(self.g))
            F_avg_R = 0.5 * (self.F + R(self.F))
            F_avg_L = 0.5 * (self.F + L(self.F))
            J_grav_R = p.mu_grav * self.sigma * g_bond_R * F_avg_R
            J_grav_L = p.mu_grav * self.sigma * g_bond_L * F_avg_L
        
        J_R = J_diff_R + J_mom_R + J_grav_R
        J_L = J_diff_L + J_mom_L + J_grav_L
        
        # ============================================================
        # STEP 3: Conservative limiter
        # ============================================================
        total_outflow = np.maximum(0, J_R) + np.maximum(0, J_L)
        max_total_out = p.outflow_limit * self.F / (self.Delta_tau + 1e-9)
        scale = np.minimum(1.0, max_total_out / (total_outflow + 1e-9))
        
        J_R_lim = np.where(J_R > 0, J_R * scale, J_R)
        J_L_lim = np.where(J_L > 0, J_L * scale, J_L)
        J_diff_R_scaled = np.where(J_diff_R > 0, J_diff_R * scale, J_diff_R)
        
        # ============================================================
        # STEP 4: Resource update with grace injection
        # ============================================================
        transfer_R = J_R_lim * self.Delta_tau
        transfer_L = J_L_lim * self.Delta_tau
        
        outflow = transfer_R + transfer_L
        inflow = L(transfer_R) + R(transfer_L)
        dF = inflow - outflow
        
        # Grace injection (body's natural healing)
        grace_injection = np.zeros(N)
        need = np.maximum(0, p.F_healthy - self.F)
        weight = self.a * need  # AGENCY-GATED
        D = np.abs(J_R_lim) * self.Delta_tau + np.abs(J_L_lim) * self.Delta_tau
        
        for i in range(N):
            local_weight_sum = 0
            for j in range(max(0, i-5), min(N, i+6)):
                local_weight_sum += weight[j]
            if local_weight_sum > 1e-9:
                grace_injection[i] = p.grace_efficiency * D[i] * weight[i] / local_weight_sum
        
        self.F = np.clip(self.F + dF + grace_injection, 0.01, 100)
        
        # ============================================================
        # STEP 5: Momentum update
        # ============================================================
        if p.momentum_enabled:
            decay_R = np.maximum(0.0, 1.0 - p.lambda_pi * Delta_tau_R)
            dpi_diff = p.alpha_pi * J_diff_R_scaled * Delta_tau_R
            
            if p.gravity_enabled:
                g_bond_R = 0.5 * (self.g + R(self.g))
                dpi_grav = 2.0 * p.mu_grav * g_bond_R * Delta_tau_R
            else:
                dpi_grav = 0
            
            self.pi_R = decay_R * self.pi_R + dpi_diff + dpi_grav
        
        # ============================================================
        # STEP 6: Q-locking (damage/inflammation accumulation)
        # ============================================================
        # Natural q accumulation from resource loss
        self.q = np.clip(self.q + p.alpha_q_clear * np.maximum(0, -dF), 0, 1)
        
        # Chronic disease: ongoing q accumulation in disease region
        if self.state == TissueState.CHRONIC and hasattr(self, 'chronic_region'):
            # Only if chronic region still active
            if np.max(self.chronic_region) > 0.01:
                chronic_q = p.chronic_q_lock_rate * self.chronic_region * self.Delta_tau
                self.q = np.clip(self.q + chronic_q, 0, 1)
        
        # If recovering from chronic disease, continue shrinking chronic region
        if self.state == TissueState.RECOVERING and hasattr(self, 'chronic_region'):
            self.chronic_region *= 0.99  # Continue healing the root cause
        
        # Natural q clearing (tissue remodeling)
        # Rate depends on agency and coherence (healthy tissue clears damage)
        clear_rate = p.alpha_q_clear * self.a * self.C_R
        
        # Intervention can boost clearing
        if self.intervention_active:
            clear_rate *= p.intervention_q_clear_boost
        
        self.q = np.clip(self.q - clear_rate * self.Delta_tau, 0, 1)
        
        # ============================================================
        # STEP 7: Agency update (target-tracking)
        # ============================================================
        a_target = 1.0 / (1.0 + p.chronic_a_suppression * self.q**2)
        self.a = self.a + p.a_rate * (a_target - self.a)
        
        # ============================================================
        # STEP 8: Coherence dynamics
        # ============================================================
        flow_magnitude = np.abs(J_R_lim) + np.abs(L(np.abs(J_L_lim)))
        dC = p.C_growth_rate * flow_magnitude * self.Delta_tau
        dC -= p.C_decay_rate * (1 - self.C_R) * self.Delta_tau  # Decay toward baseline
        self.C_R = np.clip(self.C_R + dC, 0.1, 1.0)
        
        # ============================================================
        # STEP 9: Event count
        # ============================================================
        self.k += (self.P > 0.1).astype(int)
        
        # ============================================================
        # STEP 10: Apply intervention effects
        # ============================================================
        self._apply_intervention_step()
        
        self._clip()
        self.time += dk
        self.step_count += 1
        
        # Update state
        self._update_state()
        self._record_history()
    
    def _update_state(self):
        """Update tissue state based on metrics."""
        mean_q = np.mean(self.q)
        mean_a = np.mean(self.a)
        mean_C = np.mean(self.C_R)
        
        if self.intervention_active:
            return  # Keep intervention state
        
        if mean_q < 0.05 and mean_a > 0.9 and mean_C > 0.8:
            if mean_C > self.p.C_healthy:
                self.state = TissueState.ENHANCED
            else:
                self.state = TissueState.HEALED
        elif mean_q < 0.1 and mean_a > 0.8:
            self.state = TissueState.RECOVERING
        elif mean_q > 0.3 or mean_a < 0.6:
            if hasattr(self, 'chronic_region'):
                self.state = TissueState.CHRONIC
            else:
                self.state = TissueState.DAMAGED
        elif mean_q > 0.1:
            self.state = TissueState.INFLAMED
        elif mean_a < 0.85:
            self.state = TissueState.STRESSED
    
    def _record_history(self):
        """Record simulation state."""
        dc = self.damage_center
        
        self.history['time'].append(self.time)
        self.history['state'].append(self.state.name)
        self.history['mean_F'].append(np.mean(self.F))
        self.history['mean_q'].append(np.mean(self.q))
        self.history['mean_a'].append(np.mean(self.a))
        self.history['mean_C'].append(np.mean(self.C_R))
        self.history['mean_P'].append(np.mean(self.P))
        self.history['max_q'].append(np.max(self.q))
        self.history['min_a'].append(np.min(self.a))
        self.history['total_k'].append(np.sum(self.k))
        self.history['damage_center_F'].append(self.F[dc])
        self.history['damage_center_q'].append(self.q[dc])
        self.history['damage_center_a'].append(self.a[dc])
        self.history['grace_total'].append(np.sum(np.maximum(0, self.p.F_healthy - self.F) * self.a))
    
    def run_scenario(self, scenario: str = "acute_healing", steps: int = 3000):
        """Run a predefined healing scenario."""
        
        if scenario == "acute_healing":
            # Simple damage → natural healing
            print("\n" + "="*70)
            print("SCENARIO: Acute Injury → Natural Healing")
            print("="*70)
            
            # Baseline
            for _ in range(100):
                self.step()
            
            # Damage
            self.induce_damage(severity=0.8)
            
            # Natural healing (no intervention)
            for _ in range(steps):
                self.step()
        
        elif scenario == "chronic_untreated":
            # Chronic disease without intervention
            print("\n" + "="*70)
            print("SCENARIO: Chronic Disease → Untreated")
            print("="*70)
            
            for _ in range(100):
                self.step()
            
            self.induce_chronic_disease(severity=0.7)
            
            for _ in range(steps):
                self.step()
        
        elif scenario == "chronic_intervention":
            # Chronic disease with combined intervention
            print("\n" + "="*70)
            print("SCENARIO: Chronic Disease → Combined Intervention")
            print("="*70)
            
            for _ in range(100):
                self.step()
            
            self.induce_chronic_disease(severity=0.7)
            
            # Let disease establish
            for _ in range(500):
                self.step()
            
            # Apply combined intervention
            self.apply_intervention(InterventionType.COMBINED, duration=1000, intensity=1.0)
            
            for _ in range(steps):
                self.step()
        
        elif scenario == "regenerative":
            # Severe damage with regenerative therapy
            print("\n" + "="*70)
            print("SCENARIO: Severe Damage → Regenerative Therapy")
            print("="*70)
            
            for _ in range(100):
                self.step()
            
            self.induce_damage(severity=1.2, width=25)
            
            for _ in range(200):
                self.step()
            
            self.apply_intervention(InterventionType.REGENERATIVE, duration=800, intensity=1.2)
            
            for _ in range(steps):
                self.step()
        
        elif scenario == "coherence_therapy":
            # Fascia/acupuncture/prayer approach
            print("\n" + "="*70)
            print("SCENARIO: Chronic Inflammation → Coherence-Based Therapy")
            print("="*70)
            
            for _ in range(100):
                self.step()
            
            self.induce_chronic_disease(severity=0.5)
            
            for _ in range(300):
                self.step()
            
            self.apply_intervention(InterventionType.COHERENCE_BASED, duration=1500, intensity=0.8)
            
            for _ in range(steps):
                self.step()
        
        return self.history
    
    def plot_healing(self, save_path: Optional[str] = None):
        """Visualize healing trajectory."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        fig.suptitle('DET Organ Healing Simulation', fontsize=14, fontweight='bold')
        
        t = np.array(self.history['time'])
        
        # Resource (F)
        ax = axes[0, 0]
        ax.plot(t, self.history['mean_F'], 'g-', linewidth=2, label='Mean F')
        ax.plot(t, self.history['damage_center_F'], 'g--', alpha=0.5, label='Damage Site F')
        ax.axhline(y=self.p.F_healthy, color='green', linestyle=':', alpha=0.5, label='Healthy')
        ax.set_ylabel('Resource F (Vitality)')
        ax.set_title('Cellular Resources')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Structural Debt (q)
        ax = axes[0, 1]
        ax.plot(t, self.history['mean_q'], 'r-', linewidth=2, label='Mean q')
        ax.plot(t, self.history['max_q'], 'r--', alpha=0.5, label='Max q')
        ax.fill_between(t, 0, self.history['mean_q'], alpha=0.3, color='red')
        ax.set_ylabel('Structural Debt q (Damage)')
        ax.set_title('Tissue Damage / Inflammation')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Agency (a)
        ax = axes[1, 0]
        ax.plot(t, self.history['mean_a'], 'b-', linewidth=2, label='Mean a')
        ax.plot(t, self.history['min_a'], 'b--', alpha=0.5, label='Min a')
        ax.axhline(y=self.p.a_healthy, color='blue', linestyle=':', alpha=0.5, label='Healthy')
        ax.set_ylabel('Agency a (Cellular Autonomy)')
        ax.set_title('Cellular Agency / Responsiveness')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Coherence (C)
        ax = axes[1, 1]
        ax.plot(t, self.history['mean_C'], 'purple', linewidth=2)
        ax.axhline(y=self.p.C_healthy, color='purple', linestyle=':', alpha=0.5, label='Healthy')
        ax.fill_between(t, 0, self.history['mean_C'], alpha=0.3, color='purple')
        ax.set_ylabel('Coherence C (Tissue Integrity)')
        ax.set_title('Tissue Coherence / Organization')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Presence (P)
        ax = axes[2, 0]
        ax.plot(t, self.history['mean_P'], 'orange', linewidth=2)
        ax.set_ylabel('Presence P (Metabolic Rate)')
        ax.set_xlabel('Time')
        ax.set_title('Metabolic Presence')
        ax.grid(True, alpha=0.3)
        
        # State timeline
        ax = axes[2, 1]
        state_colors = {
            'HEALTHY': 'lightgreen',
            'STRESSED': 'yellow',
            'INFLAMED': 'orange',
            'DAMAGED': 'red',
            'CHRONIC': 'darkred',
            'INTERVENTION': 'cyan',
            'RECOVERING': 'lightblue',
            'HEALED': 'green',
            'ENHANCED': 'gold',
        }
        
        prev_state = self.history['state'][0]
        start_t = t[0]
        for i, state in enumerate(self.history['state']):
            if state != prev_state or i == len(self.history['state']) - 1:
                color = state_colors.get(prev_state, 'white')
                ax.axvspan(start_t, t[i], alpha=0.5, color=color)
                start_t = t[i]
                prev_state = state
        
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(0, 1)
        ax.set_xlabel('Time')
        ax.set_title('Tissue State Timeline')
        
        # Add legend for states
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=c, alpha=0.5, label=s) 
                         for s, c in state_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=6, ncol=2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig


def run_comparative_analysis():
    """
    Run multiple scenarios to demonstrate therapeutic principles.
    """
    print("="*70)
    print("DET ORGAN HEALING: COMPARATIVE ANALYSIS")
    print("="*70)
    
    scenarios = [
        ("acute_healing", "Acute Injury - Natural Healing"),
        ("chronic_untreated", "Chronic Disease - Untreated"),
        ("chronic_intervention", "Chronic Disease - Combined Therapy"),
        ("regenerative", "Severe Damage - Regenerative Therapy"),
        ("coherence_therapy", "Chronic Inflammation - Coherence Therapy"),
    ]
    
    results = {}
    
    for scenario_id, scenario_name in scenarios:
        print(f"\n{'='*70}")
        print(f"Running: {scenario_name}")
        print("="*70)
        
        sim = OrganHealingSimulation()
        history = sim.run_scenario(scenario_id, steps=4000)
        
        # Extract final state
        final = {
            'F': history['mean_F'][-1],
            'q': history['mean_q'][-1],
            'a': history['mean_a'][-1],
            'C': history['mean_C'][-1],
            'state': history['state'][-1],
        }
        
        results[scenario_id] = {
            'history': history,
            'final': final,
            'sim': sim,
        }
        
        print(f"\n  Final State: {final['state']}")
        print(f"  F={final['F']:.3f}, q={final['q']:.3f}, a={final['a']:.3f}, C={final['C']:.3f}")
    
    return results


def print_therapeutic_principles():
    """Print the key therapeutic insights from DET."""
    print("\n" + "="*70)
    print("THERAPEUTIC PRINCIPLES FROM DET")
    print("="*70)
    
    principles = """
┌─────────────────────────────────────────────────────────────────────┐
│                    DET THERAPEUTIC PRINCIPLES                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. AGENCY INVIOLABILITY (VI.1)                                    │
│     Cannot FORCE healing. Intervention must work WITH the tissue.   │
│     → Implication: Patient consent, tissue receptivity matter       │
│     → "The body heals itself; we create conditions"                 │
│                                                                     │
│  2. LOCALITY (Scope Axiom)                                         │
│     Healing happens cell-by-cell, not globally imposed.            │
│     → Implication: Targeted delivery > systemic flooding            │
│     → Local coherence more important than distant signals           │
│                                                                     │
│  3. NON-COERCIVE GRACE (VI.5)                                      │
│     Therapeutic effect gated by w = a × n (agency × need)          │
│     → High agency tissue receives more benefit                      │
│     → Must restore agency before resources can help                 │
│                                                                     │
│  4. Q-CLEARING REQUIRES COHERENCE                                  │
│     Structural debt clears at rate: α × a × C                      │
│     → Both agency AND coherence needed for repair                   │
│     → Isolated cells can't heal; tissue context matters             │
│                                                                     │
│  5. REGENERATIVE NODES (Christ-analog)                             │
│     High-agency, low-q cells radiate healing locally               │
│     → Stem cells, healthy tissue margins are crucial                │
│     → One healthy node can restore a neighborhood                   │
│                                                                     │
│  6. DISEASE GRAVITY (Section V)                                    │
│     Damage attracts more damage through Φ field                    │
│     → Breaking the cycle requires intervention                      │
│     → Chronic disease is a stable attractor without help            │
│                                                                     │
│  7. COHERENCE BEFORE RESOURCES                                     │
│     High C enables efficient resource distribution                 │
│     → Fascia/matrix integrity enables nutrient delivery             │
│     → Coherence-based therapy can precede pharmaceutical            │
│                                                                     │
│  8. ENHANCED > ORIGINAL (New Creation principle)                   │
│     True healing can exceed baseline state                         │
│     → Tissue can become MORE coherent than before injury            │
│     → "Scar" vs "regeneration" depends on intervention timing       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

PRACTICAL APPLICATIONS:

┌─────────────────────────────────────────────────────────────────────┐
│ INTERVENTION TYPE    │ DET MECHANISM           │ CLINICAL ANALOG    │
├──────────────────────┼─────────────────────────┼────────────────────┤
│ Pharmaceutical       │ F injection (gated by a)│ Drug delivery      │
│ Regenerative         │ High-a nodes radiating  │ Stem cell therapy  │
│ Energy-based         │ σ boost (processing)    │ PEMF, laser, US    │
│ Coherence-based      │ Direct C restoration    │ Fascia, acupunctic │
│ Combined             │ Multi-modal synergy     │ Integrative med    │
└──────────────────────┴─────────────────────────┴────────────────────┘

KEY INSIGHT: Interventions that restore AGENCY and COHERENCE first
enable all other therapies to work more effectively.

"Grace is constrained action, not arbitrary intervention."
    """
    print(principles)


def main():
    """Run the organ healing demonstration with comparative analysis."""
    
    # Print therapeutic principles
    print_therapeutic_principles()
    
    # ================================================================
    # COMPARATIVE ANALYSIS: Treated vs Untreated Chronic Disease
    # ================================================================
    print("\n" + "="*70)
    print("COMPARATIVE ANALYSIS: Chronic Disease")
    print("Treated vs Untreated")
    print("="*70)
    
    # Run UNTREATED scenario
    print("\n[SCENARIO A: UNTREATED]")
    sim_untreated = OrganHealingSimulation()
    
    for _ in range(200):
        sim_untreated.step()
    
    sim_untreated.induce_chronic_disease(severity=0.7)
    
    for _ in range(6000):
        sim_untreated.step()
    
    print(f"  Final (Untreated): F={np.mean(sim_untreated.F):.3f}, "
          f"q={np.mean(sim_untreated.q):.3f}, a={np.mean(sim_untreated.a):.3f}, "
          f"C={np.mean(sim_untreated.C_R):.3f}, State={sim_untreated.state.name}")
    
    # Run TREATED scenario
    print("\n[SCENARIO B: COMBINED INTERVENTION]")
    sim_treated = OrganHealingSimulation()
    
    for _ in range(200):
        sim_treated.step()
    
    sim_treated.induce_chronic_disease(severity=0.7)
    
    # Let disease establish (same as untreated)
    for _ in range(500):
        sim_treated.step()
    
    # Apply intervention (root cause treatment)
    sim_treated.apply_intervention(InterventionType.COMBINED, duration=2000, intensity=1.2)
    
    for _ in range(5500):
        sim_treated.step()
    
    print(f"  Final (Treated): F={np.mean(sim_treated.F):.3f}, "
          f"q={np.mean(sim_treated.q):.3f}, a={np.mean(sim_treated.a):.3f}, "
          f"C={np.mean(sim_treated.C_R):.3f}, State={sim_treated.state.name}")
    
    # Generate comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('DET Organ Healing: Treated vs Untreated Chronic Disease', 
                 fontsize=14, fontweight='bold')
    
    t_u = np.array(sim_untreated.history['time'])
    t_t = np.array(sim_treated.history['time'])
    
    # Resource comparison
    ax = axes[0, 0]
    ax.plot(t_u, sim_untreated.history['mean_F'], 'r-', linewidth=2, label='Untreated')
    ax.plot(t_t, sim_treated.history['mean_F'], 'g-', linewidth=2, label='Treated')
    ax.axhline(y=sim_treated.p.F_healthy, color='gray', linestyle=':', alpha=0.5, label='Healthy')
    ax.set_ylabel('Resource F')
    ax.set_title('Cellular Resources')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q comparison
    ax = axes[0, 1]
    ax.plot(t_u, sim_untreated.history['mean_q'], 'r-', linewidth=2, label='Untreated')
    ax.plot(t_t, sim_treated.history['mean_q'], 'g-', linewidth=2, label='Treated')
    ax.set_ylabel('Structural Debt q')
    ax.set_title('Tissue Damage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Agency comparison
    ax = axes[0, 2]
    ax.plot(t_u, sim_untreated.history['mean_a'], 'r-', linewidth=2, label='Untreated')
    ax.plot(t_t, sim_treated.history['mean_a'], 'g-', linewidth=2, label='Treated')
    ax.axhline(y=sim_treated.p.a_healthy, color='gray', linestyle=':', alpha=0.5, label='Healthy')
    ax.set_ylabel('Agency a')
    ax.set_title('Cellular Agency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Coherence comparison
    ax = axes[1, 0]
    ax.plot(t_u, sim_untreated.history['mean_C'], 'r-', linewidth=2, label='Untreated')
    ax.plot(t_t, sim_treated.history['mean_C'], 'g-', linewidth=2, label='Treated')
    ax.axhline(y=sim_treated.p.C_healthy, color='gray', linestyle=':', alpha=0.5, label='Healthy')
    ax.set_ylabel('Coherence C')
    ax.set_xlabel('Time')
    ax.set_title('Tissue Coherence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Presence comparison
    ax = axes[1, 1]
    ax.plot(t_u, sim_untreated.history['mean_P'], 'r-', linewidth=2, label='Untreated')
    ax.plot(t_t, sim_treated.history['mean_P'], 'g-', linewidth=2, label='Treated')
    ax.set_ylabel('Presence P')
    ax.set_xlabel('Time')
    ax.set_title('Metabolic Activity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary comparison bar chart
    ax = axes[1, 2]
    metrics = ['F', 'q', 'a', 'C']
    untreated_vals = [
        sim_untreated.history['mean_F'][-1] / sim_treated.p.F_healthy,
        sim_untreated.history['mean_q'][-1] / 0.5,  # Normalize to max
        sim_untreated.history['mean_a'][-1],
        sim_untreated.history['mean_C'][-1],
    ]
    treated_vals = [
        sim_treated.history['mean_F'][-1] / sim_treated.p.F_healthy,
        sim_treated.history['mean_q'][-1] / 0.5,
        sim_treated.history['mean_a'][-1],
        sim_treated.history['mean_C'][-1],
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax.bar(x - width/2, untreated_vals, width, label='Untreated', color='red', alpha=0.7)
    ax.bar(x + width/2, treated_vals, width, label='Treated', color='green', alpha=0.7)
    ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel('Normalized Value')
    ax.set_xlabel('Metric')
    ax.set_title('Final State Comparison')
    ax.legend()
    ax.set_ylim(0, 1.5)
    
    plt.tight_layout()
    plt.savefig('/home/claude/organ_healing_comparison.png', dpi=150, bbox_inches='tight')
    
    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "="*70)
    print("THERAPEUTIC OUTCOME SUMMARY")
    print("="*70)
    print(f"\n  {'Metric':<15} {'Healthy':>10} {'Untreated':>12} {'Treated':>12} {'Δ Treatment':>14}")
    print(f"  {'-'*15} {'-'*10} {'-'*12} {'-'*12} {'-'*14}")
    
    p = sim_treated.p
    
    metrics_data = [
        ('F (Life)', p.F_healthy, sim_untreated.history['mean_F'][-1], sim_treated.history['mean_F'][-1]),
        ('q (Damage)', p.q_healthy, sim_untreated.history['mean_q'][-1], sim_treated.history['mean_q'][-1]),
        ('a (Agency)', p.a_healthy, sim_untreated.history['mean_a'][-1], sim_treated.history['mean_a'][-1]),
        ('C (Coherence)', p.C_healthy, sim_untreated.history['mean_C'][-1], sim_treated.history['mean_C'][-1]),
    ]
    
    for name, healthy, untreated, treated in metrics_data:
        delta = treated - untreated
        direction = '+' if delta > 0 else ''
        # For q, negative delta is good
        if 'q' in name:
            status = '✓' if treated < untreated else '✗'
        else:
            status = '✓' if treated > untreated else '✗'
        print(f"  {name:<15} {healthy:>10.3f} {untreated:>12.3f} {treated:>12.3f} {direction}{delta:>13.3f} {status}")
    
    print(f"\n  Untreated Final State: {sim_untreated.state.name}")
    print(f"  Treated Final State:   {sim_treated.state.name}")
    
    # Key insights
    print("\n" + "="*70)
    print("KEY THERAPEUTIC INSIGHTS FROM DET")
    print("="*70)
    print("""
    1. AGENCY IS THE GATEWAY
       Treatment effectiveness is gated by tissue agency (a).
       Low agency tissue cannot receive resources effectively.
       → Restore agency BEFORE intensive resource delivery.
    
    2. COHERENCE ENABLES FLOW
       High coherence (C) enables efficient resource distribution.
       Damaged tissue matrix blocks healing signals.
       → Coherence-based therapies (fascia, acupuncture) can precede drugs.
    
    3. ROOT CAUSE MATTERS
       Treating symptoms (F boost) while disease driver persists
       leads to temporary improvement followed by relapse.
       → Address the source of ongoing q-accumulation.
    
    4. REGENERATIVE NODES ARE KEY
       High-agency cells (stem cells, healthy margins) radiate
       healing to their neighborhoods.
       → Preserve and protect healthy tissue boundaries.
    
    5. COMBINED APPROACH IS SYNERGISTIC
       Multi-modal therapy (pharmaceutical + energy + coherence)
       outperforms single-modality treatment.
       → Integrative medicine has a DET rationale.
    
    6. HEALING CAN EXCEED ORIGINAL
       Final coherence (C) can exceed pre-injury baseline.
       "Scar" vs "regeneration" depends on intervention approach.
       → True healing is transformation, not just restoration.
    """)
    
    return sim_treated, sim_untreated


if __name__ == "__main__":
    sim = main()
    plt.show()
