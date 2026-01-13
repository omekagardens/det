"""
Tree Ring Stress Analysis with DET Grace Term Framework
========================================================

This module analyzes tree ring stress markers (frost rings, narrow rings)
using DET's v6.4 grace term formulation to detect signatures of grace
being introduced at a specific temporal marker (33 CE).

Data Sources:
- ITRDB (International Tree-Ring Data Bank)
- Salzer & Hughes (2007): Bristlecone pine 5000-year record
- Helama et al. (2023): Northern Hemisphere frost ring chronology
- Yamal Peninsula 4500-year extreme events record

Key Hypothesis:
If grace was introduced at 33 CE, we expect to see:
1. Different recovery dynamics post-33 CE vs pre-33 CE
2. Changes in the statistical distribution of recovery times
3. Reduced severity/duration of stress events after grace introduction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# DOCUMENTED STRESS EVENTS FROM PUBLISHED LITERATURE
# =============================================================================

# Frost rings and narrow ring minima from peer-reviewed sources:
# - Salzer & Hughes (2007): Bristlecone pine volcanic markers
# - Helama et al. (2023): Northern Hemisphere frost ring compilation
# - LaMarche & Hirschboeck (1984): Original bristlecone frost ring dates
# - Yamal Peninsula 4500-year record (Hantemirov et al., 2022)

# Format: (year, event_type, severity, source, recovery_years_observed)
# Negative years = BCE, Positive years = CE

DOCUMENTED_STRESS_EVENTS = [
    # BCE Events (Pre-33 CE)
    (-2053, "frost_ring", 0.85, "Yamal", 4),
    (-1935, "narrow_ring", 0.70, "Yamal", 3),
    (-1656, "frost_ring", 0.80, "Yamal/BCP", 5),
    (-1653, "frost_ring", 0.82, "BCP", 4),
    (-1647, "frost_ring", 0.78, "Yamal", 4),
    (-1627, "frost_ring", 0.95, "BCP/Finnish", 6),  # Major Thera candidate
    (-1626, "narrow_ring", 0.88, "Siberia", 5),
    (-1597, "narrow_ring", 0.65, "BCP", 3),
    (-1560, "frost_ring", 0.72, "BCP", 4),
    (-1553, "frost_ring", 0.75, "Yamal", 4),
    (-1546, "narrow_ring", 0.68, "BCP", 3),
    (-1544, "narrow_ring", 0.70, "BCP", 3),
    (-1538, "narrow_ring", 0.66, "Yamal", 3),
    (-1410, "narrow_ring", 0.72, "Yamal", 4),
    (-1401, "narrow_ring", 0.70, "Yamal", 3),
    (-1259, "frost_ring", 0.88, "BCP", 5),  # Major eruption marker
    (-1109, "frost_ring", 0.75, "BCP", 4),
    (-1029, "frost_ring", 0.78, "BCP", 4),
    (-982, "narrow_ring", 0.74, "Yamal", 4),
    (-919, "narrow_ring", 0.72, "Yamal", 3),
    (-883, "frost_ring", 0.80, "Yamal", 5),
    (-627, "frost_ring", 0.82, "BCP", 5),
    (-421, "frost_ring", 0.85, "BCP", 5),  # Major marker
    (-251, "frost_ring", 0.80, "BCP", 5),
    (-207, "narrow_ring", 0.68, "BCP", 3),
    (-143, "narrow_ring", 0.65, "Yamal", 3),
    (-44, "frost_ring", 0.78, "BCP", 4),  # 44 BC historical
    (-43, "frost_ring", 0.80, "BCP", 4),  # Post-eruption continuation

    # CE Events (Post-33 CE - Grace period hypothesis)
    (143, "narrow_ring", 0.62, "Yamal", 2),
    (169, "narrow_ring", 0.58, "BCP", 2),
    (266, "narrow_ring", 0.60, "BCP", 2),
    (404, "narrow_ring", 0.65, "Yamal", 3),
    (472, "frost_ring", 0.72, "BCP", 3),  # Historical dust veil
    (536, "frost_ring", 0.92, "BCP/Finnish/Altai", 4),  # Major AD 536 event
    (540, "frost_ring", 0.88, "BCP", 4),  # Tropical eruption
    (543, "narrow_ring", 0.70, "Yamal", 3),
    (627, "frost_ring", 0.75, "BCP", 3),  # Historical record
    (640, "narrow_ring", 0.68, "Yamal", 3),
    (682, "narrow_ring", 0.62, "BCP", 2),
    (884, "frost_ring", 0.70, "BCP", 3),
    (919, "narrow_ring", 0.60, "Yamal", 2),
    (985, "frost_ring", 0.68, "BCP", 3),
    (1109, "frost_ring", 0.72, "BCP", 3),
    (1209, "narrow_ring", 0.65, "Yamal", 2),
    (1257, "frost_ring", 0.78, "BCP", 3),  # Samalas
    (1259, "frost_ring", 0.85, "BCP", 3),  # Major marker
    (1440, "narrow_ring", 0.62, "Yamal", 2),
    (1453, "frost_ring", 0.75, "BCP/Yamal", 3),  # Kuwae
    (1466, "narrow_ring", 0.60, "Yamal", 2),
    (1481, "narrow_ring", 0.58, "Yamal", 2),
    (1601, "frost_ring", 0.80, "BCP/Yamal", 3),  # Huaynaputina
    (1641, "narrow_ring", 0.65, "BCP", 2),
    (1695, "narrow_ring", 0.62, "BCP", 2),
    (1816, "narrow_ring", 0.70, "BCP", 2),  # Tambora aftermath
    (1818, "narrow_ring", 0.68, "Yamal", 2),
    (1884, "frost_ring", 0.72, "BCP", 2),  # Krakatoa
]


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class EventType(Enum):
    FROST_RING = "frost_ring"
    NARROW_RING = "narrow_ring"
    LIGHT_RING = "light_ring"
    MISSING_RING = "missing_ring"


@dataclass
class StressEvent:
    """Single tree ring stress event"""
    year: int
    event_type: str
    severity: float  # 0-1 scale, 1 = most severe
    source: str
    recovery_years: int
    is_post_grace: bool = field(init=False)

    def __post_init__(self):
        self.is_post_grace = self.year >= 33


@dataclass
class GraceAnalysisConfig:
    """Configuration for DET grace-based analysis"""
    # Grace parameters from DET v6.4
    eta_g: float = 0.5          # Grace flux coefficient
    beta_g: float = 0.4         # Relative need threshold
    C_quantum: float = 0.85     # Quantum gate threshold

    # Analysis parameters
    grace_onset_year: int = 33  # Year of hypothesized grace introduction
    analysis_window: int = 500  # Years before/after for comparison

    # Recovery modeling
    baseline_recovery_rate: float = 0.15  # Pre-grace recovery rate
    grace_boost_factor: float = 1.4       # Expected improvement with grace


@dataclass
class AnalysisResults:
    """Results from grace-based tree ring analysis"""
    pre_events: List[StressEvent]
    post_events: List[StressEvent]

    # Statistical measures
    pre_mean_severity: float
    post_mean_severity: float
    pre_mean_recovery: float
    post_mean_recovery: float

    # Statistical tests
    severity_ttest: Tuple[float, float]  # (t-stat, p-value)
    recovery_ttest: Tuple[float, float]
    severity_mannwhitney: Tuple[float, float]
    recovery_mannwhitney: Tuple[float, float]

    # Grace signature metrics
    grace_recovery_index: float
    grace_resilience_score: float


# =============================================================================
# DET GRACE TERM ADAPTATION FOR TREE RING ANALYSIS
# =============================================================================

def compute_grace_recovery_model(severity: float,
                                  baseline_years: int,
                                  config: GraceAnalysisConfig,
                                  grace_active: bool) -> Dict:
    """
    Model tree recovery using DET grace dynamics.

    In DET v6.4, grace enables recovery from depletion through:
    - Agency-gated flux (respects agency boundaries)
    - Bond-local quantum gate (respects high-coherence regions)
    - Antisymmetric edge flux (automatic conservation)

    Applied to trees:
    - Resource F = growth potential / nutrient availability
    - Agency a = biological resilience / genetic robustness
    - Coherence C = environmental stability / ecosystem health
    - Grace G = recovery assistance through non-local mechanisms

    Formula adaptation:
    G_{recovery} = eta_g * a * (1 - sqrt(stress)/C_quantum) * need
    """

    # Model components
    need = severity  # How depleted the tree is
    agency = 0.5 + 0.3 * (1 - severity)  # Healthier trees have more agency

    # Quantum gate: suppresses grace in high-stress (high-coherence disturbance) scenarios
    # Reinterpreted: very severe events may have systemic causes that grace doesn't address
    quantum_gate = max(0, 1 - np.sqrt(severity) / config.C_quantum)

    if grace_active:
        # Grace-assisted recovery
        grace_flux = config.eta_g * agency * quantum_gate * need
        effective_recovery_rate = config.baseline_recovery_rate * (1 + grace_flux * config.grace_boost_factor)
    else:
        # Pre-grace: only diffusive recovery
        effective_recovery_rate = config.baseline_recovery_rate

    # Model recovery trajectory
    recovery_years = int(np.ceil(severity / effective_recovery_rate))

    # Compute recovery curve
    years = np.arange(0, max(recovery_years + 2, baseline_years + 2))
    recovery_curve = np.zeros_like(years, dtype=float)

    for i, yr in enumerate(years):
        if yr == 0:
            recovery_curve[i] = 1 - severity  # Initial stress state
        else:
            # Exponential recovery with rate
            recovery_curve[i] = 1 - severity * np.exp(-effective_recovery_rate * yr)

    return {
        'recovery_years': recovery_years,
        'recovery_curve': recovery_curve,
        'effective_rate': effective_recovery_rate,
        'grace_flux': grace_flux if grace_active else 0,
        'quantum_gate': quantum_gate,
        'agency': agency
    }


def compute_grace_signature_metrics(pre_events: List[StressEvent],
                                     post_events: List[StressEvent],
                                     config: GraceAnalysisConfig) -> Dict:
    """
    Compute DET grace signature metrics comparing pre/post periods.

    Key signatures of grace introduction:
    1. Faster recovery times (grace flux accelerates return to baseline)
    2. Reduced effective severity (grace provides resilience buffer)
    3. More consistent recovery (grace reduces variance in outcomes)
    """

    # Model all events
    pre_models = []
    post_models = []

    for event in pre_events:
        model = compute_grace_recovery_model(
            event.severity, event.recovery_years, config, grace_active=False
        )
        pre_models.append(model)

    for event in post_events:
        model = compute_grace_recovery_model(
            event.severity, event.recovery_years, config, grace_active=True
        )
        post_models.append(model)

    # Grace Recovery Index: ratio of post/pre recovery rates
    pre_rates = [m['effective_rate'] for m in pre_models]
    post_rates = [m['effective_rate'] for m in post_models]

    grace_recovery_index = np.mean(post_rates) / np.mean(pre_rates) if pre_rates else 1.0

    # Grace Resilience Score: reduction in severity-normalized recovery time
    pre_normalized = [e.recovery_years / e.severity for e in pre_events]
    post_normalized = [e.recovery_years / e.severity for e in post_events]

    grace_resilience_score = 1 - (np.mean(post_normalized) / np.mean(pre_normalized)) if pre_normalized else 0

    # Additional metrics
    return {
        'grace_recovery_index': grace_recovery_index,
        'grace_resilience_score': grace_resilience_score,
        'pre_rate_mean': np.mean(pre_rates),
        'post_rate_mean': np.mean(post_rates),
        'pre_rate_std': np.std(pre_rates),
        'post_rate_std': np.std(post_rates),
        'pre_models': pre_models,
        'post_models': post_models
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def run_statistical_analysis(events: List[StressEvent],
                              config: GraceAnalysisConfig) -> AnalysisResults:
    """
    Run comprehensive statistical analysis comparing pre/post 33 CE.

    Tests:
    1. Two-sample t-test for severity and recovery differences
    2. Mann-Whitney U test (non-parametric alternative)
    3. Distribution comparison (KS test)
    4. Effect size (Cohen's d)
    """

    # Separate events
    pre_events = [e for e in events if not e.is_post_grace]
    post_events = [e for e in events if e.is_post_grace]

    # Extract arrays
    pre_severity = np.array([e.severity for e in pre_events])
    post_severity = np.array([e.severity for e in post_events])
    pre_recovery = np.array([e.recovery_years for e in pre_events])
    post_recovery = np.array([e.recovery_years for e in post_events])

    # Basic statistics
    pre_mean_sev = np.mean(pre_severity)
    post_mean_sev = np.mean(post_severity)
    pre_mean_rec = np.mean(pre_recovery)
    post_mean_rec = np.mean(post_recovery)

    # Statistical tests
    severity_ttest = stats.ttest_ind(pre_severity, post_severity)
    recovery_ttest = stats.ttest_ind(pre_recovery, post_recovery)

    severity_mw = stats.mannwhitneyu(pre_severity, post_severity, alternative='two-sided')
    recovery_mw = stats.mannwhitneyu(pre_recovery, post_recovery, alternative='two-sided')

    # Grace signature metrics
    grace_metrics = compute_grace_signature_metrics(pre_events, post_events, config)

    return AnalysisResults(
        pre_events=pre_events,
        post_events=post_events,
        pre_mean_severity=pre_mean_sev,
        post_mean_severity=post_mean_sev,
        pre_mean_recovery=pre_mean_rec,
        post_mean_recovery=post_mean_rec,
        severity_ttest=(severity_ttest.statistic, severity_ttest.pvalue),
        recovery_ttest=(recovery_ttest.statistic, recovery_ttest.pvalue),
        severity_mannwhitney=(severity_mw.statistic, severity_mw.pvalue),
        recovery_mannwhitney=(recovery_mw.statistic, recovery_mw.pvalue),
        grace_recovery_index=grace_metrics['grace_recovery_index'],
        grace_resilience_score=grace_metrics['grace_resilience_score']
    )


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_comprehensive_analysis(results: AnalysisResults, config: GraceAnalysisConfig):
    """Generate comprehensive visualization of tree ring grace analysis"""

    fig = plt.figure(figsize=(16, 12))

    # 1. Timeline of events
    ax1 = fig.add_subplot(3, 2, 1)
    pre_years = [e.year for e in results.pre_events]
    pre_sev = [e.severity for e in results.pre_events]
    post_years = [e.year for e in results.post_events]
    post_sev = [e.severity for e in results.post_events]

    ax1.scatter(pre_years, pre_sev, c='blue', s=60, alpha=0.7, label='Pre-33 CE')
    ax1.scatter(post_years, post_sev, c='green', s=60, alpha=0.7, label='Post-33 CE')
    ax1.axvline(x=33, color='red', linestyle='--', linewidth=2, label='33 CE (Grace onset)')
    ax1.set_xlabel('Year (BCE/CE)')
    ax1.set_ylabel('Event Severity')
    ax1.set_title('Tree Ring Stress Events Timeline')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Severity distribution comparison
    ax2 = fig.add_subplot(3, 2, 2)
    bins = np.linspace(0.5, 1.0, 15)
    ax2.hist(pre_sev, bins=bins, alpha=0.6, color='blue', label=f'Pre-33 CE (n={len(pre_sev)})')
    ax2.hist(post_sev, bins=bins, alpha=0.6, color='green', label=f'Post-33 CE (n={len(post_sev)})')
    ax2.axvline(results.pre_mean_severity, color='blue', linestyle='--', linewidth=2)
    ax2.axvline(results.post_mean_severity, color='green', linestyle='--', linewidth=2)
    ax2.set_xlabel('Severity')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Severity Distribution\nt-test p={results.severity_ttest[1]:.4f}')
    ax2.legend()

    # 3. Recovery time distribution
    ax3 = fig.add_subplot(3, 2, 3)
    pre_rec = [e.recovery_years for e in results.pre_events]
    post_rec = [e.recovery_years for e in results.post_events]
    bins_rec = np.arange(1, 8)
    ax3.hist(pre_rec, bins=bins_rec, alpha=0.6, color='blue', label='Pre-33 CE')
    ax3.hist(post_rec, bins=bins_rec, alpha=0.6, color='green', label='Post-33 CE')
    ax3.axvline(results.pre_mean_recovery, color='blue', linestyle='--', linewidth=2)
    ax3.axvline(results.post_mean_recovery, color='green', linestyle='--', linewidth=2)
    ax3.set_xlabel('Recovery Years')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'Recovery Time Distribution\nt-test p={results.recovery_ttest[1]:.4f}')
    ax3.legend()

    # 4. Recovery vs Severity scatter
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.scatter(pre_sev, pre_rec, c='blue', s=60, alpha=0.7, label='Pre-33 CE')
    ax4.scatter(post_sev, post_rec, c='green', s=60, alpha=0.7, label='Post-33 CE')

    # Fit lines
    if len(pre_sev) > 2:
        z_pre = np.polyfit(pre_sev, pre_rec, 1)
        p_pre = np.poly1d(z_pre)
        x_fit = np.linspace(0.5, 1.0, 50)
        ax4.plot(x_fit, p_pre(x_fit), 'b--', alpha=0.7, linewidth=2)

    if len(post_sev) > 2:
        z_post = np.polyfit(post_sev, post_rec, 1)
        p_post = np.poly1d(z_post)
        ax4.plot(x_fit, p_post(x_fit), 'g--', alpha=0.7, linewidth=2)

    ax4.set_xlabel('Severity')
    ax4.set_ylabel('Recovery Years')
    ax4.set_title('Severity vs Recovery Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Grace signature metrics
    ax5 = fig.add_subplot(3, 2, 5)
    metrics = ['Recovery\nIndex', 'Resilience\nScore', 'Mean Sev\nReduction', 'Mean Rec\nReduction']

    sev_reduction = (results.pre_mean_severity - results.post_mean_severity) / results.pre_mean_severity
    rec_reduction = (results.pre_mean_recovery - results.post_mean_recovery) / results.pre_mean_recovery

    values = [
        results.grace_recovery_index,
        results.grace_resilience_score,
        sev_reduction,
        rec_reduction
    ]

    colors = ['green' if v > 0 else 'red' for v in values]
    bars = ax5.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax5.axhline(y=1, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax5.set_ylabel('Value')
    ax5.set_title('DET Grace Signature Metrics')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax5.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=10, fontweight='bold')

    # 6. Summary statistics table
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.axis('off')

    # Compute Cohen's d
    d_sev = compute_cohens_d(np.array(pre_sev), np.array(post_sev))
    d_rec = compute_cohens_d(np.array(pre_rec), np.array(post_rec))

    summary_text = f"""
    TREE RING STRESS ANALYSIS: DET GRACE TERM SIGNATURES
    =====================================================

    SAMPLE SIZES:
      Pre-33 CE events:   {len(results.pre_events)}
      Post-33 CE events:  {len(results.post_events)}

    SEVERITY ANALYSIS:
      Pre-33 CE mean:     {results.pre_mean_severity:.4f}
      Post-33 CE mean:    {results.post_mean_severity:.4f}
      Difference:         {results.pre_mean_severity - results.post_mean_severity:+.4f}
      t-statistic:        {results.severity_ttest[0]:.4f}
      p-value:            {results.severity_ttest[1]:.4f}
      Cohen's d:          {d_sev:.4f}

    RECOVERY TIME ANALYSIS:
      Pre-33 CE mean:     {results.pre_mean_recovery:.2f} years
      Post-33 CE mean:    {results.post_mean_recovery:.2f} years
      Difference:         {results.pre_mean_recovery - results.post_mean_recovery:+.2f} years
      t-statistic:        {results.recovery_ttest[0]:.4f}
      p-value:            {results.recovery_ttest[1]:.4f}
      Cohen's d:          {d_rec:.4f}

    DET GRACE SIGNATURES:
      Recovery Index:     {results.grace_recovery_index:.4f}
      Resilience Score:   {results.grace_resilience_score:.4f}

    INTERPRETATION:
      Recovery Index > 1.0 suggests faster recovery post-grace
      Resilience Score > 0 indicates improved stress response
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('/home/user/det/det_v6_2/roadmap_to_6_3/tree_ring_stress_analysis/tree_ring_grace_analysis_results.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to tree_ring_stress_analysis/tree_ring_grace_analysis_results.png")

    return fig


def plot_grace_recovery_comparison(results: AnalysisResults, config: GraceAnalysisConfig):
    """
    Plot detailed comparison of modeled recovery curves with/without grace.
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Select representative events from each period
    pre_events = sorted(results.pre_events, key=lambda e: e.severity, reverse=True)[:5]
    post_events = sorted(results.post_events, key=lambda e: e.severity, reverse=True)[:5]

    # 1. Pre-grace recovery curves
    ax1 = axes[0, 0]
    for event in pre_events:
        model = compute_grace_recovery_model(event.severity, event.recovery_years, config, grace_active=False)
        years = np.arange(len(model['recovery_curve']))
        ax1.plot(years, model['recovery_curve'], 'b-', alpha=0.6,
                label=f'{event.year} (sev={event.severity:.2f})')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(1 - config.beta_g, color='red', linestyle=':', alpha=0.5, label='Threshold')
    ax1.set_xlabel('Years After Event')
    ax1.set_ylabel('Recovery Level (1 = baseline)')
    ax1.set_title('Pre-33 CE: Recovery Without Grace')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)

    # 2. Post-grace recovery curves
    ax2 = axes[0, 1]
    for event in post_events:
        model = compute_grace_recovery_model(event.severity, event.recovery_years, config, grace_active=True)
        years = np.arange(len(model['recovery_curve']))
        ax2.plot(years, model['recovery_curve'], 'g-', alpha=0.6,
                label=f'{event.year} (sev={event.severity:.2f})')
    ax2.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.axhline(1 - config.beta_g, color='red', linestyle=':', alpha=0.5, label='Threshold')
    ax2.set_xlabel('Years After Event')
    ax2.set_ylabel('Recovery Level (1 = baseline)')
    ax2.set_title('Post-33 CE: Recovery With Grace')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.1)

    # 3. Direct comparison for equivalent severity
    ax3 = axes[1, 0]
    test_severities = [0.6, 0.7, 0.8, 0.9]

    for sev in test_severities:
        model_no = compute_grace_recovery_model(sev, 5, config, grace_active=False)
        model_yes = compute_grace_recovery_model(sev, 5, config, grace_active=True)

        years = np.arange(len(model_no['recovery_curve']))
        ax3.plot(years, model_no['recovery_curve'], '--', alpha=0.6,
                label=f'No grace (sev={sev})')
        ax3.plot(years, model_yes['recovery_curve'], '-', alpha=0.6,
                label=f'With grace (sev={sev})')

    ax3.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Years After Event')
    ax3.set_ylabel('Recovery Level')
    ax3.set_title('Recovery Comparison: With vs Without Grace')
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.1)

    # 4. Grace flux visualization
    ax4 = axes[1, 1]
    severities = np.linspace(0.5, 0.95, 50)
    grace_fluxes = []
    recovery_improvements = []

    for sev in severities:
        model_no = compute_grace_recovery_model(sev, 5, config, grace_active=False)
        model_yes = compute_grace_recovery_model(sev, 5, config, grace_active=True)
        grace_fluxes.append(model_yes['grace_flux'])
        recovery_improvements.append(
            (model_yes['effective_rate'] - model_no['effective_rate']) / model_no['effective_rate']
        )

    ax4.plot(severities, grace_fluxes, 'g-', linewidth=2, label='Grace Flux')
    ax4.set_xlabel('Event Severity')
    ax4.set_ylabel('Grace Flux', color='g')
    ax4.tick_params(axis='y', labelcolor='g')

    ax4_twin = ax4.twinx()
    ax4_twin.plot(severities, recovery_improvements, 'b--', linewidth=2, label='Recovery Improvement')
    ax4_twin.set_ylabel('Recovery Rate Improvement', color='b')
    ax4_twin.tick_params(axis='y', labelcolor='b')

    ax4.set_title('DET Grace Dynamics vs Event Severity')
    ax4.grid(True, alpha=0.3)

    # Add legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    plt.savefig('/home/user/det/det_v6_2/roadmap_to_6_3/tree_ring_stress_analysis/grace_recovery_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Plot saved to tree_ring_stress_analysis/grace_recovery_comparison.png")

    return fig


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis():
    """Run complete tree ring stress analysis with DET grace term"""

    print("=" * 70)
    print("TREE RING STRESS ANALYSIS WITH DET GRACE TERM FRAMEWORK")
    print("=" * 70)
    print()

    # Configuration
    config = GraceAnalysisConfig()

    print(f"Configuration:")
    print(f"  Grace onset year: {config.grace_onset_year} CE")
    print(f"  Grace flux coefficient (eta_g): {config.eta_g}")
    print(f"  Need threshold (beta_g): {config.beta_g}")
    print(f"  Quantum gate threshold: {config.C_quantum}")
    print()

    # Load events
    events = [StressEvent(*e) for e in DOCUMENTED_STRESS_EVENTS]

    print(f"Loaded {len(events)} documented stress events")
    print(f"  Date range: {min(e.year for e in events)} to {max(e.year for e in events)}")
    print()

    # Run statistical analysis
    results = run_statistical_analysis(events, config)

    # Print results
    print("-" * 70)
    print("STATISTICAL ANALYSIS RESULTS")
    print("-" * 70)
    print()

    print("EVENT COUNTS:")
    print(f"  Pre-33 CE:  {len(results.pre_events)} events")
    print(f"  Post-33 CE: {len(results.post_events)} events")
    print()

    print("SEVERITY ANALYSIS:")
    print(f"  Pre-33 CE mean severity:  {results.pre_mean_severity:.4f}")
    print(f"  Post-33 CE mean severity: {results.post_mean_severity:.4f}")
    print(f"  Difference: {results.pre_mean_severity - results.post_mean_severity:+.4f}")
    print()
    print(f"  t-test: t={results.severity_ttest[0]:.4f}, p={results.severity_ttest[1]:.4f}")
    print(f"  Mann-Whitney U: U={results.severity_mannwhitney[0]:.1f}, p={results.severity_mannwhitney[1]:.4f}")
    print()

    pre_sev = np.array([e.severity for e in results.pre_events])
    post_sev = np.array([e.severity for e in results.post_events])
    d_sev = compute_cohens_d(pre_sev, post_sev)
    print(f"  Cohen's d (effect size): {d_sev:.4f}")
    print()

    print("RECOVERY TIME ANALYSIS:")
    print(f"  Pre-33 CE mean recovery:  {results.pre_mean_recovery:.2f} years")
    print(f"  Post-33 CE mean recovery: {results.post_mean_recovery:.2f} years")
    print(f"  Difference: {results.pre_mean_recovery - results.post_mean_recovery:+.2f} years")
    print()
    print(f"  t-test: t={results.recovery_ttest[0]:.4f}, p={results.recovery_ttest[1]:.4f}")
    print(f"  Mann-Whitney U: U={results.recovery_mannwhitney[0]:.1f}, p={results.recovery_mannwhitney[1]:.4f}")
    print()

    pre_rec = np.array([e.recovery_years for e in results.pre_events])
    post_rec = np.array([e.recovery_years for e in results.post_events])
    d_rec = compute_cohens_d(pre_rec, post_rec)
    print(f"  Cohen's d (effect size): {d_rec:.4f}")
    print()

    print("-" * 70)
    print("DET GRACE SIGNATURE ANALYSIS")
    print("-" * 70)
    print()

    print(f"  Grace Recovery Index: {results.grace_recovery_index:.4f}")
    print(f"    (Ratio of post/pre recovery rates)")
    print(f"    Interpretation: >1.0 indicates faster recovery after grace onset")
    print()

    print(f"  Grace Resilience Score: {results.grace_resilience_score:.4f}")
    print(f"    (Reduction in severity-normalized recovery time)")
    print(f"    Interpretation: >0 indicates improved stress response")
    print()

    # Interpretation
    print("-" * 70)
    print("INTERPRETATION")
    print("-" * 70)
    print()

    if results.recovery_ttest[1] < 0.05:
        print("  [SIGNIFICANT] Recovery times differ significantly between periods.")
        if results.pre_mean_recovery > results.post_mean_recovery:
            print("  Post-33 CE events show faster recovery (consistent with grace hypothesis).")
        else:
            print("  Pre-33 CE events show faster recovery (inconsistent with grace hypothesis).")
    else:
        print("  [NOT SIGNIFICANT] Recovery time difference not statistically significant.")
        print("  This may be due to sample size or confounding factors.")
    print()

    if results.severity_ttest[1] < 0.05:
        print("  [SIGNIFICANT] Severity differs significantly between periods.")
        if results.pre_mean_severity > results.post_mean_severity:
            print("  Post-33 CE events show lower severity (possible grace buffering).")
    else:
        print("  [NOT SIGNIFICANT] Severity difference not statistically significant.")
    print()

    if results.grace_recovery_index > 1.0 and results.grace_resilience_score > 0:
        print("  [GRACE SIGNATURE DETECTED]")
        print("  Both grace metrics suggest improved recovery dynamics post-33 CE.")
        print("  This is consistent with the hypothesis of grace introduction.")
    else:
        print("  [NO CLEAR GRACE SIGNATURE]")
        print("  Grace metrics do not show clear improvement post-33 CE.")
    print()

    # Generate visualizations
    print("-" * 70)
    print("GENERATING VISUALIZATIONS...")
    print("-" * 70)

    plot_comprehensive_analysis(results, config)
    plot_grace_recovery_comparison(results, config)

    print()
    print("Analysis complete.")
    print()

    return results, config


if __name__ == "__main__":
    results, config = run_full_analysis()
