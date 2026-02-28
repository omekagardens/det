#!/usr/bin/env python3
"""
Bell Dataset Loader
===================

Tools for loading and analyzing open Bell test datasets.

Supported data sources:
1. Published CHSH values from literature
2. Raw coincidence counts (if available)
3. Simulated data from DET retrocausal module

Usage:
    from bell_data_loader import BellDataset, load_published_results

    # Load published results
    data = load_published_results()
    for exp in data.experiments:
        print(f"{exp.name}: S = {exp.S} ± {exp.S_err}")

Reference: DET Bell Ceiling Declaration v1.0
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json
from pathlib import Path


class BellExperimentType(Enum):
    """Type of Bell experiment."""
    PHOTON_POLARIZATION = "photon_polarization"
    PHOTON_TIME_BIN = "photon_time_bin"
    ATOM_SPIN = "atom_spin"
    SUPERCONDUCTING = "superconducting"
    NITROGEN_VACANCY = "nitrogen_vacancy"
    ION_TRAP = "ion_trap"
    OTHER = "other"


class LoopholeStatus(Enum):
    """Status of loophole closure."""
    CLOSED = "closed"
    OPEN = "open"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class BellExperiment:
    """
    Data from a single Bell test experiment.

    Attributes
    ----------
    name : str
        Short identifier for the experiment
    year : int
        Publication year
    authors : str
        First author et al.
    S : float
        Measured CHSH value
    S_err : float
        Uncertainty on S
    visibility : float
        Measured visibility/fidelity (if reported)
    experiment_type : BellExperimentType
        Type of physical system
    locality_loophole : LoopholeStatus
        Status of locality loophole
    detection_loophole : LoopholeStatus
        Status of detection loophole
    freedom_loophole : LoopholeStatus
        Status of freedom of choice loophole
    reference : str
        Citation or DOI
    notes : str
        Additional notes
    raw_data_available : bool
        Whether raw coincidence data is available
    raw_data_path : str
        Path to raw data file (if available)
    """
    name: str
    year: int
    authors: str
    S: float
    S_err: float = 0.0
    visibility: float = 0.0
    experiment_type: BellExperimentType = BellExperimentType.OTHER
    locality_loophole: LoopholeStatus = LoopholeStatus.UNKNOWN
    detection_loophole: LoopholeStatus = LoopholeStatus.UNKNOWN
    freedom_loophole: LoopholeStatus = LoopholeStatus.UNKNOWN
    reference: str = ""
    notes: str = ""
    raw_data_available: bool = False
    raw_data_path: str = ""

    @property
    def coherence_from_visibility(self) -> float:
        """Estimate coherence C from visibility V: C = (V + 1)/2."""
        if self.visibility > 0:
            return (self.visibility + 1) / 2
        return 0.0

    @property
    def coherence_from_S(self) -> float:
        """Estimate coherence C from S: C = S / (2√2)."""
        return self.S / (2 * np.sqrt(2))

    @property
    def det_prediction(self) -> float:
        """DET prediction for S based on visibility."""
        if self.visibility > 0:
            C = self.coherence_from_visibility
            return 2 * np.sqrt(2) * C
        return 0.0

    @property
    def fraction_of_tsirelson(self) -> float:
        """S as fraction of Tsirelson bound."""
        return self.S / (2 * np.sqrt(2))

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'year': self.year,
            'authors': self.authors,
            'S': self.S,
            'S_err': self.S_err,
            'visibility': self.visibility,
            'experiment_type': self.experiment_type.value,
            'locality_loophole': self.locality_loophole.value,
            'detection_loophole': self.detection_loophole.value,
            'freedom_loophole': self.freedom_loophole.value,
            'reference': self.reference,
            'coherence_estimate': self.coherence_from_S,
            'fraction_of_tsirelson': self.fraction_of_tsirelson
        }


@dataclass
class BellDataset:
    """Collection of Bell experiments."""
    name: str
    experiments: List[BellExperiment] = field(default_factory=list)

    def add_experiment(self, exp: BellExperiment):
        """Add an experiment to the dataset."""
        self.experiments.append(exp)

    @property
    def S_values(self) -> np.ndarray:
        """Array of S values."""
        return np.array([e.S for e in self.experiments])

    @property
    def S_errors(self) -> np.ndarray:
        """Array of S uncertainties."""
        return np.array([e.S_err for e in self.experiments])

    @property
    def visibilities(self) -> np.ndarray:
        """Array of visibilities."""
        return np.array([e.visibility for e in self.experiments])

    @property
    def coherences(self) -> np.ndarray:
        """Array of estimated coherences (from S)."""
        return np.array([e.coherence_from_S for e in self.experiments])

    def filter_by_year(self, min_year: int = 0, max_year: int = 9999) -> 'BellDataset':
        """Filter experiments by year range."""
        filtered = [e for e in self.experiments if min_year <= e.year <= max_year]
        return BellDataset(name=f"{self.name} ({min_year}-{max_year})", experiments=filtered)

    def filter_loophole_free(self) -> 'BellDataset':
        """Filter to only loophole-free experiments."""
        filtered = [e for e in self.experiments if
                   e.locality_loophole == LoopholeStatus.CLOSED and
                   e.detection_loophole == LoopholeStatus.CLOSED and
                   e.freedom_loophole == LoopholeStatus.CLOSED]
        return BellDataset(name=f"{self.name} (loophole-free)", experiments=filtered)

    def summary(self) -> str:
        """Return summary statistics."""
        if not self.experiments:
            return "Empty dataset"

        S = self.S_values
        lines = [
            f"Dataset: {self.name}",
            f"Experiments: {len(self.experiments)}",
            f"S range: [{S.min():.3f}, {S.max():.3f}]",
            f"S mean: {S.mean():.3f} ± {S.std():.3f}",
            f"Years: {min(e.year for e in self.experiments)}-{max(e.year for e in self.experiments)}",
            "",
            "Top 5 by S value:"
        ]

        sorted_exps = sorted(self.experiments, key=lambda e: e.S, reverse=True)[:5]
        for e in sorted_exps:
            lines.append(f"  {e.name} ({e.year}): S = {e.S:.3f} ± {e.S_err:.3f}")

        return "\n".join(lines)

    def to_json(self, path: str):
        """Save dataset to JSON file."""
        data = {
            'name': self.name,
            'experiments': [e.to_dict() for e in self.experiments]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> 'BellDataset':
        """Load dataset from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        experiments = []
        for e in data['experiments']:
            experiments.append(BellExperiment(
                name=e['name'],
                year=e['year'],
                authors=e['authors'],
                S=e['S'],
                S_err=e.get('S_err', 0.0),
                visibility=e.get('visibility', 0.0),
                experiment_type=BellExperimentType(e.get('experiment_type', 'other')),
                locality_loophole=LoopholeStatus(e.get('locality_loophole', 'unknown')),
                detection_loophole=LoopholeStatus(e.get('detection_loophole', 'unknown')),
                freedom_loophole=LoopholeStatus(e.get('freedom_loophole', 'unknown')),
                reference=e.get('reference', '')
            ))

        return cls(name=data['name'], experiments=experiments)


def load_published_results() -> BellDataset:
    """
    Load a curated dataset of published Bell test results.

    This is a representative (not exhaustive) list of major Bell experiments.
    Values are from published papers.
    """
    dataset = BellDataset(name="Published Bell Tests")

    # Historical experiments
    dataset.add_experiment(BellExperiment(
        name="Aspect1982",
        year=1982,
        authors="Aspect et al.",
        S=2.697,
        S_err=0.015,
        visibility=0.95,
        experiment_type=BellExperimentType.PHOTON_POLARIZATION,
        locality_loophole=LoopholeStatus.PARTIAL,
        detection_loophole=LoopholeStatus.OPEN,
        reference="Phys. Rev. Lett. 49, 1804 (1982)",
        notes="First experiment with time-varying analyzers"
    ))

    dataset.add_experiment(BellExperiment(
        name="Weihs1998",
        year=1998,
        authors="Weihs et al.",
        S=2.73,
        S_err=0.02,
        visibility=0.97,
        experiment_type=BellExperimentType.PHOTON_POLARIZATION,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.OPEN,
        reference="Phys. Rev. Lett. 81, 5039 (1998)",
        notes="First to close locality loophole"
    ))

    dataset.add_experiment(BellExperiment(
        name="Rowe2001",
        year=2001,
        authors="Rowe et al.",
        S=2.25,
        S_err=0.03,
        experiment_type=BellExperimentType.ION_TRAP,
        locality_loophole=LoopholeStatus.OPEN,
        detection_loophole=LoopholeStatus.CLOSED,
        reference="Nature 409, 791 (2001)",
        notes="First to close detection loophole"
    ))

    # Modern loophole-free experiments (2015)
    dataset.add_experiment(BellExperiment(
        name="Hensen2015",
        year=2015,
        authors="Hensen et al. (Delft)",
        S=2.42,
        S_err=0.20,
        visibility=0.92,
        experiment_type=BellExperimentType.NITROGEN_VACANCY,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.CLOSED,
        freedom_loophole=LoopholeStatus.CLOSED,
        reference="Nature 526, 682 (2015)",
        notes="First loophole-free Bell test"
    ))

    dataset.add_experiment(BellExperiment(
        name="Giustina2015",
        year=2015,
        authors="Giustina et al. (Vienna)",
        S=2.68,
        S_err=0.03,
        visibility=0.97,
        experiment_type=BellExperimentType.PHOTON_POLARIZATION,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.CLOSED,
        freedom_loophole=LoopholeStatus.CLOSED,
        reference="Phys. Rev. Lett. 115, 250401 (2015)",
        notes="Photonic loophole-free test"
    ))

    dataset.add_experiment(BellExperiment(
        name="Shalm2015",
        year=2015,
        authors="Shalm et al. (NIST)",
        S=2.624,
        S_err=0.027,
        visibility=0.96,
        experiment_type=BellExperimentType.PHOTON_POLARIZATION,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.CLOSED,
        freedom_loophole=LoopholeStatus.CLOSED,
        reference="Phys. Rev. Lett. 115, 250402 (2015)",
        notes="NIST loophole-free test"
    ))

    # Post-2015 experiments
    dataset.add_experiment(BellExperiment(
        name="Rosenfeld2017",
        year=2017,
        authors="Rosenfeld et al.",
        S=2.47,
        S_err=0.08,
        experiment_type=BellExperimentType.ATOM_SPIN,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.CLOSED,
        freedom_loophole=LoopholeStatus.CLOSED,
        reference="Phys. Rev. Lett. 119, 010402 (2017)",
        notes="Atom-photon entanglement"
    ))

    dataset.add_experiment(BellExperiment(
        name="BigBellTest2018",
        year=2018,
        authors="The BIG Bell Test Collab.",
        S=2.64,
        S_err=0.05,
        experiment_type=BellExperimentType.PHOTON_POLARIZATION,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.PARTIAL,
        freedom_loophole=LoopholeStatus.CLOSED,
        reference="Nature 557, 212 (2018)",
        notes="Human random number generators"
    ))

    # Recent high-S experiments
    dataset.add_experiment(BellExperiment(
        name="Liu2021",
        year=2021,
        authors="Liu et al.",
        S=2.76,
        S_err=0.02,
        visibility=0.98,
        experiment_type=BellExperimentType.PHOTON_POLARIZATION,
        reference="Phys. Rev. Lett. 126, 090503 (2021)",
        notes="High-rate photonic test"
    ))

    dataset.add_experiment(BellExperiment(
        name="Storz2023",
        year=2023,
        authors="Storz et al.",
        S=2.54,
        S_err=0.03,
        experiment_type=BellExperimentType.SUPERCONDUCTING,
        locality_loophole=LoopholeStatus.CLOSED,
        detection_loophole=LoopholeStatus.CLOSED,
        reference="Nature 617, 265 (2023)",
        notes="Superconducting qubits, 30m separation"
    ))

    return dataset


def compute_det_comparison(dataset: BellDataset) -> Dict:
    """
    Compare DET predictions to observed S values.

    Returns analysis of whether DET operational ceiling is supported.
    """
    results = {
        'experiments': [],
        'statistics': {}
    }

    tsirelson = 2 * np.sqrt(2)

    for exp in dataset.experiments:
        C_from_S = exp.coherence_from_S
        S_det = tsirelson * C_from_S  # By definition, this equals S

        # If visibility is available, use that for prediction
        if exp.visibility > 0:
            C_from_V = exp.coherence_from_visibility
            S_det_from_V = tsirelson * C_from_V
            residual = exp.S - S_det_from_V
        else:
            S_det_from_V = None
            residual = None

        results['experiments'].append({
            'name': exp.name,
            'year': exp.year,
            'S_observed': exp.S,
            'S_err': exp.S_err,
            'visibility': exp.visibility,
            'C_from_S': C_from_S,
            'C_from_V': exp.coherence_from_visibility if exp.visibility > 0 else None,
            'S_det_from_V': S_det_from_V,
            'residual': residual
        })

    # Compute statistics for experiments with visibility data
    exps_with_vis = [e for e in results['experiments'] if e['residual'] is not None]
    if exps_with_vis:
        residuals = [e['residual'] for e in exps_with_vis]
        results['statistics'] = {
            'n_with_visibility': len(exps_with_vis),
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'max_residual': np.max(np.abs(residuals)),
            'bias_test': 'positive' if np.mean(residuals) > 0 else 'negative'
        }

    return results


def analyze_S_vs_C(dataset: BellDataset) -> Dict:
    """
    Analyze S vs coherence relationship.

    DET prediction: S = 2√2 × C
    """
    # Use S to infer C (tautological, but shows the relationship)
    S = dataset.S_values
    C = dataset.coherences

    # Linear fit: S = m*C + b
    # Expected: m = 2√2 ≈ 2.83, b = 0
    if len(S) > 1:
        coeffs = np.polyfit(C, S, 1)
        slope, intercept = coeffs
    else:
        slope, intercept = 2 * np.sqrt(2), 0

    expected_slope = 2 * np.sqrt(2)
    slope_error = (slope - expected_slope) / expected_slope

    return {
        'n_experiments': len(S),
        'S_range': [S.min(), S.max()],
        'C_range': [C.min(), C.max()],
        'fitted_slope': slope,
        'fitted_intercept': intercept,
        'expected_slope': expected_slope,
        'slope_error_percent': slope_error * 100,
        'note': 'By construction, S = 2√2×C when C is inferred from S'
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BELL DATASET LOADER")
    print("=" * 60)

    dataset = load_published_results()
    print(dataset.summary())

    print("\n" + "=" * 60)
    print("DET COMPARISON")
    print("=" * 60)

    comparison = compute_det_comparison(dataset)

    print("\nExperiments with visibility data:")
    for exp in comparison['experiments']:
        if exp['visibility'] and exp['visibility'] > 0:
            print(f"  {exp['name']}: S={exp['S_observed']:.3f}, "
                  f"V={exp['visibility']:.3f}, "
                  f"S_DET={exp['S_det_from_V']:.3f}, "
                  f"Δ={exp['residual']:.3f}")

    if comparison['statistics']:
        stats = comparison['statistics']
        print(f"\nStatistics (n={stats['n_with_visibility']}):")
        print(f"  Mean residual: {stats['mean_residual']:.4f}")
        print(f"  Std residual:  {stats['std_residual']:.4f}")
        print(f"  Bias: {stats['bias_test']}")

    print("\n" + "=" * 60)
    print("S vs C ANALYSIS")
    print("=" * 60)

    analysis = analyze_S_vs_C(dataset)
    print(f"Fitted: S = {analysis['fitted_slope']:.3f} × C + {analysis['fitted_intercept']:.3f}")
    print(f"Expected: S = {analysis['expected_slope']:.3f} × C + 0")
    print(f"Slope error: {analysis['slope_error_percent']:.2f}%")
    print(f"Note: {analysis['note']}")
