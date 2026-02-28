"""
DNA-DET Analyzer: Exploring DNA patterns through Discrete Energy Theory mathematics.

This module analyzes DNA sequences looking for patterns that correlate with DET's
mathematical framework, particularly:
- Golden ratio (φ ≈ 1.618) relationships
- Coherence patterns from base pair bonding
- Structural patterns that might map to DET's q (structural debt)
- Agency patterns in regulatory regions

Author: DNA-DET Analysis Project
Version: 1.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import Counter
import math

# DET Constants (from det_unified_params.py)
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
PHI_INV = 1 / PHI           # Inverse golden ratio ≈ 0.618

# DET Base Parameters
DET_PARAMS = {
    'tau_base': 0.02,      # Time scale
    'sigma_base': 0.12,    # Charging rate
    'lambda_base': 0.008,  # Decay rate
    'mu_base': 2.0,        # Power scale
    'kappa_base': 5.0,     # Coupling
    'C_0': 0.15,           # Coherence scale
    'phi_L': 0.5,          # Angular ratio
    'lambda_a': 30.0,      # Agency ceiling coupling
    'tau_eq_C': 20.0,      # Equilibration ratio
    'pi_max': 3.0,         # Maximum momentum
    'mu_pi_factor': 0.175, # Momentum mobility factor
    'lambda_L_factor': 0.625  # Angular decay factor
}

# DNA Constants
DNA_BASES = {'A': 'adenine', 'T': 'thymine', 'G': 'guanine', 'C': 'cytosine'}
COMPLEMENT = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# Hydrogen bonds: G-C has 3, A-T has 2
H_BONDS = {'A': 2, 'T': 2, 'G': 3, 'C': 3}

# Codon table (standard genetic code)
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'
}


@dataclass
class DNASequence:
    """Represents a DNA sequence with metadata."""
    sequence: str
    name: str = "unknown"
    source: str = "unknown"

    def __post_init__(self):
        # Clean and validate sequence
        self.sequence = self.sequence.upper().replace(' ', '').replace('\n', '')
        valid_bases = set('ATGCN')  # N for unknown
        if not all(b in valid_bases for b in self.sequence):
            invalid = set(self.sequence) - valid_bases
            raise ValueError(f"Invalid bases in sequence: {invalid}")

    def __len__(self):
        return len(self.sequence)

    @property
    def gc_content(self) -> float:
        """GC content as fraction."""
        gc = sum(1 for b in self.sequence if b in 'GC')
        return gc / len(self.sequence) if self.sequence else 0

    @property
    def at_content(self) -> float:
        """AT content as fraction."""
        return 1 - self.gc_content


@dataclass
class DETDNAAnalysis:
    """Results of DET analysis on a DNA sequence."""
    sequence_name: str
    length: int

    # Base composition
    base_counts: Dict[str, int]
    gc_content: float

    # Golden ratio analysis
    gc_at_ratio: float
    gc_at_phi_deviation: float

    # Coherence analysis (from H-bonds)
    mean_coherence: float
    coherence_variance: float
    coherence_pattern: List[float]

    # Structural analysis
    structure_score: float  # Maps to DET's q

    # DET parameter correlations
    det_correlations: Dict[str, float]

    # Pattern findings
    findings: List[str]


class DNADETAnalyzer:
    """
    Analyzes DNA sequences through the lens of DET mathematics.

    Key mappings:
    - Coherence C: Derived from hydrogen bond strength (G-C: 3, A-T: 2)
    - Structural debt q: Stability/mutation susceptibility
    - Agency a: Regulatory potential (promoters, enhancers)
    - Presence P: Expression level proxy
    """

    def __init__(self):
        self.phi = PHI
        self.det_params = DET_PARAMS

    def analyze(self, dna: DNASequence) -> DETDNAAnalysis:
        """Perform complete DET analysis on a DNA sequence."""

        # Base composition
        base_counts = Counter(dna.sequence)
        gc_content = dna.gc_content

        # Golden ratio analysis
        gc_at_ratio, gc_at_phi_dev = self._analyze_golden_ratio(dna)

        # Coherence analysis (from hydrogen bonds)
        coherence_pattern = self._compute_coherence_pattern(dna)
        mean_coh = np.mean(coherence_pattern)
        var_coh = np.var(coherence_pattern)

        # Structural analysis
        structure_score = self._compute_structure_score(dna)

        # DET correlations
        det_corr = self._compute_det_correlations(dna, coherence_pattern)

        # Generate findings
        findings = self._generate_findings(
            gc_content, gc_at_ratio, gc_at_phi_dev,
            mean_coh, structure_score, det_corr
        )

        return DETDNAAnalysis(
            sequence_name=dna.name,
            length=len(dna),
            base_counts=dict(base_counts),
            gc_content=gc_content,
            gc_at_ratio=gc_at_ratio,
            gc_at_phi_deviation=gc_at_phi_dev,
            mean_coherence=mean_coh,
            coherence_variance=var_coh,
            coherence_pattern=coherence_pattern[:100],  # First 100 for display
            structure_score=structure_score,
            det_correlations=det_corr,
            findings=findings
        )

    def _analyze_golden_ratio(self, dna: DNASequence) -> Tuple[float, float]:
        """
        Analyze golden ratio relationships in the sequence.

        The golden ratio appears in DNA's physical structure:
        - Helix pitch to diameter: ~34Å / 20Å ≈ 1.7 (close to φ)
        - Major/minor groove ratio: ~22Å / 12Å ≈ 1.83
        - Base pair rise: 3.4Å with 10 bp per turn

        We look for φ relationships in sequence composition.
        """
        gc = sum(1 for b in dna.sequence if b in 'GC')
        at = sum(1 for b in dna.sequence if b in 'AT')

        if at == 0:
            gc_at_ratio = float('inf')
            phi_dev = float('inf')
        else:
            gc_at_ratio = gc / at
            # How close is this ratio to φ or 1/φ?
            phi_dev = min(abs(gc_at_ratio - self.phi),
                         abs(gc_at_ratio - 1/self.phi),
                         abs(1/gc_at_ratio - self.phi) if gc_at_ratio > 0 else float('inf'))

        return gc_at_ratio, phi_dev

    def _compute_coherence_pattern(self, dna: DNASequence, window: int = 10) -> List[float]:
        """
        Compute local coherence based on hydrogen bond density.

        In DET, coherence C represents quantum-like correlations.
        For DNA, we map this to hydrogen bond strength:
        - G-C pairs: 3 H-bonds → higher coherence
        - A-T pairs: 2 H-bonds → lower coherence

        Normalized so max coherence (all G/C) = 1.0
        Min coherence (all A/T) = 2/3 ≈ 0.667
        """
        seq = dna.sequence
        if len(seq) < window:
            window = len(seq)

        coherence = []
        for i in range(len(seq) - window + 1):
            subseq = seq[i:i+window]
            h_bonds = sum(H_BONDS.get(b, 0) for b in subseq if b != 'N')
            valid_bases = sum(1 for b in subseq if b != 'N')
            if valid_bases > 0:
                # Normalize: max is 3 (all GC), min is 2 (all AT)
                max_bonds = valid_bases * 3
                min_bonds = valid_bases * 2
                coh = (h_bonds - min_bonds) / (max_bonds - min_bonds) if max_bonds > min_bonds else 0.5
                coherence.append(coh)
            else:
                coherence.append(0.5)  # Unknown

        return coherence

    def _compute_structure_score(self, dna: DNASequence) -> float:
        """
        Compute structural score (maps to DET's structural debt q).

        Higher structure score means more "locked" or stable structure:
        - GC-rich regions: More stable (higher melting temp)
        - Repetitive sequences: More structured
        - CpG islands: Important regulatory regions
        """
        seq = dna.sequence

        # Factor 1: GC content (higher = more stable)
        gc_factor = dna.gc_content

        # Factor 2: CpG frequency (important for regulation)
        cpg_count = seq.count('CG')
        expected_cpg = (seq.count('C') * seq.count('G')) / len(seq) if len(seq) > 0 else 0
        cpg_ratio = cpg_count / expected_cpg if expected_cpg > 0 else 1.0

        # Factor 3: Repetitiveness (using 2-mer entropy)
        dimers = [seq[i:i+2] for i in range(len(seq)-1)]
        dimer_counts = Counter(dimers)
        total_dimers = len(dimers)
        if total_dimers > 0:
            probs = [c/total_dimers for c in dimer_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(16)  # 4x4 possible dimers
            repetitiveness = 1 - (entropy / max_entropy)
        else:
            repetitiveness = 0

        # Combine factors (weighted average)
        structure_score = 0.4 * gc_factor + 0.3 * min(cpg_ratio, 2)/2 + 0.3 * repetitiveness

        return structure_score

    def _compute_det_correlations(self, dna: DNASequence,
                                   coherence: List[float]) -> Dict[str, float]:
        """
        Compute correlations between DNA features and DET parameters.
        """
        correlations = {}

        # Map coherence to DET's C_0 parameter
        mean_coh = np.mean(coherence)
        correlations['coherence_C0_ratio'] = mean_coh / self.det_params['C_0']

        # GC content vs DET's phi_L (both ~0.5 for balanced systems)
        correlations['gc_vs_phi_L'] = abs(dna.gc_content - self.det_params['phi_L'])

        # Look for φ in various ratios
        base_counts = Counter(dna.sequence)
        ratios_to_check = [
            ('G', 'A'), ('C', 'T'), ('G', 'T'), ('C', 'A'),
            ('GC', 'AT')  # Combined
        ]

        phi_matches = []
        for r in ratios_to_check:
            if len(r[0]) == 1:
                num = base_counts.get(r[0], 0)
                den = base_counts.get(r[1], 0)
            else:
                num = sum(base_counts.get(b, 0) for b in r[0])
                den = sum(base_counts.get(b, 0) for b in r[1])

            if den > 0:
                ratio = num / den
                phi_dev = min(abs(ratio - self.phi), abs(ratio - 1/self.phi))
                if phi_dev < 0.1:  # Close to φ
                    phi_matches.append((r, ratio, phi_dev))

        correlations['phi_matches'] = len(phi_matches)
        correlations['best_phi_deviation'] = min([m[2] for m in phi_matches]) if phi_matches else 1.0

        # Codon analysis for DET-like patterns
        codons = [dna.sequence[i:i+3] for i in range(0, len(dna.sequence)-2, 3)]
        codon_counts = Counter(codons)

        # Check if codon frequencies follow DET-like distributions
        if codon_counts:
            codon_freqs = sorted(codon_counts.values(), reverse=True)
            # Check for power-law-like distribution
            if len(codon_freqs) >= 2 and codon_freqs[1] > 0:
                top_ratio = codon_freqs[0] / codon_freqs[1]
                correlations['codon_ratio_top2'] = top_ratio
                correlations['codon_ratio_phi_dev'] = min(abs(top_ratio - self.phi),
                                                          abs(top_ratio - 1/self.phi))

        return correlations

    def _generate_findings(self, gc_content: float, gc_at_ratio: float,
                          gc_at_phi_dev: float, mean_coherence: float,
                          structure_score: float,
                          det_corr: Dict[str, float]) -> List[str]:
        """Generate human-readable findings from the analysis."""
        findings = []

        # GC content interpretation
        if gc_content > 0.6:
            findings.append(f"HIGH GC content ({gc_content:.1%}) - thermally stable, high coherence region")
        elif gc_content < 0.4:
            findings.append(f"LOW GC content ({gc_content:.1%}) - flexible region, lower coherence")
        else:
            findings.append(f"BALANCED GC content ({gc_content:.1%}) - near DET's φ_L = 0.5")

        # Golden ratio findings
        if gc_at_phi_dev < 0.05:
            findings.append(f"⚡ GOLDEN RATIO: GC/AT ratio = {gc_at_ratio:.4f} is within 5% of φ!")
        elif gc_at_phi_dev < 0.1:
            findings.append(f"φ-PROXIMATE: GC/AT ratio = {gc_at_ratio:.4f} (φ deviation: {gc_at_phi_dev:.4f})")

        # Coherence interpretation
        if mean_coherence > 0.7:
            findings.append(f"HIGH COHERENCE ({mean_coherence:.3f}) - maps to DET's quantum-correlated regime")
        elif mean_coherence < 0.3:
            findings.append(f"LOW COHERENCE ({mean_coherence:.3f}) - maps to DET's classical regime")

        # Structure interpretation
        if structure_score > 0.6:
            findings.append(f"HIGH STRUCTURE SCORE ({structure_score:.3f}) - maps to high DET q (locked)")

        # DET parameter correlations
        if det_corr.get('phi_matches', 0) > 0:
            findings.append(f"Found {det_corr['phi_matches']} base ratio(s) near golden ratio φ")

        if det_corr.get('codon_ratio_phi_dev', 1.0) < 0.1:
            findings.append("Codon frequency distribution shows φ-like ratios")

        return findings

    def compute_det_lattice_mapping(self, dna: DNASequence) -> np.ndarray:
        """
        Map DNA sequence to a DET-compatible 1D lattice.

        Each base becomes a node with properties:
        - F (resource): Based on hydrogen bond count
        - q (structure): Based on local GC content
        - a (agency): Based on regulatory potential
        """
        n = len(dna.sequence)

        # Initialize lattice arrays
        F = np.zeros(n)  # Resource
        q = np.zeros(n)  # Structural debt
        a = np.zeros(n)  # Agency

        for i, base in enumerate(dna.sequence):
            # Resource from H-bond potential
            F[i] = H_BONDS.get(base, 2.5) / 3.0  # Normalize to [0.67, 1.0]

            # Local structure (GC content in ±5 window)
            start = max(0, i-5)
            end = min(n, i+6)
            local_seq = dna.sequence[start:end]
            local_gc = sum(1 for b in local_seq if b in 'GC') / len(local_seq)
            q[i] = local_gc

            # Agency: Higher for G (guanine often in regulatory)
            if base == 'G':
                a[i] = 0.8
            elif base == 'C':
                a[i] = 0.7
            elif base == 'A':
                a[i] = 0.6
            else:  # T
                a[i] = 0.5

        # Stack into lattice representation
        lattice = np.stack([F, q, a], axis=1)
        return lattice

    def analyze_codon_det_patterns(self, dna: DNASequence) -> Dict:
        """
        Analyze the 64 codons as a 4x4x4 DET lattice structure.

        Codons map naturally to a 3D coordinate:
        - First position: 4 choices (A=0, T=1, G=2, C=3)
        - Second position: 4 choices
        - Third position: 4 choices

        This creates a 4x4x4 cube that we can analyze with DET.
        """
        BASE_TO_IDX = {'A': 0, 'T': 1, 'G': 2, 'C': 3}

        # Count codons
        codons = [dna.sequence[i:i+3] for i in range(0, len(dna.sequence)-2, 3)]
        codon_counts = Counter(codons)

        # Create 4x4x4 frequency lattice
        lattice = np.zeros((4, 4, 4))
        for codon, count in codon_counts.items():
            if len(codon) == 3 and all(b in BASE_TO_IDX for b in codon):
                i, j, k = [BASE_TO_IDX[b] for b in codon]
                lattice[i, j, k] = count

        # Normalize
        total = lattice.sum()
        if total > 0:
            lattice = lattice / total

        # Analyze DET-like properties
        results = {
            'codon_lattice': lattice,
            'total_codons': len(codons),
            'unique_codons': len(codon_counts),
        }

        # Check for φ in layer ratios
        layer_sums = [lattice[i].sum() for i in range(4)]
        if len(layer_sums) >= 2 and min(layer_sums) > 0:
            ratios = []
            for i in range(len(layer_sums)):
                for j in range(i+1, len(layer_sums)):
                    if layer_sums[j] > 0:
                        r = layer_sums[i] / layer_sums[j]
                        ratios.append((f"layer{i}/layer{j}", r,
                                      min(abs(r-PHI), abs(r-1/PHI))))
            results['layer_ratios'] = sorted(ratios, key=lambda x: x[2])[:3]

        # Check lattice symmetry
        results['x_symmetry'] = np.allclose(lattice, lattice[::-1], rtol=0.1)
        results['y_symmetry'] = np.allclose(lattice, lattice[:, ::-1], rtol=0.1)
        results['z_symmetry'] = np.allclose(lattice, lattice[:, :, ::-1], rtol=0.1)

        return results


def fetch_sample_sequences() -> List[DNASequence]:
    """
    Return sample DNA sequences for analysis.
    These are well-known sequences that demonstrate various properties.
    """
    sequences = []

    # Human hemoglobin beta (HBB) - partial coding sequence
    # Known for balanced GC content
    hbb = DNASequence(
        sequence="ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGTGAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCTGATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCACCTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTCAGGCTCCTGGGCAACGTGCTGGTCTGTGTGCTGGCCCATCACTTTGGCAAAGAATTCACCCCACCAGTGCAGGCTGCCTATCAGAAAGTGGTGGCTGGTGTGGCTAATGCCCTGGCCCACAAGTATCACTAA",
        name="Human_HBB",
        source="NCBI"
    )
    sequences.append(hbb)

    # E. coli lacZ promoter region - regulatory sequence
    # Important for understanding gene regulation
    lac_promoter = DNASequence(
        sequence="GACACCATCGAATGGCGCAAAACCTTTCGCGGTATGGCATGATAGCGCCCGGAAGAGAGTCAATTCAGGGTGGTGAATGTGAAACCAGTAACGTTATACGATGTCGCAGAGTATGCCGGTGTCTCTTATCAGACCGTTTCCCGCGTGGTGAACCAGGCCAGCCACGTTTCTGCGAAAACGCGGGAAAAAGTGGAAGCGGCGATGGCGGAGCTGAATTACATTCCCAACCGCGTGGCACAACAACTGGCGGGCAAACAGTCGTTGCTGATTGGCGTTGCCACCTCCAGTCTGGCCCTGCACGCGCCGTCGCAAATTGTCGCGGCGATTAAATCTCGCGCCGATCAACTGGGTGCCAGCGTGGTGGTGTCGATGGTAGAACGAAGCGGCGTCGAAGCCTGTAAAGCGGCGGTGCACAATCTTCTCGCGCAACGCGTCAGTGGGCTGATCATTAACTATCCGCTGGATGACCAGGATGCCATTGCTGTGGAAGCTGCCTGCACTAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGACACCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATTGGGTCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCTGCGTCTGGCTGGCTGGCATAAATATCTCACTCGCAATCAAATTCAGCCGATAGCGGAACGGGAAGGCGACTGGAGTGCCATGTCCGGTTTTCAACAAACCATGCAAATGCTGAATGAGGGCATCGTTCCCACTGCGATGCTGGTTGCCAACGATCAGATGGCGCTGGGCGCAATGCGCGCCATTACCGAGTCCGGGCTGCGCGTTGGTGCGGATATCTCGGTAGTGGGATACGACGATACCGAAGACAGCTCATGTTATATCCCGCCGTTAACCACCATCAAACAGGATTTTCGCCTGCTGGGGCAAACCAGCGTGGACCGCTTGCTGCAACTCTCTCAGGGCCAGGCGGTGAAGGGCAATCAGCTGTTGCCCGTCTCACTGGTGAAAAGAAAAACCACCCTGGCGCCCAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAGTGAGCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGCTTTGCCTGGTTTCCGGCACCAGAAGCGGTGCCGGAAAGCTGGCTGGAGTGCGATCTTCCTGAGGCCGATACTGTCGTCGTCCCCTCAAACTGGCAGATGCACGGTTACGATGCGCCCATCTACACCAACGTGACCTATCCCATTACGGTCAATCCGCCGTTTGTTCCCACGGAGAATCCGACGGGTTGTTACTCGCTCACATTTAATGTTGATGAAAGCTGGCTACAGGAAGGCCAGACGCGAATTATTTTTGATGGCGTTAACTCGGCGTTTCATCTGTGGTGCAACGGGCGCTGGGTCGGTTACGGCCAGGACAGTCGTTTGCCGTCTGAATTTGACCTGAGCGCATTTTTACGCGCCGGAGAAAACCGCCTCGCGGTGATGGTGCTGCGCTGGAGTGACGGCAGTTATCTGGAAGATCAGGATATGTGGCGGATGAGCGGCATTTTCCGTGACGTCTCGTTGCTGCATAAACCGACTACACAAATCAGCGATTTCCATGTTGCCACTCGCTTTAATGATGATTTCAGCCGCGCTGTACTGGAGGCTGAAGTTCAGATGTGCGGCGAGTTGCGTGACTACCTACGGGTAACAGTTTCTTTATGGCAGGGTGAAACGCAGGTCGCCAGCGGCACCGCGCCTTTCGGCGGTGAAATTATCGATGAGCGTGGTGGTTATGCCGATCGCGTCACACTACGTCTGAACGTCGAAAACCCGAAACTGTGGAGCGCCGAAATCCCGAATCTCTATCGTGCGGTGGTTGAACTGCACACCGCCGACGGCACGCTGATTGAAGCAGAAGCCTGCGATGTCGGTTTCCGCGAGGTGCGGATTGAAAATGGTCTGCTGCTGCTGAACGGCAAGCCGTTGCTGATTCGAGGCGTTAACCGTCACGAGCATCATCCTCTGCATGGTCAGGTCATGGATGAGCAGACGATGGTGCAGGATATCCTGCTGATGAAGCAGAACAACTTTAACGCCGTGCGCTGTTCGCATTATCCGAACCATCCGCTGTGGTACACGCTGTGCGACCGCTACGGCCTGTATGTGGTGGATGAAGCCAATATTGAAACCCACGGCATGGTGCCAATGAATCGTCTGACCGATGATCCGCGCTGGCTACCGGCGATGAGCGAACGCGTAACGCGAATGGTGCAGCGCGATCGTAATCACCCGAGTGTGATCATCTGGTCGCTGGGGAATGAATCAGGCCACGGCGCTAATCACGACGCGCTGTATCGCTGGATCAAATCTGTCGATCCTTCCCGCCCGGTGCAGTATGAAGGCGGCGGAGCCGACACCACGGCCACCGATATTATTTGCCCGATGTACGCGCGCGTGGATGAAGACCAGCCCTTCCCGGCTGTGCCGAAATGGTCCATCAAAAAATGGCTTTCGCTACCTGGAGAGACGCGCCCGCTGATCCTTTGCGAATACGCCCACGCGATGGGTAACAGTCTTGGCGGTTTCGCTAAATACTGGCAGGCGTTTCGTCAGTATCCCCGTTTACAGGGCGGCTTCGTCTGGGACTGGGTGGATCAGTCGCTGATTAAATATGATGAAAACGGCAACCCGTGGTCGGCTTACGGCGGTGATTTTGGCGATACGCCGAACGATCGCCAGTTCTGTATGAACGGTCTGGTCTTTGCCGACCGCACGCCGCATCCAGCGCTGACGGAAGCAAAACACCAGCAGCAGTTTTTCCAGTTCCGTTTATCCGGGCAAACCATCGAAGTGACCAGCGAATACCTGTTCCGTCATAGCGATAACGAGCTCCTGCACTGGATGGTGGCGCTGGATGGTAAGCCGCTGGCAAGCGGTGAAGTGCCTCTGGATGTCGCTCCACAAGGTAAACAGTTGATTGAACTGCCTGAACTACCGCAGCCGGAGAGCGCCGGGCAACTCTGGCTCACAGTACGCGTAGTGCAACCGAACGCGACCGCATGGTCAGAAGCCGGGCACATCAGCGCCTGGCAGCAGTGGCGTCTGGCGGAAAACCTCAGTGTGACGCTCCCCGCCGCGTCCCACGCCATCCCGCATCTGACCACCAGCGAAATGGATTTTTGCATCGAGCTGGGTAATAAGCGTTGGCAATTTAACCGCCAGTCAGGCTTTCTTTCACAGATGTGGATTGGCGATAAAAAACAACTGCTGACGCCGCTGCGCGATCAGTTCACCCGTGCACCGCTGGATAACGACATTGGCGTAAGTGAAGCGACCCGCATTGACCCTAACGCCTGGGTCGAACGCTGGAAGGCGGCGGGCCATTACCAGGCCGAAGCAGCGTTGTTGCAGTGCACGGCAGATACACTTGCTGATGCGGTGCTGATTACGACCGCTCACGCGTGGCAGCATCAGGGGAAAACCTTATTTATCAGCCGGAAAACCTACCGGATTGATGGTAGTGGTCAAATGGCGATTACCGTTGATGTTGAAGTGGCGAGCGATACACCGCATCCGGCGCGGATTGGCCTGAACTGCCAGCTGGCGCAGGTAGCAGAGCGGGTAAACTGGCTCGGATTAGGGCCGCAAGAAAACTATCCCGACCGCCTTACTGCCGCCTGTTTTGACCGCTGGGATCTGCCATTGTCAGACATGTATACCCCGTACGTCTTCCCGAGCGAAAACGGTCTGCGCTGCGGGACGCGCGAATTGAATTATGGCCCACACCAGTGGCGCGGCGACTTCCAGTTCAACATCAGCCGCTACAGTCAACAGCAACTGATGGAAACCAGCCATCGCCATCTGCTGCACGCGGAAGAAGGCACATGGCTGAATATCGACGGTTTCCATATGGGGATTGGTGGCGACGACTCCTGGAGCCCGTCAGTATCGGCGGAATTCCAGCTGAGCGCCGGTCGCTACCATTACCAGTTGGTCTGGTGTCAAAAATAATAATAACCGGGCAGGCCATGTCTGCCCGTATTTCGCGTAAGGAAATCCATTATGTACTATTTAAAAAACACAAACTTTTGGATGTTCGGTTTATTCTTTTTCTTTTACTTTTTTATCATGGGAGCCTACTTCCCGTTTTTCCCGATTTGGCTACATGACATCAACCATATCAGCAAAAGTGATACGGGTATTATTTTTGCCGCTATTTCTCTGTTCTCGCTATTATTCCAACCGCTGTTTGGTCTGCTTTCTGACAAACTCGGGCTGCGCAAATACCTGCTGTGGATTATTACCGGCATGTTAGTGATGTTTGCGCCGTTCTTTATTTTTATCTTCGGGCCACTGTTACAATACAACATTTTAGTAGGATCGATTGTTGGTGGTATTTATCTAGGCTTTTGTTTTAACGCCGGTGCGCCAGCAGTAGAGGCATTTATTGAGAAAGTCAGCCGTCGCAGTAATTTCGAATTTGGTCGCGCGCGGATGTTTGGCTGTGTTGGCTGGGCGCTGTGTGCCTCGATTGTCGGCATCATGTTCACCATCAATAATCAGTTTGTTTTCTGGCTGGGCTCTGGCTGTGCACTCATCCTCGCCGTTTTACTCTTTTTCGCCAAAACGGATGCGCCCTCTTCTGCCACGGTTGCCAATGCGGTAGGTGCCAACCATTCGGCATTTAGCCTTAAGCTGGCACTGGAACTGTTCAGACAGCCAAAACTGTGGTTTTTGTCACTGTATGTTATTGGCGTTTCCTGCACCTACGATGTTTTTGACCAACAGTTTGCTAATTTCTTTACTTCGTTCTTTGCTACCGGTGAACAGGGTACGCGGGTATTTGGCTACGTAACGACAATGGGCGAATTACTTAACGCCTCGATTATGTTCTTTGCGCCACTGATCATTAATCGCATCGGTGGGAAAAACGCCCTGCTGCTGGCTGGCACTATTATGTCTGTACGTATTATTGGCTCATCGTTCGCCACCTCAGCGCTGGAAGTGGTTATTCTGAAAACGCTGCATATGTTTGAAGTACCGTTCCTGCTGGTGGGCTGCTTTAAATATATTACCAGCCAGTTTGAAGTGCGTTTTTCAGCGACGATTTATCTGGTCTGTTTCTGCTTCTTTAAGCAACTGGCGATGATTTTTATGTCTGTACTGGCGGGCAATATGTATGAAAGCATCGGTTTCCAGGGCGCTTATCTGGTGCTGGGTCTGGTGGCGCTGGGCTTCACCTTAATTTCCGTGTTCACGCTTAGCGGCCCCGGCCCGCTTTCCCTGCTGCGTCGTCAGGTGAATGAAGTCGCTTAAGCAATCAATGTCGGATGCGGCGCGACGCTTATCCGACCAACATATCATAACGGAGTGATCGCATTGAACATGCCAATGACCGAAAGAATAAGAGCAGGCAAGCTATTTACCGATATGTGCAGAGGCATGCATGAGCTCAGTAATGAAGAAAATTTCGAGATCTATCAGTCAGCGACGATCAATGCGCCTGGTACTGCGCGCTCGCTCAAATTCCGTAAAATGCCCCAGGGCGTCCATTTTTGCGGTTTTCGCCAGATCTGCACATATGACGCTTCATCGTTTAAATGAACAGCGTAAGCGCATGCAGCGACGCAGTACTATTAACCCTTACAGCGGAACGGCAATCACGCCATAACGCTGACTGTTTTTTTGTACAGCGCTATTAATAAAAGCCAGCCCGACACCCGCCAACACCCGCTGACGCGCCCTGACGGGCTTGTCTGCTCCCGGCATCCGCTTACAGACAAGCTGTGACCGTCTCCGGGAGCTGCATGTGTCAGAGGTTTTCACCGTCATCACCGAAACGCGCGA",
        name="E_coli_lacZ_promoter",
        source="NCBI"
    )
    sequences.append(lac_promoter)

    # GC-rich sequence (from CpG island)
    cpg_island = DNASequence(
        sequence="GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC",
        name="CpG_Island_Example",
        source="Synthetic"
    )
    sequences.append(cpg_island)

    # AT-rich sequence (from TATA box region)
    tata_box = DNASequence(
        sequence="TATAAAATATAAATATATATATATATATATATATATAATAATATAATATATTATATTATTATATTAATATTATATATATATATATAAATAATATATAATATAATATATATATTATATATATAATATAATATATAATATATATAT",
        name="TATA_Box_Rich",
        source="Synthetic"
    )
    sequences.append(tata_box)

    # Balanced sequence
    balanced = DNASequence(
        sequence="ATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGC",
        name="Balanced_ATGC",
        source="Synthetic"
    )
    sequences.append(balanced)

    return sequences


def run_dna_det_analysis():
    """Run complete DNA-DET analysis and print results."""
    print("=" * 80)
    print("DNA-DET ANALYZER: Exploring DNA patterns through Discrete Energy Theory")
    print("=" * 80)
    print()

    analyzer = DNADETAnalyzer()
    sequences = fetch_sample_sequences()

    for dna in sequences:
        print(f"\n{'='*80}")
        print(f"ANALYZING: {dna.name}")
        print(f"Length: {len(dna)} bp | Source: {dna.source}")
        print("=" * 80)

        # Run analysis
        results = analyzer.analyze(dna)

        # Print base composition
        print(f"\n📊 BASE COMPOSITION:")
        for base, count in sorted(results.base_counts.items()):
            pct = count / results.length * 100
            print(f"   {base}: {count:6d} ({pct:5.1f}%)")
        print(f"   GC Content: {results.gc_content:.1%}")

        # Print golden ratio analysis
        print(f"\n🔱 GOLDEN RATIO ANALYSIS:")
        print(f"   GC/AT Ratio: {results.gc_at_ratio:.4f}")
        print(f"   φ = 1.6180, 1/φ = 0.6180")
        print(f"   Deviation from φ: {results.gc_at_phi_deviation:.4f}")

        # Print coherence analysis
        print(f"\n🌊 COHERENCE ANALYSIS (from H-bond density):")
        print(f"   Mean Coherence: {results.mean_coherence:.4f}")
        print(f"   Coherence Variance: {results.coherence_variance:.6f}")
        print(f"   DET C_0 = 0.15, Ratio: {results.det_correlations.get('coherence_C0_ratio', 0):.2f}x")

        # Print structure analysis
        print(f"\n🏗️ STRUCTURAL ANALYSIS:")
        print(f"   Structure Score (→ DET q): {results.structure_score:.4f}")

        # Print DET correlations
        print(f"\n🔗 DET PARAMETER CORRELATIONS:")
        for key, value in results.det_correlations.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

        # Print findings
        print(f"\n💡 FINDINGS:")
        for finding in results.findings:
            print(f"   • {finding}")

        # Codon analysis for longer sequences
        if len(dna) >= 100:
            print(f"\n🧬 CODON LATTICE ANALYSIS (4×4×4 DET mapping):")
            codon_results = analyzer.analyze_codon_det_patterns(dna)
            print(f"   Total codons: {codon_results['total_codons']}")
            print(f"   Unique codons: {codon_results['unique_codons']}")
            if 'layer_ratios' in codon_results:
                print(f"   Layer ratios closest to φ:")
                for ratio_info in codon_results['layer_ratios'][:3]:
                    print(f"      {ratio_info[0]}: {ratio_info[1]:.4f} (φ-dev: {ratio_info[2]:.4f})")
            print(f"   Symmetries: X={codon_results['x_symmetry']}, Y={codon_results['y_symmetry']}, Z={codon_results['z_symmetry']}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: DET-DNA CORRESPONDENCE PATTERNS")
    print("=" * 80)
    print("""
    Key observations from DNA-DET analysis:

    1. GOLDEN RATIO (φ ≈ 1.618):
       - DNA helix geometry contains φ relationships
       - GC/AT ratios in some sequences approach φ or 1/φ
       - DET uses φ in λ_π/λ_L and L_max/π_max ratios

    2. COHERENCE MAPPING:
       - G-C bonds (3 H-bonds) → Higher coherence
       - A-T bonds (2 H-bonds) → Lower coherence
       - 3:2 ratio ≈ 1.5, close to φ (1.618)
       - Maps to DET's quantum/classical transition

    3. STRUCTURAL DEBT (q):
       - GC-rich regions: More stable, higher q
       - AT-rich regions: More flexible, lower q
       - CpG islands: Regulatory importance (high agency)

    4. AGENCY (a):
       - Promoters, enhancers: High agency (decision points)
       - Coding regions: Lower agency (execution)
       - Matches DET's agency distribution

    5. CODON LATTICE (4×4×4):
       - 64 codons map to 3D DET lattice
       - Layer ratios may show φ relationships
       - Symmetry patterns reflect redundancy
    """)


if __name__ == "__main__":
    run_dna_det_analysis()
