"""
DNA-DET Analysis Package

Modules for analyzing DNA sequences through the lens of
Discrete Energy Theory (DET) mathematics.
"""

from .dna_det_analyzer import (
    DNASequence,
    DNADETAnalyzer,
    DETDNAAnalysis,
    fetch_sample_sequences,
    run_dna_det_analysis
)

from .dna_database_fetcher import (
    NCBISequence,
    fetch_ncbi_sequence,
    parse_fasta_file,
    get_sample_sequence,
    list_available_samples,
    get_diversity_sequences,
    FAMOUS_SEQUENCES
)

from .dna_det_deep_analysis import (
    analyze_dna_phi_geometry,
    analyze_amino_acid_det_patterns,
    build_codon_det_lattice,
    analyze_codon_lattice_symmetries,
    analyze_cross_species_det_patterns,
    generate_det_dna_report,
    PHI,
    PHI_INV,
    PHI_SQ,
    DNA_HELIX_PARAMS,
    AMINO_ACIDS,
    CODON_TABLE
)

from .det_subdivision_theory import (
    DETAgent,
    SubdivisionTheory,
    AgencyInheritanceMode,
    DNAReplicationModel,
    analyze_primordial_agency_question,
    run_subdivision_analysis
)

__all__ = [
    # DNA-DET Analyzer
    'DNASequence',
    'DNADETAnalyzer',
    'DETDNAAnalysis',
    'fetch_sample_sequences',
    'run_dna_det_analysis',
    # Database fetcher
    'NCBISequence',
    'fetch_ncbi_sequence',
    'parse_fasta_file',
    'get_sample_sequence',
    'list_available_samples',
    'get_diversity_sequences',
    'FAMOUS_SEQUENCES',
    # Deep analysis
    'analyze_dna_phi_geometry',
    'analyze_amino_acid_det_patterns',
    'build_codon_det_lattice',
    'analyze_codon_lattice_symmetries',
    'analyze_cross_species_det_patterns',
    'generate_det_dna_report',
    'PHI',
    'PHI_INV',
    'PHI_SQ',
    'DNA_HELIX_PARAMS',
    'AMINO_ACIDS',
    'CODON_TABLE',
    # Subdivision theory
    'DETAgent',
    'SubdivisionTheory',
    'AgencyInheritanceMode',
    'DNAReplicationModel',
    'analyze_primordial_agency_question',
    'run_subdivision_analysis'
]
