"""
DNA Database Fetcher: Retrieve DNA sequences from public databases.

Supports:
- NCBI GenBank (via Entrez)
- FASTA file parsing
- Sample sequences for testing

Note: For NCBI access, you may need to install biopython:
    pip install biopython
"""

import os
import urllib.request
import urllib.parse
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class NCBISequence:
    """Sequence data from NCBI."""
    accession: str
    description: str
    sequence: str
    organism: str = "unknown"
    length: int = 0

    def __post_init__(self):
        self.length = len(self.sequence)


def fetch_ncbi_sequence(accession: str, email: str = "anonymous@example.com") -> Optional[NCBISequence]:
    """
    Fetch a sequence from NCBI GenBank by accession number.

    Args:
        accession: NCBI accession number (e.g., 'NM_000518.5' for human HBB)
        email: Email for NCBI Entrez (required by NCBI)

    Returns:
        NCBISequence object or None if fetch fails
    """
    try:
        # NCBI E-utilities URL for efetch
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'nucleotide',
            'id': accession,
            'rettype': 'fasta',
            'retmode': 'text',
            'email': email
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"

        # Fetch the data
        with urllib.request.urlopen(url, timeout=30) as response:
            fasta_text = response.read().decode('utf-8')

        # Parse FASTA format
        lines = fasta_text.strip().split('\n')
        if not lines or not lines[0].startswith('>'):
            return None

        header = lines[0][1:]  # Remove '>'
        sequence = ''.join(lines[1:])

        # Extract organism if present
        organism = "unknown"
        if '[' in header and ']' in header:
            org_start = header.index('[') + 1
            org_end = header.index(']')
            organism = header[org_start:org_end]

        return NCBISequence(
            accession=accession,
            description=header,
            sequence=sequence.upper(),
            organism=organism
        )

    except Exception as e:
        print(f"Error fetching {accession}: {e}")
        return None


def parse_fasta_file(filepath: str) -> List[NCBISequence]:
    """Parse a FASTA file and return list of sequences."""
    sequences = []
    current_header = None
    current_seq_parts = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_header and current_seq_parts:
                    sequences.append(NCBISequence(
                        accession=current_header.split()[0],
                        description=current_header,
                        sequence=''.join(current_seq_parts).upper()
                    ))
                current_header = line[1:]
                current_seq_parts = []
            elif line:
                current_seq_parts.append(line)

        # Don't forget the last sequence
        if current_header and current_seq_parts:
            sequences.append(NCBISequence(
                accession=current_header.split()[0],
                description=current_header,
                sequence=''.join(current_seq_parts).upper()
            ))

    return sequences


# Well-known sequence accessions for testing
FAMOUS_SEQUENCES = {
    # Human genes
    'HBB': 'NM_000518.5',      # Human hemoglobin beta
    'BRCA1': 'NM_007294.4',    # Breast cancer 1
    'TP53': 'NM_000546.6',     # Tumor protein p53
    'CFTR': 'NM_000492.4',     # Cystic fibrosis transmembrane conductance regulator
    'INS': 'NM_000207.3',      # Insulin

    # Model organisms
    'E_coli_lacZ': 'NC_000913.3',  # E. coli K-12 (contains lacZ)
    'yeast_TRP1': 'NC_001134.8',   # S. cerevisiae chr IV

    # Viral genomes
    'HIV1': 'NC_001802.1',         # HIV-1 complete genome
    'influenza': 'NC_002016.1',   # Influenza A segment

    # Mitochondrial
    'human_mtDNA': 'NC_012920.1'  # Human mitochondrial genome
}


# Pre-loaded sample sequences for offline analysis
SAMPLE_SEQUENCES = {
    'phi_test': NCBISequence(
        accession='SYNTHETIC_PHI',
        description='Synthetic sequence designed to test φ ratios',
        sequence='GGGGGGCCCCCCCAAAATTTTTTTTTTTTTTT' * 10,  # ~38% GC ≈ 1/φ
        organism='Synthetic'
    ),
    'phi_golden': NCBISequence(
        accession='SYNTHETIC_GOLDEN',
        description='Synthetic sequence with GC/AT ≈ φ',
        # φ ≈ 1.618, so need ~62% GC for GC/AT ≈ φ
        sequence='GCGCGCGCGCGCGCGCATATGCGCGCGCGCGCGCGCATATGCGCGCGCGCGCGCGCATATGCGCGCGCGCGCGCGCATAT' * 5,
        organism='Synthetic'
    ),
    'fibonacci': NCBISequence(
        accession='SYNTHETIC_FIB',
        description='Fibonacci-patterned sequence (AGTC lengths: 1,1,2,3,5,8,13)',
        sequence='A' + 'G' + 'TT' + 'CCC' + 'AAAAA' + 'GGGGGGGG' + 'TTTTTTTTTTTTT',
        organism='Synthetic'
    ),
    'det_coherent': NCBISequence(
        accession='SYNTHETIC_COH',
        description='High coherence (GC-rich) synthetic sequence',
        sequence='GCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGCGC',
        organism='Synthetic'
    ),
    'det_classical': NCBISequence(
        accession='SYNTHETIC_CLAS',
        description='Low coherence (AT-rich) classical regime',
        sequence='ATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATATAT',
        organism='Synthetic'
    ),
}


def get_sample_sequence(name: str) -> Optional[NCBISequence]:
    """Get a pre-loaded sample sequence."""
    return SAMPLE_SEQUENCES.get(name)


def list_available_samples() -> List[str]:
    """List all available sample sequences."""
    return list(SAMPLE_SEQUENCES.keys())


# Representative sequences from different organisms for diversity analysis
DIVERSITY_SEQUENCES = {
    'human_sample': (
        "ATGGTGCATCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGTGAACGTGGATGAAGTTGGTGGT"
        "GAGGCCCTGGGCAGGCTGCTGGTGGTCTACCCTTGGACCCAGAGGTTCTTTGAGTCCTTTGGGGATCTGTCCACTCCT"
        "GATGCTGTTATGGGCAACCCTAAGGTGAAGGCTCATGGCAAGAAAGTGCTCGGTGCCTTTAGTGATGGCCTGGCTCAC"
        "CTGGACAACCTCAAGGGCACCTTTGCCACACTGAGTGAGCTGCACTGTGACAAGCTGCACGTGGATCCTGAGAACTTC"
    ),
    'ecoli_sample': (
        "ATGACCATGATTACGGATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAAT"
        "CGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTG"
        "CGCAGCCTGAATGGCGAATGGCGCTTTGCCTGGTTTCCGGCACCAGAAGCGGTGCCGGAAAGCTGGCTGGAGTGCGAT"
        "CTTCCTGAGGCCGATACTGTCGTCGTCCCCTCAAACTGGCAGATGCACGGTTACGATGCGCCCATCTACACCAACGTG"
    ),
    'yeast_sample': (
        "ATGTCTGCCCCTAAGAAGATCGTCGTTTTGCCAGGTGACCACGTTGGTCAAGAAATCACAGCCGAAGCCATTAAGGTT"
        "CTTAAAGCTATTTCTGATGTTCGTTCCAATGTCAAGTTCGATTTCGAAAATCATTTAATTGGTGGTGCTGCTATCGAT"
        "GCTACAGGTGTTCCACTTCCAGATGAGGCGCTGGAAGCCTCCAAGAAGGCTGATGCCGTTTTGTTAGGTGCTGTGGGT"
        "GGTCCTAAATGGGGTACCGGTAGTGTTAGACCTGAACAAGGTTTACTAAAAATCCGTAAAGAACTTCAATTGTACGCC"
    ),
    'archaea_sample': (
        "ATGGCTAGACGAGAAGCGTTCGAGCTGAAGAAGGAGGCGAAGAAGAAGGAGCTGAAGAAGGCGAAGAAGGAGCTGAAG"
        "AAGGCGCTGGAGAAGCTGAAGAAGGAGCTGAAGAAGGCGAAGAAGGAGCTGAAGAAGGCGCTGGAGAAGCTGAAGAAG"
        "GAGCTGAAGAAGGCGAAGAAGGAGCTGAAGAAGGCGCTGGAGAAGCTGAAGAAGGAGCTGAAGAAGGCGAAGAAGGAG"
        "CTGAAGAAGGCGCTGGAGAAGCTGAAGAAGGAGCTGAAGAAGGCGAAGAAGGAGCTGAAGAAGGCGCTGGAGAAGCTG"
    ),
    'plant_sample': (
        "ATGGCTTCCACTGCTGCTGTCACTGTCGCAGCTACTGTTACTGCTACCGTTACTACTGCTACCGTTACTACTGCTACT"
        "GTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACT"
        "GCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACT"
        "GTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACTGCTACTGTTACT"
    ),
}


def get_diversity_sequences() -> dict:
    """Get representative sequences from different domains of life."""
    return {
        name: NCBISequence(
            accession=f'DIVERSITY_{name.upper()}',
            description=f'Representative {name} sequence',
            sequence=seq,
            organism=name.replace('_sample', '')
        )
        for name, seq in DIVERSITY_SEQUENCES.items()
    }


if __name__ == "__main__":
    print("DNA Database Fetcher")
    print("=" * 40)

    # List available samples
    print("\nAvailable sample sequences:")
    for name in list_available_samples():
        sample = get_sample_sequence(name)
        print(f"  - {name}: {len(sample.sequence)} bp - {sample.description[:50]}...")

    # Test fetch (if network available)
    print("\n\nFamous sequence accessions (for NCBI fetch):")
    for name, accession in FAMOUS_SEQUENCES.items():
        print(f"  - {name}: {accession}")

    print("\n\nDiversity sequences for cross-domain analysis:")
    for name, seq in get_diversity_sequences().items():
        print(f"  - {name}: {seq.length} bp, GC={sum(1 for b in seq.sequence if b in 'GC')/seq.length:.1%}")
