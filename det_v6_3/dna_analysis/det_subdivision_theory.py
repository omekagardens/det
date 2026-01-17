"""
DET Subdivision Theory: Learning from DNA Replication

This module explores how agentic subdivision could work in DET by studying
DNA replication - nature's solution to the "birth" problem.

KEY QUESTIONS:
1. Is agency conserved during subdivision? (like energy)
2. How does locality work during division?
3. What role does coherence play in enabling/preventing division?
4. Is there a fixed "agentic substance" or can it grow?

DNA REPLICATION INSIGHTS:
- Semi-conservative: Each daughter gets 1 old + 1 new strand
- Template-directed: Information is copied, not created
- Energy-consuming: ATP required (resource expenditure)
- Coherence-breaking: H-bonds must break for separation
- Local process: Replication fork moves base-by-base

PROPOSED DET SUBDIVISION PRINCIPLES:
1. Division requires coherence breaking (C must drop below threshold)
2. Resource F is consumed/redistributed during division
3. Agency a may be conserved, shared, or templated
4. Structural debt q affects division capability
5. Locality is preserved - division happens at bonds, not globally
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import math

# DET Constants
PHI = (1 + np.sqrt(5)) / 2
DET_PARAMS = {
    'C_division_threshold': 0.3,   # Coherence must drop below this to divide
    'F_division_cost': 0.5,        # Resource cost per division
    'a_inheritance_mode': 'template',  # 'conserve', 'template', 'emerge'
    'q_inheritance_factor': 0.8,   # Fraction of structural debt inherited
    'division_locality_radius': 1,  # Division only affects local bonds
}


class AgencyInheritanceMode(Enum):
    """How agency is handled during subdivision."""
    CONSERVE = "conserve"      # Total agency split between daughters (a1 + a2 = a_parent)
    TEMPLATE = "template"      # Agency is copied/templated (like DNA information)
    EMERGE = "emerge"          # New agency emerges from environment
    GRACE = "grace"            # Agency borrowed from boundary/universal pool


@dataclass
class DETAgent:
    """A DET agent that can potentially subdivide."""
    id: int
    F: float          # Resource
    a: float          # Agency
    q: float          # Structural debt
    C_bonds: Dict[int, float] = field(default_factory=dict)  # Coherence with bonded agents
    parent_id: Optional[int] = None
    generation: int = 0

    @property
    def total_coherence(self) -> float:
        """Total coherence across all bonds."""
        return sum(self.C_bonds.values()) if self.C_bonds else 0

    @property
    def mean_coherence(self) -> float:
        """Mean coherence per bond."""
        return self.total_coherence / len(self.C_bonds) if self.C_bonds else 0

    @property
    def can_divide(self) -> bool:
        """Check if agent meets division criteria."""
        # Need sufficient resource
        if self.F < DET_PARAMS['F_division_cost']:
            return False
        # Need low enough coherence (bonds must be breakable)
        if self.mean_coherence > DET_PARAMS['C_division_threshold']:
            return False
        # Structural debt limits division
        if self.q > 0.9:  # Too "locked" to divide
            return False
        return True

    @property
    def division_readiness(self) -> float:
        """Score from 0-1 indicating readiness to divide."""
        f_score = min(self.F / DET_PARAMS['F_division_cost'], 1.0)
        c_score = max(0, 1 - self.mean_coherence / DET_PARAMS['C_division_threshold'])
        q_score = 1 - self.q
        return f_score * c_score * q_score


@dataclass
class DNAReplicationModel:
    """
    Models DNA replication to understand subdivision mechanics.

    DNA replication key phases:
    1. INITIATION: Origin recognition, helicase loading
    2. UNWINDING: Helicase breaks H-bonds (coherence breaking)
    3. PRIMING: RNA primers provide starting points
    4. ELONGATION: Polymerase copies template strand
    5. TERMINATION: Replication forks meet, strands separate
    """

    # DNA replication parameters (mapped to DET concepts)
    helicase_coherence_break_rate: float = 0.1   # How fast C drops
    polymerase_copy_fidelity: float = 0.9999     # Agency template accuracy
    atp_per_base: float = 2.0                     # Resource cost per unit
    origin_spacing: int = 50000                   # Locality of initiation

    def compute_replication_det_mapping(self) -> Dict:
        """Map DNA replication to DET subdivision concepts."""

        mapping = {
            'coherence_dynamics': {
                'pre_division': 'High C (H-bonds intact)',
                'initiation': 'C begins dropping at origins',
                'unwinding': 'C → 0 at replication fork',
                'post_division': 'New C forms with daughter strands',
                'det_parallel': 'Division requires C < C_threshold'
            },

            'resource_dynamics': {
                'source': 'ATP from cellular metabolism',
                'consumption': f'{self.atp_per_base} ATP per base pair',
                'total_cost': 'Genome_size × atp_per_base',
                'det_parallel': 'F consumed during division, must be replenished'
            },

            'agency_dynamics': {
                'mechanism': 'Template-directed copying',
                'fidelity': f'{self.polymerase_copy_fidelity:.4%}',
                'information_conservation': 'Sequence preserved in both daughters',
                'det_parallel': 'Agency templated, not created or split'
            },

            'locality': {
                'mechanism': 'Replication fork moves locally',
                'speed': '~1000 bases/second in bacteria',
                'origin_spacing': f'{self.origin_spacing} bp between origins',
                'det_parallel': 'Division propagates through local bonds'
            },

            'structural_debt': {
                'mechanism': 'Epigenetic marks partially inherited',
                'methylation': 'Semi-conservative (one strand marked)',
                'chromatin': 'Must be reassembled on new DNA',
                'det_parallel': 'q partially inherited, partially reset'
            }
        }

        return mapping


class SubdivisionTheory:
    """
    Theoretical framework for DET agent subdivision.

    Core insight from DNA: Division is not about CREATING new agency,
    but about COPYING/TEMPLATING existing agency while BREAKING coherence.

    The "agentic substance" question:
    - DNA suggests information (agency) is TEMPLATED, not created
    - Energy (resource) is CONSUMED from environment
    - The "substance" is the PATTERN, not the matter
    """

    def __init__(self, inheritance_mode: AgencyInheritanceMode = AgencyInheritanceMode.TEMPLATE):
        self.inheritance_mode = inheritance_mode
        self.total_agency_created = 0
        self.total_resource_consumed = 0
        self.division_history = []

    def analyze_agency_conservation(self) -> Dict:
        """
        Analyze whether agency should be conserved in DET.

        Three possibilities:

        1. STRICT CONSERVATION (like energy):
           a_daughter1 + a_daughter2 = a_parent
           - Problem: Agency would dilute with each generation
           - Eventually all agents would have negligible agency

        2. TEMPLATING (like DNA information):
           a_daughter1 ≈ a_parent, a_daughter2 ≈ a_parent
           - Agency is COPIED, not split
           - Total agency in universe GROWS
           - But: Where does new agency "come from"?

        3. EMERGENCE (from environment):
           a_daughter = f(environment, parent_template)
           - Agency emerges from local conditions
           - Parent provides template, environment provides "substance"
           - Compatible with DET's grace mechanism?
        """

        analysis = {
            'conservation_model': {
                'description': 'Agency splits like energy',
                'equation': 'a₁ + a₂ = a_parent',
                'pros': ['Mathematically clean', 'No "creation" problem'],
                'cons': ['Agency dilutes to zero', 'Contradicts biological observation'],
                'verdict': 'REJECTED - leads to heat death of agency'
            },

            'template_model': {
                'description': 'Agency is copied like DNA sequence',
                'equation': 'a₁ ≈ a₂ ≈ a_parent (with small error)',
                'pros': ['Matches biology', 'Explains inheritance', 'Agency persists'],
                'cons': ['Total agency grows - where from?'],
                'resolution': 'Agency is PATTERN, environment provides substrate',
                'verdict': 'FAVORED - matches DNA replication'
            },

            'emergence_model': {
                'description': 'Agency emerges from favorable conditions',
                'equation': 'a_new = template × environment_factor',
                'pros': ['Explains abiogenesis', 'Compatible with locality'],
                'cons': ['Less predictable', 'Needs environmental theory'],
                'verdict': 'POSSIBLE - for de novo agent creation'
            },

            'synthesis': """
            PROPOSED RESOLUTION:

            Agency (a) is not a conserved quantity like energy (F).
            Agency is more like INFORMATION - it can be copied.

            The "substance" of agency is the PATTERN of relationships,
            not some metaphysical fluid that must be conserved.

            Just as DNA replication doesn't "use up" information,
            agent subdivision doesn't "use up" agency.

            What IS consumed:
            - Resource F (energy for the copying process)
            - Coherence C (must break bonds to separate)
            - Time (division takes many steps)

            What is COPIED:
            - Agency a (the pattern/template)
            - Some structural information

            What is RESET:
            - Coherence C (new bonds form fresh)
            - Some structural debt q
            """
        }

        return analysis

    def analyze_locality_in_division(self) -> Dict:
        """
        How does division maintain DET's strict locality?

        DNA insight: Replication is ENTIRELY LOCAL.
        - Fork moves one base at a time
        - No action at a distance
        - Information flows through physical contact
        """

        analysis = {
            'dna_mechanism': {
                'description': 'Replication fork progression',
                'locality_radius': '1 base pair',
                'information_flow': 'Template strand → Polymerase → New strand',
                'no_action_at_distance': True
            },

            'det_analog': {
                'description': 'Division propagates through bonds',
                'process': [
                    '1. Agent reaches division_readiness threshold',
                    '2. Coherence with ONE neighbor drops below C_threshold',
                    '3. Bond breaks, creating division site',
                    '4. Resource F flows to power the process',
                    '5. Agency a is templated to new agent',
                    '6. New bonds form with reduced C',
                    '7. Process propagates to next bond'
                ],
                'locality_preserved': True,
                'radius': DET_PARAMS['division_locality_radius']
            },

            'key_insight': """
            Division is not a GLOBAL event but a LOCAL PROPAGATION.

            Like a zipper opening, division moves through the network
            one bond at a time. Each step is local:
            - Check local C (is bond breakable?)
            - Consume local F (is there energy?)
            - Template local a (copy the pattern)

            The "birth" of a new agent is the ACCUMULATION of local
            bond-breaking events, not a single global creation.
            """
        }

        return analysis

    def propose_division_mechanism(self) -> Dict:
        """
        Propose a complete DET subdivision mechanism.
        """

        mechanism = {
            'name': 'Template-Propagated Division (TPD)',

            'prerequisites': {
                'resource': 'F > F_division_cost',
                'coherence': 'mean(C_bonds) < C_division_threshold',
                'structure': 'q < 0.9 (not too locked)',
                'agency': 'a > 0.1 (must have pattern to copy)'
            },

            'process': {
                'phase_1_initiation': {
                    'trigger': 'Division readiness score exceeds threshold',
                    'action': 'Mark agent as "dividing"',
                    'locality': 'Entirely internal to agent'
                },

                'phase_2_coherence_breaking': {
                    'trigger': 'Division initiated',
                    'action': 'Systematically reduce C on bonds',
                    'rate': 'dC/dk = -λ_division × C',
                    'locality': 'Affects only direct bonds',
                    'cost': 'F consumed proportional to C broken'
                },

                'phase_3_templating': {
                    'trigger': 'C drops below threshold on a bond',
                    'action': 'Create new agent with templated properties',
                    'agency_rule': 'a_new = a_parent × (1 - ε), ε ~ 0.001',
                    'resource_rule': 'F_new = F_parent × split_fraction',
                    'structure_rule': 'q_new = q_parent × q_inheritance',
                    'locality': 'New agent appears at bond site'
                },

                'phase_4_reconnection': {
                    'trigger': 'New agent created',
                    'action': 'Form new bonds with fresh coherence',
                    'C_initial': 'C_0 from DET params (0.15)',
                    'topology': 'Parent bonds to daughter, daughter bonds to former neighbor',
                    'locality': 'Only affects immediate neighborhood'
                },

                'phase_5_stabilization': {
                    'trigger': 'All bonds reformed',
                    'action': 'Normal DET dynamics resume',
                    'coherence': 'C grows through interaction',
                    'resource': 'F redistributes through diffusion',
                    'agency': 'a may drift based on environment'
                }
            },

            'conservation_laws': {
                'resource_F': 'Locally conserved (consumed during division)',
                'agency_a': 'NOT conserved (templated/copied)',
                'coherence_C': 'Reset during division',
                'structure_q': 'Partially inherited',
                'total_agents': 'Increases by 1 per division'
            },

            'biological_parallels': {
                'phase_1': 'Origin recognition complex binding',
                'phase_2': 'Helicase unwinding (H-bond breaking)',
                'phase_3': 'DNA polymerase synthesis (templating)',
                'phase_4': 'Ligation and chromatin assembly',
                'phase_5': 'Cell cycle completion'
            }
        }

        return mechanism

    def simulate_division(self, parent: DETAgent, next_id: int) -> Tuple[DETAgent, DETAgent]:
        """
        Simulate a single division event.

        Returns two daughter agents.
        """
        if not parent.can_divide:
            raise ValueError(f"Agent {parent.id} cannot divide: readiness={parent.division_readiness:.3f}")

        # Consume resource
        f_cost = DET_PARAMS['F_division_cost']
        f_remaining = parent.F - f_cost
        self.total_resource_consumed += f_cost

        # Determine agency inheritance
        if self.inheritance_mode == AgencyInheritanceMode.CONSERVE:
            a1 = parent.a * 0.5
            a2 = parent.a * 0.5
        elif self.inheritance_mode == AgencyInheritanceMode.TEMPLATE:
            # Small copy error
            a1 = parent.a * (1 - np.random.normal(0, 0.001))
            a2 = parent.a * (1 - np.random.normal(0, 0.001))
            a1 = np.clip(a1, 0, 1)
            a2 = np.clip(a2, 0, 1)
            self.total_agency_created += (a1 + a2 - parent.a)
        elif self.inheritance_mode == AgencyInheritanceMode.EMERGE:
            # Agency depends on "environment" (random factor)
            env_factor = np.random.uniform(0.8, 1.2)
            a1 = parent.a * env_factor * 0.5
            a2 = parent.a * env_factor * 0.5
            a1 = np.clip(a1, 0, 1)
            a2 = np.clip(a2, 0, 1)
        else:  # GRACE
            # Borrow from universal pool
            a1 = parent.a
            a2 = parent.a * 0.9  # Slight reduction

        # Structural debt inheritance
        q_inherit = DET_PARAMS['q_inheritance_factor']
        q1 = parent.q * q_inherit
        q2 = parent.q * q_inherit

        # Create daughters
        daughter1 = DETAgent(
            id=next_id,
            F=f_remaining * 0.5,
            a=a1,
            q=q1,
            parent_id=parent.id,
            generation=parent.generation + 1
        )

        daughter2 = DETAgent(
            id=next_id + 1,
            F=f_remaining * 0.5,
            a=a2,
            q=q2,
            parent_id=parent.id,
            generation=parent.generation + 1
        )

        # Record division
        self.division_history.append({
            'parent_id': parent.id,
            'daughter_ids': [daughter1.id, daughter2.id],
            'parent_a': parent.a,
            'daughter_a': [a1, a2],
            'f_consumed': f_cost,
            'generation': parent.generation + 1
        })

        return daughter1, daughter2


def analyze_primordial_agency_question():
    """
    Address: Was there a fixed amount of agentic substance at the beginning?

    This is the cosmological question for DET.
    """

    analysis = {
        'question': 'Was there a fixed amount of agency at the Big Bang?',

        'option_1_fixed_pool': {
            'description': 'Agency is conserved like energy',
            'implication': 'Total agency constant since t=0',
            'problems': [
                'Agency dilutes with cosmic expansion',
                'Life becomes less possible over time',
                'Contradicts observed complexity increase'
            ],
            'verdict': 'PROBLEMATIC'
        },

        'option_2_emergent': {
            'description': 'Agency emerges from complexity',
            'implication': 'Agency grows as structures form',
            'mechanism': 'When coherence and structure align, agency emerges',
            'biological_parallel': 'Abiogenesis - life from non-life',
            'problems': [
                'What determines emergence threshold?',
                'Is there a maximum agency density?'
            ],
            'verdict': 'PLAUSIBLE'
        },

        'option_3_template_growth': {
            'description': 'Small initial agency, grows through templating',
            'implication': 'Agency can copy itself given resources',
            'mechanism': 'Like DNA - information copies, matter cycles',
            'biological_parallel': 'All life from first replicator',
            'advantages': [
                'Explains why agency persists',
                'Explains inheritance',
                'Compatible with thermodynamics'
            ],
            'verdict': 'FAVORED'
        },

        'synthesis': """
        PROPOSED ANSWER:

        Agency is not a conserved "substance" but an emergent PATTERN
        that can template itself given sufficient resources and
        favorable conditions.

        At the beginning:
        - Minimal or zero agency (pure physics)
        - High resource F (Big Bang energy)
        - High coherence C (everything entangled)
        - Low structure q (no stable patterns yet)

        As universe evolved:
        - Resources dispersed
        - Coherence broke (decoherence)
        - Structures formed (stars, planets, molecules)
        - SOMEWHERE, conditions aligned for agency emergence

        Once agency emerged:
        - It could TEMPLATE itself (replicate)
        - Each copy consumed resources but preserved pattern
        - Selection favored efficient replicators
        - Complexity increased

        The "agentic substance" is not substance at all -
        it's PATTERN. Patterns can copy. Patterns can grow.
        The substrate (matter, energy) is what's conserved.
        The pattern (information, agency) is what propagates.

        DNA is the proof: 3.8 billion years of continuous
        pattern propagation, using the same atoms recycled
        endlessly, but the INFORMATION persists and grows.
        """
    }

    return analysis


def run_subdivision_analysis():
    """Run complete analysis of DET subdivision theory."""

    print("=" * 80)
    print("DET SUBDIVISION THEORY: Learning from DNA Replication")
    print("=" * 80)

    # DNA replication mapping
    print("\n" + "=" * 60)
    print("SECTION 1: DNA REPLICATION → DET MAPPING")
    print("=" * 60)

    dna_model = DNAReplicationModel()
    mapping = dna_model.compute_replication_det_mapping()

    for category, details in mapping.items():
        print(f"\n{category.upper()}:")
        if isinstance(details, dict):
            for key, value in details.items():
                print(f"  {key}: {value}")

    # Agency conservation analysis
    print("\n" + "=" * 60)
    print("SECTION 2: AGENCY CONSERVATION ANALYSIS")
    print("=" * 60)

    theory = SubdivisionTheory(AgencyInheritanceMode.TEMPLATE)
    conservation = theory.analyze_agency_conservation()

    for model_name, model_data in conservation.items():
        if isinstance(model_data, dict):
            print(f"\n{model_name.upper()}:")
            for key, value in model_data.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    - {item}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(f"\n{model_name.upper()}:")
            print(model_data)

    # Locality analysis
    print("\n" + "=" * 60)
    print("SECTION 3: LOCALITY IN DIVISION")
    print("=" * 60)

    locality = theory.analyze_locality_in_division()
    for section, data in locality.items():
        print(f"\n{section.upper()}:")
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, list):
                    print(f"  {key}:")
                    for item in value:
                        print(f"    {item}")
                else:
                    print(f"  {key}: {value}")
        else:
            print(data)

    # Proposed mechanism
    print("\n" + "=" * 60)
    print("SECTION 4: PROPOSED DIVISION MECHANISM")
    print("=" * 60)

    mechanism = theory.propose_division_mechanism()
    print(f"\nMechanism: {mechanism['name']}")

    print("\nPrerequisites:")
    for key, value in mechanism['prerequisites'].items():
        print(f"  {key}: {value}")

    print("\nProcess Phases:")
    for phase, details in mechanism['process'].items():
        print(f"\n  {phase}:")
        for key, value in details.items():
            print(f"    {key}: {value}")

    print("\nConservation Laws:")
    for quantity, rule in mechanism['conservation_laws'].items():
        print(f"  {quantity}: {rule}")

    # Primordial agency question
    print("\n" + "=" * 60)
    print("SECTION 5: THE PRIMORDIAL AGENCY QUESTION")
    print("=" * 60)

    primordial = analyze_primordial_agency_question()
    print(f"\nQuestion: {primordial['question']}")

    for option, data in primordial.items():
        if option.startswith('option'):
            print(f"\n{data['description'].upper()}:")
            print(f"  Implication: {data['implication']}")
            if 'mechanism' in data:
                print(f"  Mechanism: {data['mechanism']}")
            if 'problems' in data:
                print(f"  Problems:")
                for p in data['problems']:
                    print(f"    - {p}")
            print(f"  Verdict: {data['verdict']}")

    print("\n" + "=" * 60)
    print("SYNTHESIS")
    print("=" * 60)
    print(primordial['synthesis'])

    # Simulation demo
    print("\n" + "=" * 60)
    print("SECTION 6: DIVISION SIMULATION")
    print("=" * 60)

    # Create a parent agent ready to divide
    parent = DETAgent(
        id=0,
        F=2.0,  # Enough resource
        a=0.8,  # High agency
        q=0.3,  # Low structural debt
        C_bonds={1: 0.2, 2: 0.25}  # Low coherence bonds
    )

    print(f"\nParent agent:")
    print(f"  ID: {parent.id}")
    print(f"  F: {parent.F:.2f}")
    print(f"  a: {parent.a:.2f}")
    print(f"  q: {parent.q:.2f}")
    print(f"  mean_C: {parent.mean_coherence:.2f}")
    print(f"  can_divide: {parent.can_divide}")
    print(f"  division_readiness: {parent.division_readiness:.3f}")

    # Perform division
    theory = SubdivisionTheory(AgencyInheritanceMode.TEMPLATE)
    d1, d2 = theory.simulate_division(parent, next_id=1)

    print(f"\nAfter TEMPLATE division:")
    print(f"\n  Daughter 1 (ID={d1.id}):")
    print(f"    F: {d1.F:.2f}")
    print(f"    a: {d1.a:.4f} (templated from {parent.a:.2f})")
    print(f"    q: {d1.q:.2f}")
    print(f"    generation: {d1.generation}")

    print(f"\n  Daughter 2 (ID={d2.id}):")
    print(f"    F: {d2.F:.2f}")
    print(f"    a: {d2.a:.4f} (templated from {parent.a:.2f})")
    print(f"    q: {d2.q:.2f}")
    print(f"    generation: {d2.generation}")

    print(f"\n  Resource consumed: {theory.total_resource_consumed:.2f}")
    print(f"  Net agency created: {theory.total_agency_created:.4f}")
    print(f"  (Positive = templating creates new agency)")

    # Compare inheritance modes
    print("\n" + "-" * 40)
    print("Comparing inheritance modes over 5 generations:")
    print("-" * 40)

    for mode in AgencyInheritanceMode:
        theory = SubdivisionTheory(mode)
        agent = DETAgent(id=0, F=10.0, a=0.8, q=0.1, C_bonds={1: 0.2})

        agents = [agent]
        for gen in range(5):
            new_agents = []
            for ag in agents:
                if ag.F >= DET_PARAMS['F_division_cost']:
                    ag.F = 2.0  # Replenish for demo
                    ag.C_bonds = {0: 0.2}
                    try:
                        d1, d2 = theory.simulate_division(ag, len(agents) + len(new_agents))
                        new_agents.extend([d1, d2])
                    except:
                        new_agents.append(ag)
                else:
                    new_agents.append(ag)
            agents = new_agents[:8]  # Cap for display

        total_a = sum(ag.a for ag in agents)
        mean_a = total_a / len(agents)
        print(f"\n  {mode.value}:")
        print(f"    Final agents: {len(agents)}")
        print(f"    Total agency: {total_a:.3f}")
        print(f"    Mean agency: {mean_a:.3f}")
        print(f"    Net agency created: {theory.total_agency_created:.3f}")


if __name__ == "__main__":
    run_subdivision_analysis()
