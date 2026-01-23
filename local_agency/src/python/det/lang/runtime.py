"""
Existence-Lang Runtime Support
==============================

Runtime classes for executing transpiled Existence-Lang code.
Provides Register, WitnessToken, Creature base classes and DET integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Callable
from enum import Enum, auto
import math
import random

from .errors import WitnessToken, AgencyError, ResourceError


# ==============================================================================
# Witness Tokens (Consequences)
# ==============================================================================

class CompareResult(Enum):
    """Comparison result tokens."""
    LT = auto()
    EQ = auto()
    GT = auto()


class ReconcileResult(Enum):
    """Reconciliation result tokens."""
    EQ_OK = auto()
    EQ_FAIL = auto()
    EQ_REFUSE = auto()


class TransferResult(Enum):
    """Transfer result tokens."""
    TRANSFER_OK = auto()
    TRANSFER_PARTIAL = auto()
    TRANSFER_BLOCKED = auto()


class WriteResult(Enum):
    """Write result tokens."""
    WRITE_OK = auto()
    WRITE_REFUSED = auto()
    WRITE_PARTIAL = auto()


# ==============================================================================
# Register (Resource Container)
# ==============================================================================

@dataclass
class Register:
    """
    Resource container, optionally backed by DET node.

    Registers hold a float value representing resource (F).
    Operations on registers produce witness tokens.
    """
    _value: float = 0.0
    _positive: float = 0.0  # Signed representation: positive component
    _negative: float = 0.0  # Signed representation: negative component
    _node_id: Optional[int] = None  # Backing DET node
    _name: str = ""

    @property
    def F(self) -> float:
        """Get resource value (positive - negative)."""
        return self._positive - self._negative

    @F.setter
    def F(self, value: float):
        """Set resource value."""
        if value >= 0:
            self._positive = value
            self._negative = 0.0
        else:
            self._positive = 0.0
            self._negative = -value
        self._value = value

    @property
    def value(self) -> float:
        """Alias for F."""
        return self.F

    @value.setter
    def value(self, v: float):
        self.F = v


@dataclass
class TokenReg:
    """
    Mutable token register for witness tokens.

    Holds the result of operations as enum tokens.
    """
    _token: Any = None
    _name: str = ""

    @property
    def token(self) -> Any:
        return self._token

    @token.setter
    def token(self, value: Any):
        self._token = value

    def __eq__(self, other) -> bool:
        if isinstance(other, TokenReg):
            return self._token == other._token
        return self._token == other

    def __repr__(self) -> str:
        return f"TokenReg({self._token})"


# ==============================================================================
# Creature Base Class
# ==============================================================================

class CreatureBase:
    """
    Base class for all creatures.

    Creatures have implicit DET state: F (resource), q (debt),
    a (agency), theta (phase), sigma (rate).
    """

    def __init__(self, name: str = ""):
        self.name = name

        # DET state
        self._F: float = 100.0      # Resource
        self._q: float = 0.0        # Structural debt
        self._a: float = 1.0        # Agency [0, 1]
        self._theta: float = 0.0    # Phase [0, 2π]
        self._sigma: float = 1.0    # Processing rate
        self._P: float = 1.0        # Presence (computed)
        self._tau: float = 0.0      # Proper time accumulated

        # Coordination load
        self._H: float = 0.0

        # Bonds
        self._bonds: list['BondInstance'] = []

        # Somatic bindings
        self._sensors: dict[str, 'SensorBinding'] = {}
        self._actuators: dict[str, 'ActuatorBinding'] = {}

        # Variables
        self._vars: dict[str, Any] = {}

    @property
    def F(self) -> float:
        """Resource."""
        return self._F

    @F.setter
    def F(self, value: float):
        self._F = max(0.0, value)

    @property
    def q(self) -> float:
        """Structural debt."""
        return self._q

    @q.setter
    def q(self, value: float):
        self._q = max(0.0, value)

    @property
    def a(self) -> float:
        """Agency (capped by debt)."""
        # Agency ceiling: a_max = 1 / (1 + λ_a × q²)
        lambda_a = 0.1
        a_max = 1.0 / (1.0 + lambda_a * self._q * self._q)
        return min(self._a, a_max)

    @property
    def theta(self) -> float:
        """Phase."""
        return self._theta

    @theta.setter
    def theta(self, value: float):
        self._theta = value % (2 * math.pi)

    @property
    def sigma(self) -> float:
        """Processing rate."""
        return self._sigma

    @sigma.setter
    def sigma(self, value: float):
        self._sigma = max(0.0, value)

    @property
    def P(self) -> float:
        """Presence (scheduling weight)."""
        # P = a × σ / (1 + F_op) / (1 + H)
        F_op = max(0.0, 1.0 - self._F / 100.0)  # Operational deficit
        return self.a * self._sigma / (1.0 + F_op) / (1.0 + self._H)

    def update_presence(self):
        """Recompute presence."""
        self._P = self.P

    def agency_check(self, min_a: float = 0.0) -> bool:
        """Check if agency exceeds threshold."""
        return self.a >= min_a

    def participate(self, bond: 'BondInstance'):
        """Override in subclass: participation law."""
        pass

    def agency_block(self):
        """Override in subclass: agency-gated behavior."""
        pass

    def grace_handler(self):
        """Override in subclass: grace handling."""
        pass

    def tick(self, dt: float = 0.1):
        """Execute one tick."""
        # Update proper time
        self._tau += dt

        # Run participation with each bond
        for bond in self._bonds:
            self.participate(bond)

        # Run agency block if sufficient agency
        if self.agency_check(0.1):
            self.agency_block()

        # Check for grace need
        if self._F < 10.0:  # Threshold for grace
            self.grace_handler()

        # Update presence
        self.update_presence()


# ==============================================================================
# Bond Instance
# ==============================================================================

@dataclass
class BondInstance:
    """
    Instance of a bond between two creatures.
    """
    creature_a: CreatureBase
    creature_b: CreatureBase
    bond_type: str = "default"

    # Bond state
    C: float = 0.5              # Coherence [0, 1]
    pi: float = 0.0             # Momentum
    sigma: float = 1.0          # Conductivity

    # Parameters
    alpha_C: float = 0.15       # Coherence charging
    lambda_C: float = 0.02      # Coherence decay
    slip_threshold: float = 0.3  # Phase slip threshold

    def diffuse(self, value: float) -> float:
        """Symmetric flux exchange."""
        # Diffuse based on conductivity and coherence
        flux = value * self.sigma * self.C
        return flux

    def phase_align(self) -> float:
        """Get phase alignment."""
        return math.cos(self.creature_a.theta - self.creature_b.theta)

    def update(self, dt: float = 0.1):
        """Update bond state."""
        # Phase alignment
        align = self.phase_align()
        slip = 1.0 if align < self.slip_threshold else 0.0

        # Coherence update
        self.C += self.alpha_C * abs(self.pi) - self.lambda_C * self.C - slip * self.C
        self.C = max(0.0, min(1.0, self.C))

    @property
    def other(self) -> CreatureBase:
        """Get the other creature (from perspective of last accessor)."""
        return self.creature_b


# ==============================================================================
# Somatic Bindings
# ==============================================================================

@dataclass
class SensorBinding:
    """Binding to a sensor (afferent)."""
    name: str
    sensor_type: str
    channel: int
    _value: float = 0.0
    _raw_value: float = 0.0

    @property
    def value(self) -> float:
        return self._value

    def read(self) -> TokenReg:
        """Read sensor value, returns witness token."""
        token = TokenReg()
        token.token = self._value
        return token


@dataclass
class ActuatorBinding:
    """Binding to an actuator (efferent)."""
    name: str
    actuator_type: str
    channel: int
    _target: float = 0.0
    _output: float = 0.0

    @property
    def target(self) -> float:
        return self._target

    def set(self, value: float, agency: float = 1.0) -> WriteResult:
        """Set actuator target, agency-gated."""
        if agency < 0.1:
            return WriteResult.WRITE_REFUSED
        self._target = value
        self._output = value * agency  # Modulated by agency
        return WriteResult.WRITE_OK


# ==============================================================================
# Kernel Base Class
# ==============================================================================

class KernelBase:
    """
    Base class for kernels (function as law module).

    Kernels have ports, params, and phases. They don't return values,
    they write to output ports and produce witness tokens.
    """

    def __init__(self):
        self._ports: dict[str, Any] = {}
        self._params: dict[str, Any] = {}
        self._witness: TokenReg = TokenReg()

    def bind_port(self, name: str, value: Any):
        """Bind a value to a port."""
        self._ports[name] = value

    def get_port(self, name: str) -> Any:
        """Get port value."""
        return self._ports.get(name)

    def set_param(self, name: str, value: Any):
        """Set parameter."""
        self._params[name] = value

    def get_param(self, name: str, default: Any = None) -> Any:
        """Get parameter."""
        return self._params.get(name, default)

    def phase_COMMIT(self):
        """Override: COMMIT phase implementation."""
        pass

    def execute(self):
        """Execute kernel."""
        self.phase_COMMIT()


# ==============================================================================
# Runtime Environment
# ==============================================================================

class ExistenceRuntime:
    """
    Main runtime for executing Existence-Lang programs.

    Manages creatures, kernels, bonds, and DET core integration.
    """

    def __init__(self, det_core: Optional[Any] = None):
        self._det_core = det_core
        self._creatures: dict[str, CreatureBase] = {}
        self._kernels: dict[str, type] = {}
        self._bonds: list[BondInstance] = []
        self._tick: int = 0
        self._trace: list[Any] = []

    def register_creature_type(self, name: str, creature_class: type):
        """Register a creature type."""
        self._kernels[name] = creature_class

    def spawn_creature(self, name: str, creature_type: str, **kwargs) -> CreatureBase:
        """Spawn a creature instance."""
        if creature_type in self._kernels:
            creature = self._kernels[creature_type](name=name, **kwargs)
        else:
            creature = CreatureBase(name=name)
        self._creatures[name] = creature
        return creature

    def create_bond(self, c1_name: str, c2_name: str, bond_type: str = "default") -> BondInstance:
        """Create a bond between two creatures."""
        c1 = self._creatures.get(c1_name)
        c2 = self._creatures.get(c2_name)
        if c1 and c2:
            bond = BondInstance(creature_a=c1, creature_b=c2, bond_type=bond_type)
            self._bonds.append(bond)
            c1._bonds.append(bond)
            c2._bonds.append(bond)
            return bond
        raise ValueError(f"Creature not found: {c1_name} or {c2_name}")

    def inject_F(self, creature_name: str, amount: float):
        """Inject resource into a creature."""
        creature = self._creatures.get(creature_name)
        if creature:
            creature.F += amount

    def step(self, dt: float = 0.1):
        """Execute one simulation step."""
        self._tick += 1

        # Update all bonds
        for bond in self._bonds:
            bond.update(dt)

        # Tick all creatures (ordered by presence)
        creatures_by_presence = sorted(
            self._creatures.values(),
            key=lambda c: c.P,
            reverse=True
        )
        for creature in creatures_by_presence:
            creature.tick(dt)

        # Update DET core if available
        if self._det_core:
            self._det_core.step(dt)

    def run(self, steps: int = 100, dt: float = 0.1):
        """Run simulation for multiple steps."""
        for _ in range(steps):
            self.step(dt)

    def get_state(self) -> dict:
        """Get current runtime state."""
        return {
            "tick": self._tick,
            "creatures": {
                name: {
                    "F": c.F,
                    "q": c.q,
                    "a": c.a,
                    "theta": c.theta,
                    "sigma": c.sigma,
                    "P": c.P,
                }
                for name, c in self._creatures.items()
            },
            "bonds": [
                {
                    "a": b.creature_a.name,
                    "b": b.creature_b.name,
                    "C": b.C,
                    "pi": b.pi,
                }
                for b in self._bonds
            ],
        }

    def execute_file(self, filepath: str):
        """Execute an Existence-Lang file."""
        with open(filepath) as f:
            source = f.read()
        self.execute_source(source)

    def execute_source(self, source: str):
        """Execute Existence-Lang source code."""
        from .parser import parse
        from .transpiler import Transpiler

        # Parse
        ast = parse(source)

        # Transpile
        transpiler = Transpiler()
        python_code = transpiler.transpile(ast)

        # Execute
        exec(python_code, {"runtime": self, "math": math, "random": random})


# ==============================================================================
# Primitive Operations
# ==============================================================================

# Counter for distinct identity generation
_distinct_counter = 0

def distinct() -> tuple[Register, Register]:
    """Create two distinct identities with unique initial states."""
    global _distinct_counter
    _distinct_counter += 1
    a = Register()
    b = Register()
    # Give them distinct initial resource states
    a.F = float(_distinct_counter)
    b.F = float(_distinct_counter + 0.5)
    return (a, b)


def transfer(src: Register, dst: Register, amount: Optional[float] = None) -> TransferResult:
    """Antisymmetric resource transfer."""
    if amount is None:
        amount = src.F

    actual = min(amount, src.F)
    if actual <= 0:
        return TransferResult.TRANSFER_BLOCKED

    src.F -= actual
    dst.F += actual

    if actual < amount:
        return TransferResult.TRANSFER_PARTIAL
    return TransferResult.TRANSFER_OK


def diffuse(a: Register, b: Register, sigma: float = 1.0) -> float:
    """Symmetric flux exchange."""
    diff = a.F - b.F
    flux = sigma * diff * 0.5
    a.F -= flux
    b.F += flux
    return abs(flux)


def compare(a: Any, b: Any, epsilon: float = 1e-6) -> CompareResult:
    """Compare two values, producing past token."""
    if isinstance(a, Register):
        a = a.F
    if isinstance(b, Register):
        b = b.F

    if abs(a - b) < epsilon:
        return CompareResult.EQ
    elif a < b:
        return CompareResult.LT
    else:
        return CompareResult.GT


def reconcile(a: Register, b: Register, epsilon: float = 1e-3) -> ReconcileResult:
    """Attempt to unify two registers."""
    diff = abs(a.F - b.F)
    if diff < epsilon:
        return ReconcileResult.EQ_OK

    # Diffuse toward equality
    diffuse(a, b, sigma=1.0)

    # Check if reconciled
    if abs(a.F - b.F) < epsilon:
        return ReconcileResult.EQ_OK

    return ReconcileResult.EQ_FAIL


def choose(proposals: list[str], decisiveness: float = 1.0, seed: Optional[int] = None) -> str:
    """Choose from proposals based on decisiveness."""
    if seed is not None:
        random.seed(seed)

    if decisiveness >= 1.0:
        return proposals[0] if proposals else ""

    # Weighted random choice based on decisiveness
    weights = [decisiveness ** i for i in range(len(proposals))]
    total = sum(weights)
    r = random.random() * total
    cumsum = 0.0
    for p, w in zip(proposals, weights):
        cumsum += w
        if r <= cumsum:
            return p
    return proposals[-1] if proposals else ""


def local_seed(k: int, r: float, theta: float) -> int:
    """Generate deterministic local seed from state."""
    return int((k * 1000 + r * 100 + theta * 10) % 2147483647)
