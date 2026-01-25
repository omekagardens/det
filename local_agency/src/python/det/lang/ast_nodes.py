"""
Existence-Lang AST Node Definitions
===================================

Abstract Syntax Tree nodes for the Existence-Lang agency-first language.
Uses dataclasses for clean immutable structures.
"""

from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum, auto


# ==============================================================================
# Base Types
# ==============================================================================

class TypeKind(Enum):
    """Type kinds in Existence-Lang."""
    REGISTER = auto()
    TOKEN_REG = auto()
    FLOAT = auto()
    INT = auto()
    BOOL = auto()
    STRING = auto()
    BYTES = auto()
    VOID = auto()
    CREATURE = auto()
    BOND = auto()
    CUSTOM = auto()


@dataclass
class TypeAnnotation:
    """Type annotation node."""
    kind: TypeKind
    name: str = ""  # For custom types
    line: int = 0
    column: int = 0


@dataclass
class SourceSpan:
    """Source location span."""
    start_line: int
    start_column: int
    end_line: int = 0
    end_column: int = 0


# ==============================================================================
# Base AST Node
# ==============================================================================

@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 0
    column: int = 0


# ==============================================================================
# Expressions
# ==============================================================================

@dataclass
class Expression(ASTNode):
    """Base class for expressions."""
    pass


@dataclass(kw_only=True)
class Literal(Expression):
    """Literal value: integer, float, string, bool."""
    value: Union[int, float, str, bool]
    type_hint: TypeKind = TypeKind.VOID


@dataclass(kw_only=True)
class Identifier(Expression):
    """Identifier reference."""
    name: str


@dataclass(kw_only=True)
class BinaryOp(Expression):
    """Binary operation."""
    left: Expression
    operator: str  # +, -, *, /, ==, <, etc.
    right: Expression


@dataclass(kw_only=True)
class UnaryOp(Expression):
    """Unary operation."""
    operator: str  # !, -, ~
    operand: Expression


@dataclass(kw_only=True)
class Call(Expression):
    """Function/kernel call."""
    callee: Expression
    arguments: list[Expression] = field(default_factory=list)


@dataclass(kw_only=True)
class FieldAccess(Expression):
    """Field access: obj.field."""
    object: Expression
    field: str


@dataclass(kw_only=True)
class IndexAccess(Expression):
    """Index access: arr[idx]."""
    object: Expression
    index: Expression


@dataclass
class Distinct(Expression):
    """Distinct operation: distinct()."""
    pass


@dataclass(kw_only=True)
class WitnessToken(Expression):
    """Witness token literal: "OK", "FAIL", etc."""
    token_name: str


@dataclass
class This(Expression):
    """Reference to current creature: this."""
    pass


@dataclass(kw_only=True)
class Compare(Expression):
    """Compare operation producing past token."""
    left: Expression
    right: Expression
    epsilon: Optional[Expression] = None


@dataclass(kw_only=True)
class Choose(Expression):
    """Choose from proposals."""
    proposals: list[str]
    decisiveness: Optional[Expression] = None
    seed: Optional[Expression] = None


@dataclass
class TupleExpr(Expression):
    """Tuple expression: (a, b)."""
    elements: list[Expression] = field(default_factory=list)


@dataclass
class ArrayLiteral(Expression):
    """Array literal: [a, b, c]."""
    elements: list[Expression] = field(default_factory=list)


@dataclass(kw_only=True)
class ConditionalExpr(Expression):
    """Conditional expression: if_past(cond) then val1 else val2."""
    condition: Expression
    then_value: Expression
    else_value: Expression


@dataclass(kw_only=True)
class PrimitiveCallExpr(Expression):
    """Primitive call expression: primitive("name", arg1, arg2, ...).

    Calls external I/O functions (llm_call, exec, file_read, etc.)
    that are provided by the substrate layer.
    """
    primitive_name: str
    arguments: list[Expression] = field(default_factory=list)


# ==============================================================================
# Statements
# ==============================================================================

@dataclass
class Statement(ASTNode):
    """Base class for statements."""
    pass


@dataclass(kw_only=True)
class VarDecl(Statement):
    """Variable declaration: var x: Type := value."""
    name: str
    type_annotation: Optional[TypeAnnotation] = None
    initializer: Optional[Expression] = None
    is_alias: bool = False  # := vs =


@dataclass(kw_only=True)
class WitnessDecl(Statement):
    """Witness binding: x ::= operation()."""
    name: str
    expression: Expression


@dataclass(kw_only=True)
class Assignment(Statement):
    """Assignment statement."""
    target: Expression
    operator: str  # =, +=, -=, *=, /=
    value: Expression


@dataclass(kw_only=True)
class ExpressionStmt(Statement):
    """Expression as statement."""
    expression: Expression


@dataclass
class Block(Statement):
    """Block of statements: { ... }."""
    statements: list[Statement] = field(default_factory=list)


@dataclass(kw_only=True)
class IfPast(Statement):
    """Past-token conditional: if_past(token == X) { ... }."""
    condition: Expression
    then_block: Block
    else_block: Optional[Union[Block, 'IfPast']] = None


@dataclass(kw_only=True)
class RepeatPast(Statement):
    """Past-token loop: repeat_past(N) as i { ... }."""
    count: Expression  # Must be past token
    body: Block
    loop_var: Optional[str] = None  # Optional loop variable name


@dataclass(kw_only=True)
class WhilePast(Statement):
    """Past-token while: while_past(condition()) { ... }."""
    condition: Expression
    body: Block


@dataclass
class Return(Statement):
    """Return statement."""
    value: Optional[Expression] = None


@dataclass(kw_only=True)
class KernelCall(Statement):
    """Kernel instantiation call."""
    kernel_name: str
    port_bindings: dict[str, Expression] = field(default_factory=dict)


@dataclass(kw_only=True)
class SomaRead(Statement):
    """Somatic read: soma_read(sensor)."""
    sensor: Expression


@dataclass(kw_only=True)
class SomaWrite(Statement):
    """Somatic write: soma_write(actuator, value)."""
    actuator: Expression
    value: Expression


@dataclass(kw_only=True)
class InjectF(Statement):
    """Resource injection: inject_F(target, amount)."""
    target: Expression
    amount: Expression


@dataclass(kw_only=True)
class RequestGrace(Statement):
    """Grace request: request_grace(amount)."""
    amount: Expression


# ==============================================================================
# Port/Parameter Declarations
# ==============================================================================

class PortDirection(Enum):
    """Port direction for kernels."""
    IN = auto()
    OUT = auto()
    INOUT = auto()


@dataclass(kw_only=True)
class PortDecl(ASTNode):
    """Port declaration in kernel."""
    name: str
    direction: PortDirection
    type_annotation: TypeAnnotation


@dataclass(kw_only=True)
class ParamDecl(ASTNode):
    """Parameter declaration."""
    name: str
    type_annotation: Optional[TypeAnnotation] = None
    default_value: Optional[Expression] = None


# ==============================================================================
# Kernel Components
# ==============================================================================

@dataclass(kw_only=True)
class Proposal(ASTNode):
    """Proposal block in kernel."""
    name: str
    statements: list[Statement] = field(default_factory=list)  # Local computations
    score: Optional[Expression] = None
    effect: Optional[Block] = None


@dataclass(kw_only=True)
class PhaseBlock(ASTNode):
    """Phase block in kernel (COMMIT, etc.)."""
    phase_name: str
    proposals: list[Proposal] = field(default_factory=list)
    choice: Optional['ChoiceDecl'] = None
    commit: Optional[str] = None
    witness_write: Optional[WitnessDecl] = None
    statements: list[Statement] = field(default_factory=list)


@dataclass(kw_only=True)
class ChoiceDecl(ASTNode):
    """Choice declaration in kernel."""
    name: str
    choose_expr: Choose


# ==============================================================================
# Top-Level Declarations
# ==============================================================================

@dataclass(kw_only=True)
class Declaration(ASTNode):
    """Base class for top-level declarations."""
    name: str


@dataclass(kw_only=True)
class SensorDecl(Declaration):
    """Sensor binding: sensor name: Type @ channel(N)."""
    sensor_type: str
    channel: int


@dataclass(kw_only=True)
class ActuatorDecl(Declaration):
    """Actuator binding: actuator name: Type @ channel(N)."""
    actuator_type: str
    channel: int


@dataclass(kw_only=True)
class ParticipateBlock(ASTNode):
    """Participate block in creature."""
    bond_param: str  # Name of bond parameter
    body: Block


@dataclass(kw_only=True)
class AgencyBlock(ASTNode):
    """Agency-gated block in creature."""
    body: Block


@dataclass(kw_only=True)
class GraceBlock(ASTNode):
    """Grace handler block in creature."""
    body: Block


@dataclass
class ParamsBlock(ASTNode):
    """Parameters block."""
    params: dict[str, Expression] = field(default_factory=dict)


@dataclass(kw_only=True)
class Creature(Declaration):
    """Creature definition."""
    variables: list[VarDecl] = field(default_factory=list)
    sensors: list[SensorDecl] = field(default_factory=list)
    actuators: list[ActuatorDecl] = field(default_factory=list)
    participate: Optional[ParticipateBlock] = None
    agency: Optional[AgencyBlock] = None
    grace: Optional[GraceBlock] = None


@dataclass(kw_only=True)
class Kernel(Declaration):
    """Kernel definition (function as law module)."""
    ports: list[PortDecl] = field(default_factory=list)
    params: Optional[ParamsBlock] = None
    phases: list[PhaseBlock] = field(default_factory=list)


@dataclass(kw_only=True)
class Law(Declaration):
    """Law definition in bond."""
    parameters: list[ParamDecl] = field(default_factory=list)
    body: Block


@dataclass(kw_only=True)
class Bond(Declaration):
    """Bond definition."""
    params: Optional[ParamsBlock] = None
    laws: list[Law] = field(default_factory=list)


@dataclass
class CreaturesBlock(ASTNode):
    """Creatures block in presence."""
    creatures: dict[str, str] = field(default_factory=dict)  # name -> type


@dataclass
class BondsBlock(ASTNode):
    """Bonds block in presence."""
    bonds: list[tuple[str, str, str, str]] = field(default_factory=list)
    # (creature1, arrow_type, creature2, bond_type)


@dataclass(kw_only=True)
class InitBlock(ASTNode):
    """Init block in presence."""
    body: Block


@dataclass(kw_only=True)
class TickBlock(ASTNode):
    """Tick block in presence."""
    dt_param: str
    body: Block


@dataclass(kw_only=True)
class Presence(Declaration):
    """Presence definition (execution environment)."""
    creatures_block: Optional[CreaturesBlock] = None
    bonds_block: Optional[BondsBlock] = None
    init_block: Optional[InitBlock] = None
    tick_block: Optional[TickBlock] = None


@dataclass(kw_only=True)
class Boundary(Declaration):
    """Boundary definition for I/O."""
    body: Block


# ==============================================================================
# Program
# ==============================================================================

@dataclass
class Program(ASTNode):
    """Complete program."""
    declarations: list[Declaration] = field(default_factory=list)
    source_file: str = ""

    def get_creatures(self) -> list[Creature]:
        """Get all creature declarations."""
        return [d for d in self.declarations if isinstance(d, Creature)]

    def get_kernels(self) -> list[Kernel]:
        """Get all kernel declarations."""
        return [d for d in self.declarations if isinstance(d, Kernel)]

    def get_presences(self) -> list[Presence]:
        """Get all presence declarations."""
        return [d for d in self.declarations if isinstance(d, Presence)]

    def get_bonds(self) -> list[Bond]:
        """Get all bond declarations."""
        return [d for d in self.declarations if isinstance(d, Bond)]


# ==============================================================================
# Visitor Pattern
# ==============================================================================

class ASTVisitor:
    """Base visitor for AST traversal."""

    def visit(self, node: ASTNode):
        """Visit a node, dispatching to appropriate method."""
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.generic_visit)
        return method(node)

    def generic_visit(self, node: ASTNode):
        """Default visit - traverse children."""
        pass

    def visit_children(self, node: ASTNode):
        """Visit all child nodes."""
        for field_name in node.__dataclass_fields__:
            value = getattr(node, field_name)
            if isinstance(value, ASTNode):
                self.visit(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode):
                        self.visit(item)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, ASTNode):
                        self.visit(item)
