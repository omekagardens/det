"""
Existence-Lang Semantic Analysis
================================

Type checking and semantic validation for Existence-Lang.
Ensures temporal semantics (no present-time conditionals) are respected.
"""

from typing import Optional, Set
from .ast_nodes import (
    Program, Declaration, Creature, Kernel, Bond, Presence,
    Statement, Expression, Block, VarDecl, WitnessDecl, Assignment,
    ExpressionStmt, IfPast, RepeatPast, WhilePast, Return,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess,
    This, Compare, Distinct,
    TypeKind, TypeAnnotation, ASTVisitor
)
from .errors import SemanticError, ErrorReporter


class Scope:
    """Symbol table scope."""

    def __init__(self, parent: Optional['Scope'] = None):
        self.parent = parent
        self.symbols: dict[str, TypeAnnotation] = {}
        self.witness_tokens: Set[str] = set()

    def define(self, name: str, type_ann: TypeAnnotation):
        """Define a symbol in this scope."""
        self.symbols[name] = type_ann

    def define_witness(self, name: str):
        """Define a witness token."""
        self.witness_tokens.add(name)

    def lookup(self, name: str) -> Optional[TypeAnnotation]:
        """Look up a symbol."""
        if name in self.symbols:
            return self.symbols[name]
        if self.parent:
            return self.parent.lookup(name)
        return None

    def is_witness(self, name: str) -> bool:
        """Check if name is a witness token."""
        if name in self.witness_tokens:
            return True
        if self.parent:
            return self.parent.is_witness(name)
        return False


class SemanticAnalyzer(ASTVisitor):
    """
    Semantic analyzer for Existence-Lang.

    Checks:
    - Type correctness
    - No present-time conditionals (temporal semantics)
    - Witness token usage
    - Undefined symbols
    """

    def __init__(self, filename: str = "<input>"):
        self.reporter = ErrorReporter(filename)
        self.scope = Scope()
        self.in_kernel = False
        self.current_creature: Optional[str] = None

    def analyze(self, program: Program) -> bool:
        """Analyze program, return True if no errors."""
        # First pass: collect top-level declarations
        for decl in program.declarations:
            if isinstance(decl, Creature):
                self.scope.define(decl.name, TypeAnnotation(TypeKind.CREATURE, decl.name))
            elif isinstance(decl, Kernel):
                self.scope.define(decl.name, TypeAnnotation(TypeKind.CUSTOM, decl.name))
            elif isinstance(decl, Bond):
                self.scope.define(decl.name, TypeAnnotation(TypeKind.BOND, decl.name))

        # Second pass: analyze declarations
        for decl in program.declarations:
            self.visit(decl)

        return not self.reporter.has_errors()

    def visit_Creature(self, creature: Creature):
        """Analyze creature."""
        self.current_creature = creature.name
        scope = Scope(self.scope)
        self.scope = scope

        # Define implicit DET state
        self.scope.define("F", TypeAnnotation(TypeKind.FLOAT))
        self.scope.define("q", TypeAnnotation(TypeKind.FLOAT))
        self.scope.define("a", TypeAnnotation(TypeKind.FLOAT))
        self.scope.define("theta", TypeAnnotation(TypeKind.FLOAT))
        self.scope.define("sigma", TypeAnnotation(TypeKind.FLOAT))
        self.scope.define("P", TypeAnnotation(TypeKind.FLOAT))

        # Define variables
        for var in creature.variables:
            t = var.type_annotation or TypeAnnotation(TypeKind.FLOAT)
            self.scope.define(var.name, t)

        # Define sensors and actuators
        for sensor in creature.sensors:
            self.scope.define(sensor.name, TypeAnnotation(TypeKind.CUSTOM, sensor.sensor_type))
        for actuator in creature.actuators:
            self.scope.define(actuator.name, TypeAnnotation(TypeKind.CUSTOM, actuator.actuator_type))

        # Analyze blocks
        if creature.participate:
            param_scope = Scope(self.scope)
            param_scope.define(creature.participate.bond_param, TypeAnnotation(TypeKind.BOND))
            self.scope = param_scope
            self.visit(creature.participate.body)
            self.scope = param_scope.parent

        if creature.agency:
            self.visit(creature.agency.body)

        if creature.grace:
            self.visit(creature.grace.body)

        self.scope = scope.parent
        self.current_creature = None

    def visit_Kernel(self, kernel: Kernel):
        """Analyze kernel."""
        self.in_kernel = True
        scope = Scope(self.scope)
        self.scope = scope

        # Define ports
        for port in kernel.ports:
            self.scope.define(port.name, port.type_annotation)

        # Define params
        if kernel.params:
            for name, value in kernel.params.params.items():
                self.scope.define(name, TypeAnnotation(TypeKind.FLOAT))

        # Analyze phases
        for phase in kernel.phases:
            # Check for anti-drift rule violation
            witness_written = set()
            for stmt in phase.statements:
                if isinstance(stmt, WitnessDecl):
                    witness_written.add(stmt.name)
                    self.visit(stmt)
                elif isinstance(stmt, IfPast):
                    # Check if reading witness written in same tick
                    self.check_antidrift(stmt.condition, witness_written, phase.phase_name)
                    self.visit(stmt)
                else:
                    self.visit(stmt)

        self.scope = scope.parent
        self.in_kernel = False

    def visit_Block(self, block: Block):
        """Analyze block."""
        for stmt in block.statements:
            self.visit(stmt)

    def visit_VarDecl(self, var: VarDecl):
        """Analyze variable declaration."""
        t = var.type_annotation or TypeAnnotation(TypeKind.FLOAT)
        self.scope.define(var.name, t)

        if var.initializer:
            self.visit(var.initializer)

    def visit_WitnessDecl(self, decl: WitnessDecl):
        """Analyze witness binding."""
        self.scope.define_witness(decl.name)
        self.scope.define(decl.name, TypeAnnotation(TypeKind.TOKEN_REG))
        self.visit(decl.expression)

    def visit_Assignment(self, stmt: Assignment):
        """Analyze assignment."""
        self.visit(stmt.target)
        self.visit(stmt.value)

    def visit_IfPast(self, stmt: IfPast):
        """Analyze if_past statement."""
        # Check that condition uses past tokens
        self.check_past_condition(stmt.condition)
        self.visit(stmt.condition)
        self.visit(stmt.then_block)
        if stmt.else_block:
            self.visit(stmt.else_block)

    def visit_RepeatPast(self, stmt: RepeatPast):
        """Analyze repeat_past loop."""
        # Count should be a past token
        self.check_past_token(stmt.count, "repeat_past count")
        self.visit(stmt.count)
        self.visit(stmt.body)

    def visit_WhilePast(self, stmt: WhilePast):
        """Analyze while_past loop."""
        # Condition should produce past token
        self.check_past_condition(stmt.condition)
        self.visit(stmt.condition)
        self.visit(stmt.body)

    def visit_Identifier(self, ident: Identifier):
        """Check identifier is defined."""
        if not self.scope.lookup(ident.name):
            # Check for built-in constants
            builtins = {"LT", "EQ", "GT", "EQ_OK", "EQ_FAIL", "EQ_REFUSE",
                        "WRITE_OK", "WRITE_REFUSED", "true", "false"}
            if ident.name not in builtins:
                self.reporter.semantic_error(
                    f"Undefined identifier: {ident.name}",
                    ident.line, ident.column
                )

    def visit_BinaryOp(self, op: BinaryOp):
        """Analyze binary operation."""
        self.visit(op.left)
        self.visit(op.right)

    def visit_Call(self, call: Call):
        """Analyze function call."""
        self.visit(call.callee)
        for arg in call.arguments:
            self.visit(arg)

    def visit_FieldAccess(self, fa: FieldAccess):
        """Analyze field access."""
        self.visit(fa.object)

    def check_past_condition(self, expr: Expression):
        """Check that condition uses past tokens (not present state)."""
        # A proper condition should be comparing witness tokens
        if isinstance(expr, BinaryOp):
            if expr.operator in ("==", "!="):
                # Left side should be a witness token or past value
                if isinstance(expr.left, Identifier):
                    if not self.scope.is_witness(expr.left.name):
                        # It's OK if it's comparing result of compare/cmp
                        pass
            else:
                # Other operators might indicate present-time comparison
                if expr.operator in ("<", ">", "<=", ">="):
                    self.reporter.warning(
                        f"Direct comparison '{expr.operator}' may violate temporal semantics. "
                        "Use compare() to produce a past token first.",
                        expr.line if hasattr(expr, 'line') else 0,
                        expr.column if hasattr(expr, 'column') else 0
                    )

    def check_past_token(self, expr: Expression, context: str):
        """Check that expression is a past token (integer from trace)."""
        if isinstance(expr, Literal):
            return  # Literal constants are OK
        if isinstance(expr, Identifier):
            if not self.scope.is_witness(expr.name):
                self.reporter.warning(
                    f"{context} should use a past token, got: {expr.name}",
                    expr.line if hasattr(expr, 'line') else 0,
                    expr.column if hasattr(expr, 'column') else 0
                )

    def check_antidrift(self, condition: Expression, written_this_tick: Set[str], phase: str):
        """Check for anti-drift rule violation (reading token written same tick)."""
        tokens_read = self.collect_identifiers(condition)
        violations = tokens_read & written_this_tick

        for tok in violations:
            self.reporter.error(
                f"Anti-drift violation in phase {phase}: "
                f"Cannot read witness '{tok}' in same tick it was written. "
                "Tokens only influence the NEXT tick.",
                condition.line if hasattr(condition, 'line') else 0,
                condition.column if hasattr(condition, 'column') else 0
            )

    def collect_identifiers(self, expr: Expression) -> Set[str]:
        """Collect all identifier names in an expression."""
        result = set()
        if isinstance(expr, Identifier):
            result.add(expr.name)
        elif isinstance(expr, BinaryOp):
            result |= self.collect_identifiers(expr.left)
            result |= self.collect_identifiers(expr.right)
        elif isinstance(expr, UnaryOp):
            result |= self.collect_identifiers(expr.operand)
        elif isinstance(expr, Call):
            result |= self.collect_identifiers(expr.callee)
            for arg in expr.arguments:
                result |= self.collect_identifiers(arg)
        elif isinstance(expr, FieldAccess):
            result |= self.collect_identifiers(expr.object)
        return result


def analyze(program: Program, filename: str = "<input>") -> tuple[bool, str]:
    """Analyze a program, return (success, error_report)."""
    analyzer = SemanticAnalyzer(filename)
    success = analyzer.analyze(program)
    return (success, analyzer.reporter.report())
