"""
Existence-Lang to Python Transpiler
===================================

Transpiles Existence-Lang AST to executable Python code
that uses the det.lang.runtime classes.
"""

from typing import Optional
from .ast_nodes import (
    Program, Declaration, Creature, Kernel, Bond, Presence, Boundary,
    Statement, Expression, Block, VarDecl, WitnessDecl, Assignment,
    ExpressionStmt, IfPast, RepeatPast, WhilePast, Return, KernelCall,
    InjectF, RequestGrace,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, IndexAccess,
    Distinct, This, Compare, Choose, TupleExpr,
    TypeKind, PortDirection,
    SensorDecl, ActuatorDecl, ParticipateBlock, AgencyBlock, GraceBlock,
    Proposal, PhaseBlock, Law, ASTVisitor
)


class Transpiler(ASTVisitor):
    """Transpiles Existence-Lang AST to Python code."""

    def __init__(self):
        self.indent_level = 0
        self.output_lines: list[str] = []
        self.creature_names: set[str] = set()
        self.kernel_names: set[str] = set()
        # Track creature member variables for self. prefix
        self.creature_members: set[str] = set()
        self.in_creature_method: bool = False
        # Track kernel ports for self. prefix
        self.kernel_ports: set[str] = set()
        self.in_kernel_method: bool = False

    def indent(self) -> str:
        """Get current indentation."""
        return "    " * self.indent_level

    def emit(self, line: str):
        """Emit a line of code."""
        self.output_lines.append(self.indent() + line)

    def emit_raw(self, line: str):
        """Emit without indentation."""
        self.output_lines.append(line)

    def transpile(self, program: Program) -> str:
        """Transpile a complete program."""
        # Header
        self.emit_raw("# Generated from Existence-Lang")
        self.emit_raw("# Agency-First Programming Language for DET-OS")
        self.emit_raw("")
        self.emit_raw("import math")
        self.emit_raw("import random")
        self.emit_raw("from det.lang.runtime import (")
        self.emit_raw("    CreatureBase, KernelBase, BondInstance,")
        self.emit_raw("    Register, TokenReg, SensorBinding, ActuatorBinding,")
        self.emit_raw("    ExistenceRuntime, distinct, transfer, diffuse,")
        self.emit_raw("    compare, reconcile, choose, local_seed,")
        self.emit_raw("    CompareResult, ReconcileResult, TransferResult, WriteResult")
        self.emit_raw(")")
        self.emit_raw("")

        # Collect names first
        for decl in program.declarations:
            if isinstance(decl, Creature):
                self.creature_names.add(decl.name)
            elif isinstance(decl, Kernel):
                self.kernel_names.add(decl.name)

        # Transpile declarations
        for decl in program.declarations:
            self.transpile_declaration(decl)
            self.emit_raw("")

        return "\n".join(self.output_lines)

    def transpile_declaration(self, decl: Declaration):
        """Transpile a declaration."""
        if isinstance(decl, Creature):
            self.transpile_creature(decl)
        elif isinstance(decl, Kernel):
            self.transpile_kernel(decl)
        elif isinstance(decl, Bond):
            self.transpile_bond(decl)
        elif isinstance(decl, Presence):
            self.transpile_presence(decl)
        elif isinstance(decl, Boundary):
            self.transpile_boundary(decl)

    # ==========================================================================
    # Creature
    # ==========================================================================

    def transpile_creature(self, creature: Creature):
        """Transpile creature to Python class."""
        # Track creature member variables
        self.creature_members = set()
        for var in creature.variables:
            self.creature_members.add(var.name)
        for sensor in creature.sensors:
            self.creature_members.add(sensor.name)
        for actuator in creature.actuators:
            self.creature_members.add(actuator.name)

        self.emit(f"class {creature.name}(CreatureBase):")
        self.indent_level += 1

        # __init__
        self.emit("def __init__(self, name: str = ''):")
        self.indent_level += 1
        self.emit("super().__init__(name)")
        self.emit("")

        # Variables
        for var in creature.variables:
            init_val = self.expr_to_python(var.initializer) if var.initializer else "None"
            self.emit(f"self.{var.name} = {init_val}")

        # Sensors
        for sensor in creature.sensors:
            self.emit(f"self._sensors['{sensor.name}'] = SensorBinding(")
            self.indent_level += 1
            self.emit(f"name='{sensor.name}',")
            self.emit(f"sensor_type='{sensor.sensor_type}',")
            self.emit(f"channel={sensor.channel}")
            self.indent_level -= 1
            self.emit(")")
            self.emit(f"self.{sensor.name} = self._sensors['{sensor.name}']")

        # Actuators
        for actuator in creature.actuators:
            self.emit(f"self._actuators['{actuator.name}'] = ActuatorBinding(")
            self.indent_level += 1
            self.emit(f"name='{actuator.name}',")
            self.emit(f"actuator_type='{actuator.actuator_type}',")
            self.emit(f"channel={actuator.channel}")
            self.indent_level -= 1
            self.emit(")")
            self.emit(f"self.{actuator.name} = self._actuators['{actuator.name}']")

        self.indent_level -= 1
        self.emit("")

        # Participate method
        if creature.participate:
            self.emit(f"def participate(self, {creature.participate.bond_param}):")
            self.indent_level += 1
            self.in_creature_method = True
            self.transpile_block(creature.participate.body)
            self.in_creature_method = False
            self.indent_level -= 1
            self.emit("")

        # Agency block method
        if creature.agency:
            self.emit("def agency_block(self):")
            self.indent_level += 1
            self.in_creature_method = True
            self.transpile_block(creature.agency.body)
            self.in_creature_method = False
            self.indent_level -= 1
            self.emit("")

        # Grace handler method
        if creature.grace:
            self.emit("def grace_handler(self):")
            self.indent_level += 1
            self.in_creature_method = True
            self.transpile_block(creature.grace.body)
            self.in_creature_method = False
            self.indent_level -= 1
            self.emit("")

        self.indent_level -= 1
        self.creature_members = set()

    # ==========================================================================
    # Kernel
    # ==========================================================================

    def transpile_kernel(self, kernel: Kernel):
        """Transpile kernel to Python class."""
        # Track kernel ports
        self.kernel_ports = set()
        for port in kernel.ports:
            self.kernel_ports.add(port.name)

        self.emit(f"class {kernel.name}(KernelBase):")
        self.indent_level += 1

        # __init__
        self.emit("def __init__(self):")
        self.indent_level += 1
        self.emit("super().__init__()")

        # Ports
        for port in kernel.ports:
            self.emit(f"self.{port.name} = Register()")

        # Params
        if kernel.params:
            for name, value in kernel.params.params.items():
                val_str = self.expr_to_python(value)
                self.emit(f"self._params['{name}'] = {val_str}")

        self.indent_level -= 1
        self.emit("")

        # Phase methods
        for phase in kernel.phases:
            self.emit(f"def phase_{phase.phase_name}(self):")
            self.indent_level += 1
            self.in_kernel_method = True

            # Proposals
            if phase.proposals:
                self.emit("proposals = {}")
                for proposal in phase.proposals:
                    self.emit(f"def proposal_{proposal.name}():")
                    self.indent_level += 1
                    if proposal.effect:
                        self.transpile_block(proposal.effect)
                    else:
                        self.emit("pass")
                    self.indent_level -= 1
                    score = self.expr_to_python(proposal.score) if proposal.score else "1.0"
                    self.emit(f"proposals['{proposal.name}'] = (proposal_{proposal.name}, {score})")

            # Choice
            if phase.choice:
                props = phase.choice.choose_expr.proposals
                props_str = "[" + ", ".join(f"'{p}'" for p in props) + "]"
                dec = self.expr_to_python(phase.choice.choose_expr.decisiveness) if phase.choice.choose_expr.decisiveness else "1.0"
                self.emit(f"{phase.choice.name} = choose({props_str}, decisiveness={dec})")

            # Commit
            if phase.commit:
                self.emit(f"if '{phase.commit}' in proposals:")
                self.indent_level += 1
                self.emit(f"proposals['{phase.commit}'][0]()")
                self.indent_level -= 1

            # Other statements
            for stmt in phase.statements:
                self.transpile_statement(stmt)

            if not phase.proposals and not phase.choice and not phase.statements:
                self.emit("pass")

            self.in_kernel_method = False
            self.indent_level -= 1
            self.emit("")

        # Execute method
        self.emit("def execute(self):")
        self.indent_level += 1
        for phase in kernel.phases:
            self.emit(f"self.phase_{phase.phase_name}()")
        if not kernel.phases:
            self.emit("pass")
        self.indent_level -= 1

        self.indent_level -= 1
        self.kernel_ports = set()

    # ==========================================================================
    # Bond
    # ==========================================================================

    def transpile_bond(self, bond: Bond):
        """Transpile bond definition."""
        self.emit(f"class {bond.name}Bond(BondInstance):")
        self.indent_level += 1

        # __init__
        self.emit("def __init__(self, creature_a, creature_b):")
        self.indent_level += 1
        self.emit("super().__init__(creature_a, creature_b)")

        # Params
        if bond.params:
            for name, value in bond.params.params.items():
                val_str = self.expr_to_python(value)
                self.emit(f"self.{name} = {val_str}")

        self.indent_level -= 1
        self.emit("")

        # Laws
        for law in bond.laws:
            params_str = ", ".join(f"{p.name}" for p in law.parameters)
            self.emit(f"def {law.name}(self, {params_str}):")
            self.indent_level += 1
            self.transpile_block(law.body)
            self.indent_level -= 1
            self.emit("")

        self.indent_level -= 1

    # ==========================================================================
    # Presence
    # ==========================================================================

    def transpile_presence(self, presence: Presence):
        """Transpile presence to setup function."""
        self.emit(f"def setup_{presence.name}(runtime):")
        self.indent_level += 1

        # Create creatures
        if presence.creatures_block:
            for name, ctype in presence.creatures_block.creatures.items():
                self.emit(f"runtime.register_creature_type('{ctype}', {ctype})")
                self.emit(f"runtime.spawn_creature('{name}', '{ctype}')")

        # Create bonds
        if presence.bonds_block:
            for c1, arrow, c2, btype in presence.bonds_block.bonds:
                self.emit(f"runtime.create_bond('{c1}', '{c2}', '{btype}')")

        # Init block
        if presence.init_block:
            self.transpile_block(presence.init_block.body, context="runtime")

        self.emit("return runtime")
        self.indent_level -= 1

    # ==========================================================================
    # Boundary
    # ==========================================================================

    def transpile_boundary(self, boundary: Boundary):
        """Transpile boundary definition."""
        self.emit(f"class {boundary.name}Boundary:")
        self.indent_level += 1
        self.emit("def __init__(self):")
        self.indent_level += 1
        self.emit("pass")
        self.indent_level -= 1
        self.indent_level -= 1

    # ==========================================================================
    # Statements
    # ==========================================================================

    def transpile_block(self, block: Block, context: str = "self"):
        """Transpile a block of statements."""
        if not block.statements:
            self.emit("pass")
            return

        for stmt in block.statements:
            self.transpile_statement(stmt, context)

    def transpile_statement(self, stmt: Statement, context: str = "self"):
        """Transpile a statement."""
        if isinstance(stmt, VarDecl):
            init_val = self.expr_to_python(stmt.initializer) if stmt.initializer else "None"
            self.emit(f"{stmt.name} = {init_val}")

        elif isinstance(stmt, WitnessDecl):
            expr = self.expr_to_python(stmt.expression)
            self.emit(f"{stmt.name} = {expr}")

        elif isinstance(stmt, Assignment):
            target = self.expr_to_python(stmt.target)
            value = self.expr_to_python(stmt.value)
            if stmt.operator == "=":
                self.emit(f"{target} = {value}")
            else:
                self.emit(f"{target} {stmt.operator} {value}")

        elif isinstance(stmt, ExpressionStmt):
            expr = self.expr_to_python(stmt.expression)
            self.emit(expr)

        elif isinstance(stmt, IfPast):
            self.transpile_if_past(stmt, context)

        elif isinstance(stmt, RepeatPast):
            count = self.expr_to_python(stmt.count)
            self.emit(f"for _i in range(int({count})):")
            self.indent_level += 1
            self.transpile_block(stmt.body, context)
            self.indent_level -= 1

        elif isinstance(stmt, WhilePast):
            cond = self.expr_to_python(stmt.condition)
            self.emit(f"while {cond}:")
            self.indent_level += 1
            self.transpile_block(stmt.body, context)
            self.indent_level -= 1

        elif isinstance(stmt, Return):
            if stmt.value:
                val = self.expr_to_python(stmt.value)
                self.emit(f"return {val}")
            else:
                self.emit("return")

        elif isinstance(stmt, KernelCall):
            self.emit(f"_kernel = {stmt.kernel_name}()")
            for port, value in stmt.port_bindings.items():
                val_str = self.expr_to_python(value)
                self.emit(f"_kernel.{port} = {val_str}")
            self.emit("_kernel.execute()")

        elif isinstance(stmt, InjectF):
            target = self.expr_to_python(stmt.target)
            amount = self.expr_to_python(stmt.amount)
            if context == "runtime":
                self.emit(f"runtime.inject_F('{target}', {amount})")
            else:
                self.emit(f"{target}.F += {amount}")

        elif isinstance(stmt, RequestGrace):
            amount = self.expr_to_python(stmt.amount)
            self.emit(f"# Request grace: {amount}")

    def transpile_if_past(self, stmt: IfPast, context: str):
        """Transpile if_past statement."""
        cond = self.expr_to_python(stmt.condition)
        self.emit(f"if {cond}:")
        self.indent_level += 1
        self.transpile_block(stmt.then_block, context)
        self.indent_level -= 1

        if stmt.else_block:
            if isinstance(stmt.else_block, IfPast):
                cond2 = self.expr_to_python(stmt.else_block.condition)
                self.emit(f"elif {cond2}:")
                self.indent_level += 1
                self.transpile_block(stmt.else_block.then_block, context)
                self.indent_level -= 1
                if stmt.else_block.else_block:
                    self.emit("else:")
                    self.indent_level += 1
                    if isinstance(stmt.else_block.else_block, Block):
                        self.transpile_block(stmt.else_block.else_block, context)
                    self.indent_level -= 1
            elif isinstance(stmt.else_block, Block):
                self.emit("else:")
                self.indent_level += 1
                self.transpile_block(stmt.else_block, context)
                self.indent_level -= 1

    # ==========================================================================
    # Expressions
    # ==========================================================================

    def expr_to_python(self, expr: Expression) -> str:
        """Convert expression to Python code string."""
        if expr is None:
            return "None"

        if isinstance(expr, Literal):
            if isinstance(expr.value, str):
                return f"'{expr.value}'"
            elif isinstance(expr.value, bool):
                return "True" if expr.value else "False"
            return str(expr.value)

        elif isinstance(expr, Identifier):
            name = expr.name
            # Special cases - result enums
            if name in ("LT", "EQ", "GT"):
                return f"CompareResult.{name}"
            if name in ("EQ_OK", "EQ_FAIL", "EQ_REFUSE"):
                return f"ReconcileResult.{name}"
            if name in ("TRANSFER_OK", "TRANSFER_PARTIAL", "TRANSFER_BLOCKED"):
                return f"TransferResult.{name}"
            if name in ("WRITE_OK", "WRITE_REFUSED", "WRITE_PARTIAL"):
                return f"WriteResult.{name}"
            # Creature member variables need self. prefix
            if self.in_creature_method and name in self.creature_members:
                return f"self.{name}"
            # Kernel ports need self. prefix in phase methods
            if self.in_kernel_method and name in self.kernel_ports:
                return f"self.{name}"
            return name

        elif isinstance(expr, BinaryOp):
            left = self.expr_to_python(expr.left)
            right = self.expr_to_python(expr.right)
            op = expr.operator
            # Map operators
            if op == "==":
                return f"({left} == {right})"
            elif op == "===":
                return f"({left} is {right})"
            elif op == "≡":
                return f"({left} is {right})"
            return f"({left} {op} {right})"

        elif isinstance(expr, UnaryOp):
            operand = self.expr_to_python(expr.operand)
            if expr.operator == "!":
                return f"(not {operand})"
            return f"({expr.operator}{operand})"

        elif isinstance(expr, Call):
            callee = self.expr_to_python(expr.callee)
            args = ", ".join(self.expr_to_python(a) for a in expr.arguments)

            # Special function mappings
            if callee == "soma_read":
                return f"{args}.read().token"
            elif callee == "soma_write":
                parts = [self.expr_to_python(a) for a in expr.arguments]
                return f"{parts[0]}.set({parts[1]}, self.a)"
            elif callee == "sin":
                return f"math.sin({args})"
            elif callee == "cos":
                return f"math.cos({args})"
            elif callee == "abs":
                return f"abs({args})"
            elif callee == "sqrt":
                return f"math.sqrt({args})"
            elif callee == "clamp":
                parts = [self.expr_to_python(a) for a in expr.arguments]
                return f"max({parts[1]}, min({parts[2]}, {parts[0]}))"
            elif callee == "relu":
                return f"max(0, {args})"

            return f"{callee}({args})"

        elif isinstance(expr, FieldAccess):
            obj = self.expr_to_python(expr.object)
            return f"{obj}.{expr.field}"

        elif isinstance(expr, IndexAccess):
            obj = self.expr_to_python(expr.object)
            idx = self.expr_to_python(expr.index)
            return f"{obj}[{idx}]"

        elif isinstance(expr, Distinct):
            return "distinct()"

        elif isinstance(expr, This):
            return "self"

        elif isinstance(expr, Compare):
            left = self.expr_to_python(expr.left)
            right = self.expr_to_python(expr.right)
            if expr.epsilon:
                eps = self.expr_to_python(expr.epsilon)
                return f"compare({left}, {right}, {eps})"
            return f"compare({left}, {right})"

        elif isinstance(expr, Choose):
            props = "[" + ", ".join(f"'{p}'" for p in expr.proposals) + "]"
            dec = self.expr_to_python(expr.decisiveness) if expr.decisiveness else "1.0"
            return f"choose({props}, decisiveness={dec})"

        elif isinstance(expr, TupleExpr):
            elems = ", ".join(self.expr_to_python(e) for e in expr.elements)
            return f"({elems})"

        return "None"


def transpile(source: str) -> str:
    """Convenience function to transpile source code."""
    from .parser import parse
    ast = parse(source)
    transpiler = Transpiler()
    return transpiler.transpile(ast)
