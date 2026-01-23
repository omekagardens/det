"""
Existence-Lang Recursive Descent Parser
=======================================

Hand-written recursive descent parser for better error messages and control.
Parses source into AST for subsequent transpilation.
"""

from typing import Optional, Callable
from .tokens import Token, TokenType, Lexer
from .ast_nodes import (
    Program, Declaration, Creature, Kernel, Bond, Presence, Boundary,
    Statement, Expression, Block, VarDecl, WitnessDecl, Assignment,
    ExpressionStmt, IfPast, RepeatPast, WhilePast, Return, KernelCall,
    SomaRead, SomaWrite, InjectF, RequestGrace,
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, IndexAccess,
    Distinct, WitnessToken, This, Compare, Choose, TupleExpr,
    TypeAnnotation, TypeKind, PortDecl, PortDirection, ParamDecl,
    Proposal, PhaseBlock, ChoiceDecl,
    SensorDecl, ActuatorDecl, ParticipateBlock, AgencyBlock, GraceBlock,
    ParamsBlock, CreaturesBlock, BondsBlock, InitBlock, TickBlock, Law
)
from .errors import ParseError, ErrorReporter


class Parser:
    """Recursive descent parser for Existence-Lang."""

    def __init__(self, source: str, filename: str = "<input>"):
        self.lexer = Lexer(source, filename)
        self.tokens = self.lexer.tokenize()
        self.pos = 0
        self.reporter = ErrorReporter(filename)

    def current(self) -> Token:
        """Get current token."""
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # EOF

    def peek(self, offset: int = 0) -> Token:
        """Peek at token at current position + offset."""
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return self.tokens[-1]

    def advance(self) -> Token:
        """Advance and return current token."""
        tok = self.current()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return tok

    def check(self, *types: TokenType) -> bool:
        """Check if current token is one of the given types."""
        return self.current().type in types

    def match(self, *types: TokenType) -> Optional[Token]:
        """If current token matches, consume and return it."""
        if self.check(*types):
            return self.advance()
        return None

    def expect(self, token_type: TokenType, message: str = "") -> Token:
        """Expect a specific token type, error if not found."""
        tok = self.current()
        if tok.type != token_type:
            msg = message or f"Expected {token_type.name}, got {tok.type.name}"
            raise self.reporter.parse_error(msg, tok.line, tok.column)
        return self.advance()

    def error(self, message: str) -> ParseError:
        """Create a parse error at current position."""
        tok = self.current()
        return self.reporter.parse_error(message, tok.line, tok.column)

    # ==========================================================================
    # Program
    # ==========================================================================

    def parse(self) -> Program:
        """Parse entire program."""
        declarations = []
        while not self.check(TokenType.EOF):
            decl = self.parse_declaration()
            if decl:
                declarations.append(decl)
        return Program(declarations=declarations)

    def parse_declaration(self) -> Optional[Declaration]:
        """Parse a top-level declaration."""
        tok = self.current()

        if self.check(TokenType.CREATURE):
            return self.parse_creature()
        elif self.check(TokenType.KERNEL):
            return self.parse_kernel()
        elif self.check(TokenType.BOND):
            return self.parse_bond()
        elif self.check(TokenType.PRESENCE):
            return self.parse_presence()
        elif self.check(TokenType.BOUNDARY):
            return self.parse_boundary()
        else:
            raise self.error(f"Expected declaration, got {tok.type.name}")

    # ==========================================================================
    # Creature
    # ==========================================================================

    def parse_creature(self) -> Creature:
        """Parse creature declaration."""
        tok = self.expect(TokenType.CREATURE)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected creature name")
        self.expect(TokenType.LBRACE, "Expected '{' after creature name")

        creature = Creature(
            name=name_tok.value,
            line=tok.line,
            column=tok.column
        )

        while not self.check(TokenType.RBRACE, TokenType.EOF):
            if self.check(TokenType.VAR):
                creature.variables.append(self.parse_var_decl())
            elif self.check(TokenType.SENSOR):
                creature.sensors.append(self.parse_sensor_decl())
            elif self.check(TokenType.ACTUATOR):
                creature.actuators.append(self.parse_actuator_decl())
            elif self.check(TokenType.PARTICIPATE):
                creature.participate = self.parse_participate_block()
            elif self.check(TokenType.AGENCY):
                creature.agency = self.parse_agency_block()
            elif self.check(TokenType.GRACE):
                creature.grace = self.parse_grace_block()
            else:
                raise self.error(f"Unexpected token in creature body: {self.current().type.name}")

        self.expect(TokenType.RBRACE, "Expected '}' to close creature")
        return creature

    def parse_var_decl(self) -> VarDecl:
        """Parse variable declaration."""
        tok = self.expect(TokenType.VAR)
        name_tok = self.expect(TokenType.IDENTIFIER, "Expected variable name")

        type_ann = None
        if self.match(TokenType.COLON):
            type_ann = self.parse_type_annotation()

        initializer = None
        is_alias = False
        if self.match(TokenType.ALIAS_EQ):
            is_alias = True
            initializer = self.parse_expression()
        elif self.match(TokenType.RECONCILE_EQ):
            initializer = self.parse_expression()

        self.match(TokenType.SEMICOLON)

        return VarDecl(
            name=name_tok.value,
            type_annotation=type_ann,
            initializer=initializer,
            is_alias=is_alias,
            line=tok.line,
            column=tok.column
        )

    def parse_sensor_decl(self) -> SensorDecl:
        """Parse sensor declaration: sensor name: Type @ channel(N);"""
        tok = self.expect(TokenType.SENSOR)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.COLON)
        type_tok = self.expect(TokenType.IDENTIFIER, "Expected sensor type")
        self.expect(TokenType.AT)
        self.expect(TokenType.CHANNEL)
        self.expect(TokenType.LPAREN)
        channel_tok = self.expect(TokenType.INTEGER, "Expected channel number")
        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)

        return SensorDecl(
            name=name_tok.value,
            sensor_type=type_tok.value,
            channel=int(channel_tok.value),
            line=tok.line,
            column=tok.column
        )

    def parse_actuator_decl(self) -> ActuatorDecl:
        """Parse actuator declaration."""
        tok = self.expect(TokenType.ACTUATOR)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.COLON)
        type_tok = self.expect(TokenType.IDENTIFIER, "Expected actuator type")
        self.expect(TokenType.AT)
        self.expect(TokenType.CHANNEL)
        self.expect(TokenType.LPAREN)
        channel_tok = self.expect(TokenType.INTEGER, "Expected channel number")
        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)

        return ActuatorDecl(
            name=name_tok.value,
            actuator_type=type_tok.value,
            channel=int(channel_tok.value),
            line=tok.line,
            column=tok.column
        )

    def parse_participate_block(self) -> ParticipateBlock:
        """Parse participate block."""
        tok = self.expect(TokenType.PARTICIPATE)
        self.expect(TokenType.LPAREN)
        bond_param = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.COLON)
        self.expect(TokenType.IDENTIFIER)  # Bond type, ignore for now
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return ParticipateBlock(bond_param=bond_param, body=body, line=tok.line, column=tok.column)

    def parse_agency_block(self) -> AgencyBlock:
        """Parse agency block."""
        tok = self.expect(TokenType.AGENCY)
        body = self.parse_block()
        return AgencyBlock(body=body, line=tok.line, column=tok.column)

    def parse_grace_block(self) -> GraceBlock:
        """Parse grace block."""
        tok = self.expect(TokenType.GRACE)
        body = self.parse_block()
        return GraceBlock(body=body, line=tok.line, column=tok.column)

    # ==========================================================================
    # Kernel
    # ==========================================================================

    def parse_kernel(self) -> Kernel:
        """Parse kernel declaration."""
        tok = self.expect(TokenType.KERNEL)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACE)

        kernel = Kernel(name=name_tok.value, line=tok.line, column=tok.column)

        while not self.check(TokenType.RBRACE, TokenType.EOF):
            if self.check(TokenType.IN, TokenType.OUT, TokenType.INOUT):
                kernel.ports.append(self.parse_port_decl())
            elif self.check(TokenType.PARAMS):
                kernel.params = self.parse_params_block()
            elif self.check(TokenType.PHASE):
                kernel.phases.append(self.parse_phase_block())
            else:
                raise self.error(f"Unexpected token in kernel: {self.current().type.name}")

        self.expect(TokenType.RBRACE)
        return kernel

    def parse_port_decl(self) -> PortDecl:
        """Parse port declaration: in/out/inout name: Type;"""
        tok = self.current()

        if self.match(TokenType.IN):
            direction = PortDirection.IN
        elif self.match(TokenType.OUT):
            direction = PortDirection.OUT
        elif self.match(TokenType.INOUT):
            direction = PortDirection.INOUT
        else:
            raise self.error("Expected port direction (in/out/inout)")

        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.COLON)
        type_ann = self.parse_type_annotation()
        self.match(TokenType.SEMICOLON)

        return PortDecl(
            name=name_tok.value,
            direction=direction,
            type_annotation=type_ann,
            line=tok.line,
            column=tok.column
        )

    def parse_params_block(self) -> ParamsBlock:
        """Parse params block: params { k = v; ... }"""
        tok = self.expect(TokenType.PARAMS)
        self.expect(TokenType.LBRACE)

        params = {}
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            name_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.RECONCILE_EQ)
            value = self.parse_expression()
            self.match(TokenType.SEMICOLON)
            params[name_tok.value] = value

        self.expect(TokenType.RBRACE)
        return ParamsBlock(params=params, line=tok.line, column=tok.column)

    def parse_phase_block(self) -> PhaseBlock:
        """Parse phase block: phase NAME { ... }"""
        tok = self.expect(TokenType.PHASE)
        phase_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LBRACE)

        phase = PhaseBlock(phase_name=phase_name, line=tok.line, column=tok.column)

        while not self.check(TokenType.RBRACE, TokenType.EOF):
            if self.check(TokenType.PROPOSAL):
                phase.proposals.append(self.parse_proposal())
            elif self.check(TokenType.CHOICE):
                phase.choice = self.parse_choice_decl()
            elif self.check(TokenType.COMMIT):
                self.advance()
                if self.check(TokenType.IDENTIFIER):
                    phase.commit = self.advance().value
                self.match(TokenType.SEMICOLON)
            else:
                # Regular statement
                phase.statements.append(self.parse_statement())

        self.expect(TokenType.RBRACE)
        return phase

    def parse_proposal(self) -> Proposal:
        """Parse proposal block."""
        tok = self.expect(TokenType.PROPOSAL)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACE)

        proposal = Proposal(name=name_tok.value, line=tok.line, column=tok.column)

        while not self.check(TokenType.RBRACE, TokenType.EOF):
            if self.check(TokenType.SCORE):
                self.advance()
                self.expect(TokenType.RECONCILE_EQ)
                proposal.score = self.parse_expression()
                self.match(TokenType.SEMICOLON)
            elif self.check(TokenType.EFFECT):
                self.advance()
                proposal.effect = self.parse_block()
            else:
                raise self.error(f"Unexpected in proposal: {self.current().type.name}")

        self.expect(TokenType.RBRACE)
        return proposal

    def parse_choice_decl(self) -> ChoiceDecl:
        """Parse choice declaration."""
        tok = self.expect(TokenType.CHOICE)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.RECONCILE_EQ)
        self.expect(TokenType.CHOOSE)
        self.expect(TokenType.LPAREN)

        # Parse proposals set
        self.expect(TokenType.LBRACE)
        proposals = []
        while not self.check(TokenType.RBRACE):
            proposals.append(self.expect(TokenType.IDENTIFIER).value)
            if not self.match(TokenType.COMMA):
                break
        self.expect(TokenType.RBRACE)

        # Parse optional parameters
        decisiveness = None
        seed = None
        while self.match(TokenType.COMMA):
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.RECONCILE_EQ)
            value = self.parse_expression()
            if param_name == "decisiveness":
                decisiveness = value
            elif param_name == "seed":
                seed = value

        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)

        choose = Choose(proposals=proposals, decisiveness=decisiveness, seed=seed)
        return ChoiceDecl(name=name_tok.value, choose_expr=choose, line=tok.line, column=tok.column)

    # ==========================================================================
    # Bond
    # ==========================================================================

    def parse_bond(self) -> Bond:
        """Parse bond declaration."""
        tok = self.expect(TokenType.BOND)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACE)

        bond = Bond(name=name_tok.value, line=tok.line, column=tok.column)

        while not self.check(TokenType.RBRACE, TokenType.EOF):
            if self.check(TokenType.PARAMS):
                bond.params = self.parse_params_block()
            elif self.check(TokenType.LAW):
                bond.laws.append(self.parse_law())
            else:
                raise self.error(f"Unexpected in bond: {self.current().type.name}")

        self.expect(TokenType.RBRACE)
        return bond

    def parse_law(self) -> Law:
        """Parse law definition."""
        tok = self.expect(TokenType.LAW)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LPAREN)

        params = []
        while not self.check(TokenType.RPAREN):
            param_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            param_type = self.parse_type_annotation()
            params.append(ParamDecl(name=param_name, type_annotation=param_type))
            if not self.match(TokenType.COMMA):
                break
        self.expect(TokenType.RPAREN)

        body = self.parse_block()
        return Law(name=name_tok.value, parameters=params, body=body, line=tok.line, column=tok.column)

    # ==========================================================================
    # Presence
    # ==========================================================================

    def parse_presence(self) -> Presence:
        """Parse presence declaration."""
        tok = self.expect(TokenType.PRESENCE)
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.LBRACE)

        presence = Presence(name=name_tok.value, line=tok.line, column=tok.column)

        while not self.check(TokenType.RBRACE, TokenType.EOF):
            if self.check(TokenType.CREATURES):
                presence.creatures_block = self.parse_creatures_block()
            elif self.check(TokenType.BONDS):
                presence.bonds_block = self.parse_bonds_block()
            elif self.check(TokenType.INIT):
                presence.init_block = self.parse_init_block()
            else:
                raise self.error(f"Unexpected in presence: {self.current().type.name}")

        self.expect(TokenType.RBRACE)
        return presence

    def parse_creatures_block(self) -> CreaturesBlock:
        """Parse creatures block."""
        tok = self.expect(TokenType.CREATURES)
        self.expect(TokenType.LBRACE)

        creatures = {}
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            name_tok = self.expect(TokenType.IDENTIFIER)
            self.expect(TokenType.COLON)
            type_tok = self.expect(TokenType.IDENTIFIER)
            self.match(TokenType.SEMICOLON)
            creatures[name_tok.value] = type_tok.value

        self.expect(TokenType.RBRACE)
        return CreaturesBlock(creatures=creatures, line=tok.line, column=tok.column)

    def parse_bonds_block(self) -> BondsBlock:
        """Parse bonds block."""
        tok = self.expect(TokenType.BONDS)
        self.expect(TokenType.LBRACE)

        bonds = []
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            c1 = self.expect(TokenType.IDENTIFIER).value
            if self.match(TokenType.BOND_ARROW):
                arrow = "<->"
            elif self.match(TokenType.ARROW):
                arrow = "->"
            else:
                raise self.error("Expected bond arrow (<-> or ->)")
            c2 = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            bond_type = self.expect(TokenType.IDENTIFIER).value
            self.match(TokenType.SEMICOLON)
            bonds.append((c1, arrow, c2, bond_type))

        self.expect(TokenType.RBRACE)
        return BondsBlock(bonds=bonds, line=tok.line, column=tok.column)

    def parse_init_block(self) -> InitBlock:
        """Parse init block."""
        tok = self.expect(TokenType.INIT)
        body = self.parse_block()
        return InitBlock(body=body, line=tok.line, column=tok.column)

    # ==========================================================================
    # Boundary
    # ==========================================================================

    def parse_boundary(self) -> Boundary:
        """Parse boundary declaration."""
        tok = self.expect(TokenType.BOUNDARY)
        name_tok = self.expect(TokenType.IDENTIFIER)
        body = self.parse_block()
        return Boundary(name=name_tok.value, body=body, line=tok.line, column=tok.column)

    # ==========================================================================
    # Types
    # ==========================================================================

    def parse_type_annotation(self) -> TypeAnnotation:
        """Parse type annotation."""
        tok = self.current()

        type_map = {
            TokenType.REGISTER: TypeKind.REGISTER,
            TokenType.TOKEN_REG: TypeKind.TOKEN_REG,
            TokenType.TYPE_FLOAT: TypeKind.FLOAT,
            TokenType.TYPE_INT: TypeKind.INT,
            TokenType.TYPE_BOOL: TypeKind.BOOL,
            TokenType.TYPE_STRING: TypeKind.STRING,
            TokenType.TYPE_BYTES: TypeKind.BYTES,
            TokenType.VOID: TypeKind.VOID,
        }

        if tok.type in type_map:
            self.advance()
            return TypeAnnotation(kind=type_map[tok.type], line=tok.line, column=tok.column)
        elif tok.type == TokenType.IDENTIFIER:
            self.advance()
            return TypeAnnotation(kind=TypeKind.CUSTOM, name=tok.value, line=tok.line, column=tok.column)
        else:
            raise self.error(f"Expected type, got {tok.type.name}")

    # ==========================================================================
    # Statements
    # ==========================================================================

    def parse_block(self) -> Block:
        """Parse a block of statements."""
        tok = self.expect(TokenType.LBRACE)
        statements = []
        while not self.check(TokenType.RBRACE, TokenType.EOF):
            statements.append(self.parse_statement())
        self.expect(TokenType.RBRACE)
        return Block(statements=statements, line=tok.line, column=tok.column)

    def parse_statement(self) -> Statement:
        """Parse a statement."""
        tok = self.current()

        # Variable declaration
        if self.check(TokenType.VAR):
            return self.parse_var_decl()

        # Witness binding: x ::= expr
        if self.check(TokenType.IDENTIFIER) and self.peek(1).type == TokenType.WITNESS_BIND:
            return self.parse_witness_decl()

        # Control flow
        if self.check(TokenType.IF_PAST):
            return self.parse_if_past()
        if self.check(TokenType.REPEAT_PAST):
            return self.parse_repeat_past()
        if self.check(TokenType.WHILE_PAST):
            return self.parse_while_past()
        if self.check(TokenType.RETURN):
            return self.parse_return()

        # Kernel call
        if self.check(TokenType.CALL):
            return self.parse_kernel_call()

        # Built-in statements
        if self.check(TokenType.INJECT_F):
            return self.parse_inject_f()
        if self.check(TokenType.REQUEST_GRACE):
            return self.parse_request_grace()

        # Expression statement (including assignments)
        return self.parse_expression_statement()

    def parse_witness_decl(self) -> WitnessDecl:
        """Parse witness binding: x ::= expr."""
        tok = self.current()
        name_tok = self.expect(TokenType.IDENTIFIER)
        self.expect(TokenType.WITNESS_BIND)
        expr = self.parse_expression()
        self.match(TokenType.SEMICOLON)
        return WitnessDecl(name=name_tok.value, expression=expr, line=tok.line, column=tok.column)

    def parse_if_past(self) -> IfPast:
        """Parse if_past statement."""
        tok = self.expect(TokenType.IF_PAST)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)
        then_block = self.parse_block()

        else_block = None
        if self.match(TokenType.ELSE_PAST):
            self.expect(TokenType.LPAREN)
            else_cond = self.parse_expression()
            self.expect(TokenType.RPAREN)
            else_then = self.parse_block()
            else_block = IfPast(condition=else_cond, then_block=else_then, line=tok.line, column=tok.column)
        elif self.match(TokenType.ELSE):
            else_block = self.parse_block()

        return IfPast(condition=condition, then_block=then_block, else_block=else_block,
                      line=tok.line, column=tok.column)

    def parse_repeat_past(self) -> RepeatPast:
        """Parse repeat_past loop."""
        tok = self.expect(TokenType.REPEAT_PAST)
        self.expect(TokenType.LPAREN)
        count = self.parse_expression()
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return RepeatPast(count=count, body=body, line=tok.line, column=tok.column)

    def parse_while_past(self) -> WhilePast:
        """Parse while_past loop."""
        tok = self.expect(TokenType.WHILE_PAST)
        self.expect(TokenType.LPAREN)
        condition = self.parse_expression()
        self.expect(TokenType.RPAREN)
        body = self.parse_block()
        return WhilePast(condition=condition, body=body, line=tok.line, column=tok.column)

    def parse_return(self) -> Return:
        """Parse return statement."""
        tok = self.expect(TokenType.RETURN)
        value = None
        if not self.check(TokenType.SEMICOLON, TokenType.RBRACE):
            value = self.parse_expression()
        self.match(TokenType.SEMICOLON)
        return Return(value=value, line=tok.line, column=tok.column)

    def parse_kernel_call(self) -> KernelCall:
        """Parse kernel call: call KernelName(port: expr, ...)."""
        tok = self.expect(TokenType.CALL)
        kernel_name = self.expect(TokenType.IDENTIFIER).value
        self.expect(TokenType.LPAREN)

        bindings = {}
        while not self.check(TokenType.RPAREN, TokenType.EOF):
            port_name = self.expect(TokenType.IDENTIFIER).value
            self.expect(TokenType.COLON)
            value = self.parse_expression()
            bindings[port_name] = value
            if not self.match(TokenType.COMMA):
                break

        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)
        return KernelCall(kernel_name=kernel_name, port_bindings=bindings, line=tok.line, column=tok.column)

    def parse_inject_f(self) -> InjectF:
        """Parse inject_F statement."""
        tok = self.expect(TokenType.INJECT_F)
        self.expect(TokenType.LPAREN)
        target = self.parse_expression()
        self.expect(TokenType.COMMA)
        amount = self.parse_expression()
        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)
        return InjectF(target=target, amount=amount, line=tok.line, column=tok.column)

    def parse_request_grace(self) -> RequestGrace:
        """Parse request_grace statement."""
        tok = self.expect(TokenType.REQUEST_GRACE)
        self.expect(TokenType.LPAREN)
        amount = self.parse_expression()
        self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMICOLON)
        return RequestGrace(amount=amount, line=tok.line, column=tok.column)

    def parse_expression_statement(self) -> Statement:
        """Parse expression statement (may be assignment)."""
        tok = self.current()
        expr = self.parse_expression()

        # Check for assignment
        if self.check(TokenType.RECONCILE_EQ, TokenType.PLUS_EQ, TokenType.MINUS_EQ,
                      TokenType.STAR_EQ, TokenType.SLASH_EQ):
            op = self.advance().value
            value = self.parse_expression()
            self.match(TokenType.SEMICOLON)
            return Assignment(target=expr, operator=op, value=value, line=tok.line, column=tok.column)

        self.match(TokenType.SEMICOLON)
        return ExpressionStmt(expression=expr, line=tok.line, column=tok.column)

    # ==========================================================================
    # Expressions (Precedence Climbing)
    # ==========================================================================

    def parse_expression(self) -> Expression:
        """Parse expression."""
        return self.parse_or()

    def parse_or(self) -> Expression:
        """Parse || expression."""
        left = self.parse_and()
        while self.match(TokenType.OR):
            right = self.parse_and()
            left = BinaryOp(left=left, operator="||", right=right)
        return left

    def parse_and(self) -> Expression:
        """Parse && expression."""
        left = self.parse_equality()
        while self.match(TokenType.AND):
            right = self.parse_equality()
            left = BinaryOp(left=left, operator="&&", right=right)
        return left

    def parse_equality(self) -> Expression:
        """Parse == != expression."""
        left = self.parse_comparison()
        while self.check(TokenType.MEASURE_EQ, TokenType.NE, TokenType.COVENANT_EQ):
            op = self.advance().value
            right = self.parse_comparison()
            left = BinaryOp(left=left, operator=op, right=right)
        return left

    def parse_comparison(self) -> Expression:
        """Parse < > <= >= expression."""
        left = self.parse_additive()
        while self.check(TokenType.LT, TokenType.GT, TokenType.LE, TokenType.GE):
            op = self.advance().value
            right = self.parse_additive()
            left = BinaryOp(left=left, operator=op, right=right)
        return left

    def parse_additive(self) -> Expression:
        """Parse + - expression."""
        left = self.parse_multiplicative()
        while self.check(TokenType.PLUS, TokenType.MINUS):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = BinaryOp(left=left, operator=op, right=right)
        return left

    def parse_multiplicative(self) -> Expression:
        """Parse * / % expression."""
        left = self.parse_unary()
        while self.check(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.advance().value
            right = self.parse_unary()
            left = BinaryOp(left=left, operator=op, right=right)
        return left

    def parse_unary(self) -> Expression:
        """Parse unary expression."""
        if self.check(TokenType.NOT, TokenType.MINUS, TokenType.TILDE):
            op = self.advance().value
            operand = self.parse_unary()
            return UnaryOp(operator=op, operand=operand)
        return self.parse_postfix()

    def parse_postfix(self) -> Expression:
        """Parse postfix expressions (calls, field access, index)."""
        expr = self.parse_primary()

        while True:
            if self.match(TokenType.DOT):
                field = self.expect(TokenType.IDENTIFIER).value
                expr = FieldAccess(object=expr, field=field)
            elif self.match(TokenType.LPAREN):
                args = self.parse_arguments()
                self.expect(TokenType.RPAREN)
                expr = Call(callee=expr, arguments=args)
            elif self.match(TokenType.LBRACKET):
                index = self.parse_expression()
                self.expect(TokenType.RBRACKET)
                expr = IndexAccess(object=expr, index=index)
            else:
                break

        return expr

    def parse_arguments(self) -> list[Expression]:
        """Parse function arguments."""
        args = []
        if not self.check(TokenType.RPAREN):
            args.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                args.append(self.parse_expression())
        return args

    def parse_primary(self) -> Expression:
        """Parse primary expression."""
        tok = self.current()

        # Literals
        if self.match(TokenType.INTEGER):
            return Literal(value=int(tok.value), type_hint=TypeKind.INT, line=tok.line, column=tok.column)
        if self.match(TokenType.FLOAT):
            return Literal(value=float(tok.value), type_hint=TypeKind.FLOAT, line=tok.line, column=tok.column)
        if self.match(TokenType.STRING):
            return Literal(value=tok.value, type_hint=TypeKind.STRING, line=tok.line, column=tok.column)
        if self.match(TokenType.TRUE):
            return Literal(value=True, type_hint=TypeKind.BOOL, line=tok.line, column=tok.column)
        if self.match(TokenType.FALSE):
            return Literal(value=False, type_hint=TypeKind.BOOL, line=tok.line, column=tok.column)

        # This
        if self.match(TokenType.THIS):
            return This(line=tok.line, column=tok.column)

        # Distinct
        if self.match(TokenType.DISTINCT):
            self.expect(TokenType.LPAREN)
            self.expect(TokenType.RPAREN)
            return Distinct(line=tok.line, column=tok.column)

        # Compare
        if self.match(TokenType.CMP):
            self.expect(TokenType.LPAREN)
            left = self.parse_expression()
            self.expect(TokenType.COMMA)
            right = self.parse_expression()
            epsilon = None
            if self.match(TokenType.COMMA):
                epsilon = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return Compare(left=left, right=right, epsilon=epsilon, line=tok.line, column=tok.column)

        # Primitive functions (transfer, diffuse, choose) - treat as identifiers when used as function calls
        if self.check(TokenType.TRANSFER, TokenType.DIFFUSE, TokenType.CHOOSE):
            func_tok = self.advance()
            return Identifier(name=func_tok.value, line=func_tok.line, column=func_tok.column)

        # Soma read
        if self.match(TokenType.SOMA_READ):
            self.expect(TokenType.LPAREN)
            sensor = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return Call(callee=Identifier(name="soma_read", line=tok.line, column=tok.column),
                       arguments=[sensor], line=tok.line, column=tok.column)

        # Soma write
        if self.match(TokenType.SOMA_WRITE):
            self.expect(TokenType.LPAREN)
            actuator = self.parse_expression()
            self.expect(TokenType.COMMA)
            value = self.parse_expression()
            self.expect(TokenType.RPAREN)
            return Call(callee=Identifier(name="soma_write", line=tok.line, column=tok.column),
                       arguments=[actuator, value], line=tok.line, column=tok.column)

        # Tuple or grouped expression
        if self.match(TokenType.LPAREN):
            expr = self.parse_expression()
            if self.match(TokenType.COMMA):
                # Tuple
                elements = [expr]
                elements.append(self.parse_expression())
                while self.match(TokenType.COMMA):
                    elements.append(self.parse_expression())
                self.expect(TokenType.RPAREN)
                return TupleExpr(elements=elements, line=tok.line, column=tok.column)
            self.expect(TokenType.RPAREN)
            return expr

        # Identifier
        if self.match(TokenType.IDENTIFIER):
            return Identifier(name=tok.value, line=tok.line, column=tok.column)

        raise self.error(f"Unexpected token in expression: {tok.type.name}")


def parse(source: str, filename: str = "<input>") -> Program:
    """Convenience function to parse source code."""
    parser = Parser(source, filename)
    return parser.parse()
