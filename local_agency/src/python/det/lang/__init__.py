"""
Existence-Lang - Agency-First Programming Language
==================================================

A programming language for DET-OS where agency is the primitive
and traditional computing concepts (logic, arithmetic, equality)
emerge from agency acting over time.

Core Philosophy:
    Agency creates distinction.
    Distinction creates movement.
    Movement leaves trace.
    Trace becomes math.

Four Equalities:
    := (alias)     - Compile-time binding, no cost
    == (measure)   - Trace equality, produces past token
    =  (reconcile) - Attempted unification, may fail
    ≡  (covenant)  - Bond truth, checks coherence

Usage:
    from det.lang import parse, transpile, ExistenceRuntime

    # Parse Existence-Lang source
    ast = parse(source_code)

    # Transpile to Python
    python_code = transpile(source_code)

    # Execute with runtime
    runtime = ExistenceRuntime()
    runtime.execute_source(source_code)
"""

# Core modules
from .errors import (
    ExistenceError, LexerError, ParseError, SemanticError, RuntimeError,
    TypeError, UndefinedError, AgencyError, ResourceError,
    WitnessToken, ErrorReporter, SourceLocation
)

from .tokens import Token, TokenType, Lexer

from .ast_nodes import (
    # Base
    ASTNode, Expression, Statement, Declaration,
    TypeAnnotation, TypeKind,
    # Expressions
    Literal, Identifier, BinaryOp, UnaryOp, Call, FieldAccess, IndexAccess,
    Distinct, WitnessToken as WitnessTokenExpr, This, Compare, Choose, TupleExpr,
    # Statements
    Block, VarDecl, WitnessDecl, Assignment, ExpressionStmt,
    IfPast, RepeatPast, WhilePast, Return, KernelCall,
    InjectF, RequestGrace,
    # Declarations
    Creature, Kernel, Bond, Presence, Boundary, Program,
    # Kernel components
    PortDecl, PortDirection, ParamDecl, Proposal, PhaseBlock,
    # Creature components
    SensorDecl, ActuatorDecl, ParticipateBlock, AgencyBlock, GraceBlock,
    # Visitor
    ASTVisitor
)

from .parser import Parser, parse

from .semantic import SemanticAnalyzer, analyze, Scope

from .transpiler import Transpiler, transpile

from .eis_compiler import (
    EISCompiler, compile_to_eis, compile_source as compile_to_eis_source,
    CompiledProgram, CompiledCreature, CompiledKernel, RegAlloc
)

from .runtime import (
    # Core classes
    ExistenceRuntime, CreatureBase, KernelBase, BondInstance,
    Register, TokenReg, SensorBinding, ActuatorBinding,
    # Result types
    CompareResult, ReconcileResult, TransferResult, WriteResult,
    # Primitive operations
    distinct, transfer, diffuse, compare, reconcile, choose, local_seed
)

# Standard library
from .stdlib import (
    Transfer, Diffuse, Distinct as DistinctKernel, Compare as CompareKernel,
    AddSigned, SubSigned, MulByPastToken, Reconcile,
    GraceOffer, GraceAccept, GraceFlow
)

__version__ = "0.1.0"

__all__ = [
    # Errors
    "ExistenceError", "LexerError", "ParseError", "SemanticError", "RuntimeError",
    "TypeError", "UndefinedError", "AgencyError", "ResourceError",
    "WitnessToken", "ErrorReporter", "SourceLocation",

    # Tokens
    "Token", "TokenType", "Lexer",

    # AST
    "ASTNode", "Expression", "Statement", "Declaration",
    "TypeAnnotation", "TypeKind",
    "Literal", "Identifier", "BinaryOp", "UnaryOp", "Call", "FieldAccess",
    "Block", "VarDecl", "WitnessDecl", "Assignment", "IfPast", "RepeatPast",
    "Creature", "Kernel", "Bond", "Presence", "Program",
    "ASTVisitor",

    # Parser
    "Parser", "parse",

    # Semantic
    "SemanticAnalyzer", "analyze", "Scope",

    # Transpiler
    "Transpiler", "transpile",

    # EIS Compiler
    "EISCompiler", "compile_to_eis", "compile_to_eis_source",
    "CompiledProgram", "CompiledCreature", "CompiledKernel", "RegAlloc",

    # Runtime
    "ExistenceRuntime", "CreatureBase", "KernelBase", "BondInstance",
    "Register", "TokenReg", "SensorBinding", "ActuatorBinding",
    "CompareResult", "ReconcileResult", "TransferResult", "WriteResult",
    "distinct", "transfer", "diffuse", "compare", "reconcile", "choose",

    # Standard Library Kernels
    "Transfer", "Diffuse", "DistinctKernel", "CompareKernel",
    "AddSigned", "SubSigned", "MulByPastToken", "Reconcile",
    "GraceOffer", "GraceAccept", "GraceFlow",
]
