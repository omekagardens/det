"""
Existence-Lang Error Types and Reporting
========================================

Error types for lexing, parsing, and semantic analysis.
Errors are consequences in DET terms - they produce witness tokens.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto


class ErrorSeverity(Enum):
    """Error severity levels."""
    WARNING = auto()
    ERROR = auto()
    FATAL = auto()


@dataclass
class SourceLocation:
    """Location in source code."""
    line: int
    column: int
    filename: str = "<input>"

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}:{self.column}"


@dataclass
class ExistenceError(Exception):
    """Base error type for Existence-Lang."""
    message: str
    location: Optional[SourceLocation] = None
    severity: ErrorSeverity = ErrorSeverity.ERROR

    def __str__(self) -> str:
        loc = f" at {self.location}" if self.location else ""
        return f"[{self.severity.name}]{loc}: {self.message}"


class LexerError(ExistenceError):
    """Error during tokenization."""
    pass


class ParseError(ExistenceError):
    """Error during parsing."""
    pass


class SemanticError(ExistenceError):
    """Error during semantic analysis."""
    pass


class RuntimeError(ExistenceError):
    """Error during execution."""
    pass


class TypeError(SemanticError):
    """Type mismatch error."""
    pass


class UndefinedError(SemanticError):
    """Undefined identifier error."""
    pass


class AgencyError(RuntimeError):
    """Insufficient agency for operation."""
    pass


class ResourceError(RuntimeError):
    """Insufficient resource for operation."""
    pass


class WitnessToken(Enum):
    """Witness tokens for operation outcomes (consequences, not errors)."""
    # Reconciliation outcomes
    EQ_OK = auto()
    EQ_FAIL = auto()
    EQ_REFUSE = auto()

    # Comparison outcomes
    LT = auto()
    EQ = auto()
    GT = auto()
    EQ_TRUE = auto()
    EQ_FALSE = auto()
    EQ_APPROX = auto()

    # Transfer outcomes
    TRANSFER_OK = auto()
    TRANSFER_PARTIAL = auto()
    TRANSFER_BLOCKED = auto()

    # Bond outcomes
    BOND_OK = auto()
    BOND_REFUSED = auto()
    BOND_EXISTS = auto()

    # Covenant outcomes
    COV_ALIGNED = auto()
    COV_DRIFT = auto()
    COV_BROKEN = auto()

    # Write outcomes
    WRITE_OK = auto()
    WRITE_REFUSED = auto()
    WRITE_PARTIAL = auto()

    # Subtraction outcomes
    SUB_OK = auto()
    SUB_FAIL = auto()
    SUB_REFUSE = auto()

    # Proposal outcomes
    ACCEPTED = auto()
    REJECTED = auto()
    MODIFIED = auto()

    # General
    OK = auto()
    FAIL = auto()


class ErrorReporter:
    """Collects and reports errors."""

    def __init__(self, filename: str = "<input>"):
        self.filename = filename
        self.errors: list[ExistenceError] = []
        self.warnings: list[ExistenceError] = []

    def error(self, message: str, line: int = 0, column: int = 0) -> ExistenceError:
        """Report an error."""
        loc = SourceLocation(line, column, self.filename)
        err = ExistenceError(message, loc, ErrorSeverity.ERROR)
        self.errors.append(err)
        return err

    def warning(self, message: str, line: int = 0, column: int = 0) -> ExistenceError:
        """Report a warning."""
        loc = SourceLocation(line, column, self.filename)
        warn = ExistenceError(message, loc, ErrorSeverity.WARNING)
        self.warnings.append(warn)
        return warn

    def lexer_error(self, message: str, line: int, column: int) -> LexerError:
        """Report a lexer error."""
        loc = SourceLocation(line, column, self.filename)
        err = LexerError(message, loc)
        self.errors.append(err)
        return err

    def parse_error(self, message: str, line: int, column: int) -> ParseError:
        """Report a parse error."""
        loc = SourceLocation(line, column, self.filename)
        err = ParseError(message, loc)
        self.errors.append(err)
        return err

    def semantic_error(self, message: str, line: int = 0, column: int = 0) -> SemanticError:
        """Report a semantic error."""
        loc = SourceLocation(line, column, self.filename)
        err = SemanticError(message, loc)
        self.errors.append(err)
        return err

    def has_errors(self) -> bool:
        """Check if any errors were reported."""
        return len(self.errors) > 0

    def report(self) -> str:
        """Generate error report."""
        lines = []
        for warn in self.warnings:
            lines.append(str(warn))
        for err in self.errors:
            lines.append(str(err))
        return "\n".join(lines)

    def clear(self):
        """Clear all errors and warnings."""
        self.errors.clear()
        self.warnings.clear()
