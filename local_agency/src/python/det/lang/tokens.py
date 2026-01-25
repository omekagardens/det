"""
Existence-Lang Lexer and Token Types
====================================

Tokenization for the Existence-Lang agency-first programming language.
Handles the four equalities: := (alias), == (measure), = (reconcile), ≡ (covenant)
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Iterator
import re

from .errors import LexerError, ErrorReporter, SourceLocation


class TokenType(Enum):
    """Token types for Existence-Lang (~60 tokens)."""

    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    IDENTIFIER = auto()

    # Keywords - Top level
    CREATURE = auto()       # creature
    KERNEL = auto()         # kernel
    BOND = auto()           # bond
    PRESENCE = auto()       # presence
    BOUNDARY = auto()       # boundary

    # Keywords - Blocks
    VAR = auto()            # var
    SENSOR = auto()         # sensor
    ACTUATOR = auto()       # actuator
    PARTICIPATE = auto()    # participate
    AGENCY = auto()         # agency
    GRACE = auto()          # grace
    PHASE = auto()          # phase
    PARAMS = auto()         # params
    CREATURES = auto()      # creatures
    BONDS = auto()          # bonds
    INIT = auto()           # init
    LAW = auto()            # law
    PIPE = auto()           # pipe

    # Keywords - Kernel
    IN = auto()             # in
    OUT = auto()            # out
    INOUT = auto()          # inout
    PROPOSAL = auto()       # proposal
    SCORE = auto()          # score
    EFFECT = auto()         # effect
    CHOICE = auto()         # choice
    COMMIT = auto()         # commit
    CALL = auto()           # call

    # Keywords - Types
    REGISTER = auto()       # Register
    TOKEN_REG = auto()      # TokenReg
    TYPE_FLOAT = auto()     # float
    TYPE_INT = auto()       # int
    TYPE_BOOL = auto()      # bool
    TYPE_STRING = auto()    # string
    TYPE_BYTES = auto()     # bytes

    # Keywords - Control flow
    IF_PAST = auto()        # if_past
    THEN = auto()           # then (for conditional expressions)
    ELSE_PAST = auto()      # else_past
    ELSE = auto()           # else
    REPEAT_PAST = auto()    # repeat_past
    WHILE_PAST = auto()     # while_past
    AS = auto()             # as (for loop variable binding)
    FORECAST = auto()       # forecast
    SIMULATE = auto()       # simulate
    PROPOSE = auto()        # propose
    RETURN = auto()         # return
    MATCH = auto()          # match

    # Keywords - Primitives
    DISTINCT = auto()       # distinct
    TRANSFER = auto()       # transfer
    DIFFUSE = auto()        # diffuse
    CMP = auto()            # cmp
    CHOOSE = auto()         # choose

    # Keywords - Built-ins
    THIS = auto()           # this
    TRUE = auto()           # true
    FALSE = auto()          # false
    VOID = auto()           # void
    PRIMITIVE = auto()      # primitive (external I/O call)

    # Keywords - Somatic
    SOMA_READ = auto()      # soma_read
    SOMA_WRITE = auto()     # soma_write
    CHANNEL = auto()        # channel

    # Keywords - Grace
    INJECT_F = auto()       # inject_F
    REQUEST_GRACE = auto()  # request_grace

    # Keywords - Constants
    CONST = auto()          # const

    # Keywords - Module
    IMPORT = auto()         # import

    # Four equalities
    ALIAS_EQ = auto()       # := (alias equality)
    MEASURE_EQ = auto()     # == (trace equality/measurement)
    RECONCILE_EQ = auto()   # = (reconciliation)
    COVENANT_EQ = auto()    # ≡ or === (covenant equality)

    # Witness binding
    WITNESS_BIND = auto()   # ::= (witness token binding)

    # Operators
    PLUS = auto()           # +
    MINUS = auto()          # -
    STAR = auto()           # *
    SLASH = auto()          # /
    PERCENT = auto()        # %

    PLUS_EQ = auto()        # +=
    MINUS_EQ = auto()       # -=
    STAR_EQ = auto()        # *=
    SLASH_EQ = auto()       # /=

    # Comparison (past-only)
    LT = auto()             # <
    GT = auto()             # >
    LE = auto()             # <=
    GE = auto()             # >=
    NE = auto()             # !=

    # Logical
    AND = auto()            # &&
    OR = auto()             # ||
    NOT = auto()            # !

    # Bitwise
    AMP = auto()            # &
    PIPE_OP = auto()        # |
    CARET = auto()          # ^
    TILDE = auto()          # ~

    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]

    COMMA = auto()          # ,
    DOT = auto()            # .
    COLON = auto()          # :
    SEMICOLON = auto()      # ;
    AT = auto()             # @
    ARROW = auto()          # ->
    FAT_ARROW = auto()      # =>
    TILDE_ARROW = auto()    # <~
    DOUBLE_COLON = auto()   # ::
    BOND_ARROW = auto()     # <->

    # Special
    NEWLINE = auto()
    EOF = auto()
    COMMENT = auto()


@dataclass
class Token:
    """A lexical token."""
    type: TokenType
    value: str
    line: int
    column: int

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value!r}, {self.line}:{self.column})"


# Keyword mapping
KEYWORDS = {
    # Top level
    "creature": TokenType.CREATURE,
    "kernel": TokenType.KERNEL,
    "bond": TokenType.BOND,
    "presence": TokenType.PRESENCE,
    "boundary": TokenType.BOUNDARY,

    # Blocks
    "var": TokenType.VAR,
    "sensor": TokenType.SENSOR,
    "actuator": TokenType.ACTUATOR,
    "participate": TokenType.PARTICIPATE,
    "agency": TokenType.AGENCY,
    "grace": TokenType.GRACE,
    "phase": TokenType.PHASE,
    "params": TokenType.PARAMS,
    "parameters": TokenType.PARAMS,
    "creatures": TokenType.CREATURES,
    "bonds": TokenType.BONDS,
    "init": TokenType.INIT,
    "law": TokenType.LAW,
    "pipe": TokenType.PIPE,

    # Kernel
    "in": TokenType.IN,
    "out": TokenType.OUT,
    "inout": TokenType.INOUT,
    "proposal": TokenType.PROPOSAL,
    "score": TokenType.SCORE,
    "effect": TokenType.EFFECT,
    "choice": TokenType.CHOICE,
    "commit": TokenType.COMMIT,
    "call": TokenType.CALL,

    # Types
    "Register": TokenType.REGISTER,
    "TokenReg": TokenType.TOKEN_REG,
    "float": TokenType.TYPE_FLOAT,
    "int": TokenType.TYPE_INT,
    "bool": TokenType.TYPE_BOOL,
    "string": TokenType.TYPE_STRING,
    "bytes": TokenType.TYPE_BYTES,

    # Control flow
    "if_past": TokenType.IF_PAST,
    "then": TokenType.THEN,
    "else_past": TokenType.ELSE_PAST,
    "else": TokenType.ELSE,
    "repeat_past": TokenType.REPEAT_PAST,
    "while_past": TokenType.WHILE_PAST,
    "as": TokenType.AS,
    "forecast": TokenType.FORECAST,
    "simulate": TokenType.SIMULATE,
    "propose": TokenType.PROPOSE,
    "return": TokenType.RETURN,
    "match": TokenType.MATCH,

    # Primitives
    "distinct": TokenType.DISTINCT,
    "transfer": TokenType.TRANSFER,
    "diffuse": TokenType.DIFFUSE,
    "cmp": TokenType.CMP,
    "compare": TokenType.CMP,
    "choose": TokenType.CHOOSE,

    # Built-ins
    "this": TokenType.THIS,
    "true": TokenType.TRUE,
    "false": TokenType.FALSE,
    "void": TokenType.VOID,
    "primitive": TokenType.PRIMITIVE,

    # Somatic
    "soma_read": TokenType.SOMA_READ,
    "soma_write": TokenType.SOMA_WRITE,
    "channel": TokenType.CHANNEL,

    # Grace
    "inject_F": TokenType.INJECT_F,
    "request_grace": TokenType.REQUEST_GRACE,

    # Constants
    "const": TokenType.CONST,

    # Module
    "import": TokenType.IMPORT,
}


class Lexer:
    """Tokenizer for Existence-Lang."""

    __slots__ = ('source', 'source_len', 'filename', 'pos', 'line', 'column', 'reporter')

    def __init__(self, source: str, filename: str = "<input>"):
        self.source = source
        self.source_len = len(source)  # Cache length - never changes
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.column = 1
        self.reporter = ErrorReporter(filename)

    def peek(self, offset: int = 0) -> str:
        """Peek at character at current position + offset."""
        idx = self.pos + offset
        if idx >= self.source_len:
            return '\0'
        return self.source[idx]

    def advance(self) -> str:
        """Advance and return current character."""
        if self.pos >= self.source_len:
            return '\0'
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def skip_whitespace(self):
        """Skip whitespace (not newlines for now)."""
        # Inline peek/advance for hot path - avoid method call overhead
        source = self.source
        source_len = self.source_len
        pos = self.pos
        while pos < source_len:
            ch = source[pos]
            if ch == ' ' or ch == '\t' or ch == '\r':
                pos += 1
                self.column += 1
            else:
                break
        self.pos = pos

    def skip_comment(self) -> Optional[Token]:
        """Skip single-line or multi-line comment."""
        start_line = self.line
        start_col = self.column

        if self.peek() == '/' and self.peek(1) == '/':
            # Single-line comment
            self.advance()  # /
            self.advance()  # /
            while self.peek() != '\n' and self.peek() != '\0':
                self.advance()
            return None

        if self.peek() == '/' and self.peek(1) == '*':
            # Multi-line comment
            self.advance()  # /
            self.advance()  # *
            while True:
                if self.peek() == '\0':
                    raise self.reporter.lexer_error(
                        "Unterminated multi-line comment",
                        start_line, start_col
                    )
                if self.peek() == '*' and self.peek(1) == '/':
                    self.advance()  # *
                    self.advance()  # /
                    break
                self.advance()
            return None

        return None

    def read_string(self) -> Token:
        """Read a string literal."""
        start_line = self.line
        start_col = self.column
        quote = self.advance()  # " or '

        chars = []
        while True:
            ch = self.peek()
            if ch == '\0' or ch == '\n':
                raise self.reporter.lexer_error(
                    "Unterminated string literal",
                    start_line, start_col
                )
            if ch == quote:
                self.advance()
                break
            if ch == '\\':
                self.advance()
                escape = self.advance()
                escape_map = {'n': '\n', 't': '\t', 'r': '\r', '\\': '\\', '"': '"', "'": "'"}
                chars.append(escape_map.get(escape, escape))
            else:
                chars.append(self.advance())

        return Token(TokenType.STRING, ''.join(chars), start_line, start_col)

    def read_number(self) -> Token:
        """Read a number literal (integer or float)."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        source = self.source
        source_len = self.source_len
        pos = self.pos
        has_dot = False
        has_exp = False

        # Handle negative sign
        if pos < source_len and source[pos] == '-':
            pos += 1

        # Check for hex prefix 0x
        if pos < source_len and source[pos] == '0' and pos + 1 < source_len and source[pos + 1] in 'xX':
            pos += 2  # Skip 0x
            while pos < source_len:
                ch = source[pos]
                if ('0' <= ch <= '9') or ('a' <= ch <= 'f') or ('A' <= ch <= 'F'):
                    pos += 1
                elif ch == '_':
                    pos += 1  # Skip underscores
                else:
                    break
            # Build value without underscores
            value = source[start_pos:pos].replace('_', '')
            self.pos = pos
            self.column = start_col + (pos - start_pos)
            return Token(TokenType.INTEGER, value, start_line, start_col)

        # Regular decimal number
        while pos < source_len:
            ch = source[pos]
            if '0' <= ch <= '9':
                pos += 1
            elif ch == '.' and not has_dot and not has_exp:
                if pos + 1 < source_len and '0' <= source[pos + 1] <= '9':
                    has_dot = True
                    pos += 1
                else:
                    break
            elif ch in 'eE' and not has_exp:
                has_exp = True
                pos += 1
                if pos < source_len and source[pos] in '+-':
                    pos += 1
            elif ch == '_':
                pos += 1  # Skip underscores
            else:
                break

        # Build value without underscores
        value = source[start_pos:pos].replace('_', '')
        self.pos = pos
        self.column = start_col + (pos - start_pos)

        if has_dot or has_exp:
            return Token(TokenType.FLOAT, value, start_line, start_col)
        return Token(TokenType.INTEGER, value, start_line, start_col)

    def read_identifier(self) -> Token:
        """Read an identifier or keyword."""
        start_line = self.line
        start_col = self.column
        start_pos = self.pos

        # Fast scan using direct string access
        source = self.source
        source_len = self.source_len
        pos = self.pos

        while pos < source_len:
            ch = source[pos]
            # Check if alphanumeric or underscore (common chars first)
            if ch == '_' or ('a' <= ch <= 'z') or ('A' <= ch <= 'Z') or ('0' <= ch <= '9'):
                pos += 1
            else:
                break

        # Extract identifier directly from source slice
        value = source[start_pos:pos]
        length = pos - start_pos

        # Update position tracking
        self.pos = pos
        self.column = start_col + length

        token_type = KEYWORDS.get(value, TokenType.IDENTIFIER)
        return Token(token_type, value, start_line, start_col)

    def next_token(self) -> Token:
        """Get the next token."""
        self.skip_whitespace()

        # Check for comments
        if self.peek() == '/' and self.peek(1) in '/*':
            self.skip_comment()
            return self.next_token()

        start_line = self.line
        start_col = self.column
        ch = self.peek()

        # End of file
        if ch == '\0':
            return Token(TokenType.EOF, '', start_line, start_col)

        # Newline
        if ch == '\n':
            self.advance()
            return Token(TokenType.NEWLINE, '\n', start_line, start_col)

        # String literals
        if ch in '"\'':
            return self.read_string()

        # Numbers
        if ch.isdigit() or (ch == '-' and self.peek(1).isdigit()):
            return self.read_number()

        # Identifiers and keywords
        if ch.isalpha() or ch == '_':
            return self.read_identifier()

        # Multi-character operators
        two_char = ch + self.peek(1)
        three_char = two_char + self.peek(2)

        # Three-character operators
        if three_char == '::=':
            self.advance(); self.advance(); self.advance()
            return Token(TokenType.WITNESS_BIND, '::=', start_line, start_col)
        if three_char == '===':
            self.advance(); self.advance(); self.advance()
            return Token(TokenType.COVENANT_EQ, '===', start_line, start_col)
        if three_char == '<->':
            self.advance(); self.advance(); self.advance()
            return Token(TokenType.BOND_ARROW, '<->', start_line, start_col)

        # Two-character operators
        two_char_ops = {
            ':=': TokenType.ALIAS_EQ,
            '::': TokenType.DOUBLE_COLON,
            '==': TokenType.MEASURE_EQ,
            '!=': TokenType.NE,
            '<=': TokenType.LE,
            '>=': TokenType.GE,
            '->': TokenType.ARROW,
            '=>': TokenType.FAT_ARROW,
            '<~': TokenType.TILDE_ARROW,
            '&&': TokenType.AND,
            '||': TokenType.OR,
            '+=': TokenType.PLUS_EQ,
            '-=': TokenType.MINUS_EQ,
            '*=': TokenType.STAR_EQ,
            '/=': TokenType.SLASH_EQ,
        }

        if two_char in two_char_ops:
            self.advance(); self.advance()
            return Token(two_char_ops[two_char], two_char, start_line, start_col)

        # Single-character operators
        single_char_ops = {
            '=': TokenType.RECONCILE_EQ,
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.STAR,
            '/': TokenType.SLASH,
            '%': TokenType.PERCENT,
            '<': TokenType.LT,
            '>': TokenType.GT,
            '!': TokenType.NOT,
            '&': TokenType.AMP,
            '|': TokenType.PIPE_OP,
            '^': TokenType.CARET,
            '~': TokenType.TILDE,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            ',': TokenType.COMMA,
            '.': TokenType.DOT,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
            '@': TokenType.AT,
        }

        if ch in single_char_ops:
            self.advance()
            return Token(single_char_ops[ch], ch, start_line, start_col)

        # Unicode covenant equality
        if ch == '≡':
            self.advance()
            return Token(TokenType.COVENANT_EQ, '≡', start_line, start_col)

        # Unknown character
        self.advance()
        raise self.reporter.lexer_error(
            f"Unexpected character: {ch!r}",
            start_line, start_col
        )

    def tokenize(self) -> list[Token]:
        """Tokenize the entire source, returning list of tokens."""
        tokens = []
        while True:
            tok = self.next_token()
            # Skip newlines for now (could be significant in some contexts)
            if tok.type == TokenType.NEWLINE:
                continue
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        return tokens

    def __iter__(self) -> Iterator[Token]:
        """Iterate over tokens."""
        while True:
            tok = self.next_token()
            if tok.type == TokenType.NEWLINE:
                continue
            yield tok
            if tok.type == TokenType.EOF:
                break
