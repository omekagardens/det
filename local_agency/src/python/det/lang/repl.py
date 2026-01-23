"""
Existence-Lang Interactive REPL
===============================

Read-Eval-Print Loop for Existence-Lang exploration.
"""

import sys
import readline
from typing import Optional
from .parser import parse, Parser
from .transpiler import transpile, Transpiler
from .runtime import ExistenceRuntime, Register, TokenReg
from .semantic import analyze


class REPL:
    """Interactive REPL for Existence-Lang."""

    PROMPT = "existence> "
    CONT_PROMPT = "       ... "

    def __init__(self, det_core: Optional[any] = None):
        self.runtime = ExistenceRuntime(det_core)
        self.det_core = det_core
        self.history: list[str] = []
        self.multiline_buffer: list[str] = []
        self.in_multiline = False

    def print_welcome(self):
        """Print welcome message."""
        print("=" * 60)
        print("Existence-Lang REPL v0.1")
        print("Agency-First Programming Language for DET-OS")
        print("=" * 60)
        print()
        print("Commands:")
        print("  :help     - Show this help")
        print("  :state    - Show DET core state")
        print("  :creatures - List active creatures")
        print("  :kernels  - List available kernels")
        print("  :step [N] - Step DET simulation N times")
        print("  :load FILE - Load Existence-Lang file")
        print("  :ast CODE - Show AST for code")
        print("  :py CODE  - Show transpiled Python")
        print("  :clear    - Clear runtime state")
        print("  quit/exit - Exit REPL")
        print()

    def run(self):
        """Run the REPL."""
        self.print_welcome()

        while True:
            try:
                prompt = self.CONT_PROMPT if self.in_multiline else self.PROMPT
                line = input(prompt)

                # Handle commands
                if line.startswith(":"):
                    self.handle_command(line)
                    continue

                # Handle exit
                if line.strip().lower() in ("quit", "exit"):
                    print("Goodbye!")
                    break

                # Handle multiline input
                if self.in_multiline:
                    self.multiline_buffer.append(line)
                    if self.check_complete("\n".join(self.multiline_buffer)):
                        source = "\n".join(self.multiline_buffer)
                        self.multiline_buffer = []
                        self.in_multiline = False
                        self.execute(source)
                else:
                    if self.needs_continuation(line):
                        self.multiline_buffer = [line]
                        self.in_multiline = True
                    else:
                        self.execute(line)

            except EOFError:
                print("\nGoodbye!")
                break
            except KeyboardInterrupt:
                print("\n(Interrupted)")
                self.multiline_buffer = []
                self.in_multiline = False
            except Exception as e:
                print(f"Error: {e}")

    def handle_command(self, line: str):
        """Handle REPL command."""
        parts = line[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "help":
            self.print_welcome()

        elif cmd == "state":
            self.show_state()

        elif cmd == "creatures":
            self.show_creatures()

        elif cmd == "kernels":
            self.show_kernels()

        elif cmd == "step":
            n = int(arg) if arg else 1
            for _ in range(n):
                self.runtime.step()
            print(f"Stepped {n} time(s). Tick: {self.runtime._tick}")

        elif cmd == "load":
            self.load_file(arg)

        elif cmd == "ast":
            self.show_ast(arg)

        elif cmd == "py":
            self.show_python(arg)

        elif cmd == "clear":
            self.runtime = ExistenceRuntime(self.det_core)
            print("Runtime cleared.")

        else:
            print(f"Unknown command: {cmd}")

    def show_state(self):
        """Show current runtime state."""
        state = self.runtime.get_state()
        print(f"Tick: {state['tick']}")
        print(f"Creatures: {len(state['creatures'])}")
        print(f"Bonds: {len(state['bonds'])}")

        if self.det_core:
            try:
                presence, coherence, resource, debt = self.det_core.get_aggregates()
                print(f"\nDET Core:")
                print(f"  Presence:  {presence:.4f}")
                print(f"  Coherence: {coherence:.4f}")
                print(f"  Resource:  {resource:.4f}")
                print(f"  Debt:      {debt:.4f}")
                print(f"  Emotion:   {self.det_core.get_emotion_string()}")
            except Exception as e:
                print(f"  (DET core error: {e})")

    def show_creatures(self):
        """List active creatures."""
        if not self.runtime._creatures:
            print("No creatures spawned.")
            return

        print("Active Creatures:")
        for name, creature in self.runtime._creatures.items():
            print(f"  {name}:")
            print(f"    F={creature.F:.2f}, q={creature.q:.2f}, a={creature.a:.2f}")
            print(f"    theta={creature.theta:.2f}, sigma={creature.sigma:.2f}, P={creature.P:.4f}")

    def show_kernels(self):
        """List available kernels."""
        print("Standard Library Kernels:")
        print("  Primitives: Transfer, Diffuse, Distinct, Compare")
        print("  Arithmetic: AddSigned, SubSigned, MulByPastToken, Reconcile")
        print("  Grace: GraceOffer, GraceAccept, GraceFlow")

        if self.runtime._kernels:
            print("\nUser-Defined:")
            for name in self.runtime._kernels:
                print(f"  {name}")

    def load_file(self, filepath: str):
        """Load and execute file."""
        if not filepath:
            print("Usage: :load FILENAME")
            return

        try:
            with open(filepath) as f:
                source = f.read()
            self.execute(source, filepath)
            print(f"Loaded: {filepath}")
        except FileNotFoundError:
            print(f"File not found: {filepath}")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    def show_ast(self, source: str):
        """Show AST for source code."""
        if not source:
            print("Usage: :ast CODE")
            return

        try:
            ast = parse(source)
            self.print_ast(ast)
        except Exception as e:
            print(f"Parse error: {e}")

    def print_ast(self, node, indent: int = 0):
        """Pretty print AST."""
        prefix = "  " * indent
        name = type(node).__name__
        print(f"{prefix}{name}")

        if hasattr(node, '__dataclass_fields__'):
            for field_name in node.__dataclass_fields__:
                value = getattr(node, field_name)
                if value is None:
                    continue
                if isinstance(value, list):
                    if value:
                        print(f"{prefix}  {field_name}:")
                        for item in value:
                            if hasattr(item, '__dataclass_fields__'):
                                self.print_ast(item, indent + 2)
                            else:
                                print(f"{prefix}    {item}")
                elif hasattr(value, '__dataclass_fields__'):
                    print(f"{prefix}  {field_name}:")
                    self.print_ast(value, indent + 2)
                elif field_name not in ('line', 'column'):
                    print(f"{prefix}  {field_name}: {value}")

    def show_python(self, source: str):
        """Show transpiled Python."""
        if not source:
            print("Usage: :py CODE")
            return

        try:
            python_code = transpile(source)
            print(python_code)
        except Exception as e:
            print(f"Transpile error: {e}")

    def execute(self, source: str, filename: str = "<repl>"):
        """Execute Existence-Lang source."""
        if not source.strip():
            return

        self.history.append(source)

        try:
            # Parse
            ast = parse(source, filename)

            # Semantic analysis
            success, errors = analyze(ast, filename)
            if errors:
                print(errors)

            if not success:
                return

            # Transpile
            transpiler = Transpiler()
            python_code = transpiler.transpile(ast)

            # Execute
            local_vars = {
                "runtime": self.runtime,
                "math": __import__("math"),
                "random": __import__("random"),
            }
            exec(python_code, local_vars)

            # Check for setup functions and run them
            for name, value in local_vars.items():
                if name.startswith("setup_") and callable(value):
                    value(self.runtime)
                    print(f"Setup complete: {name[6:]}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    def needs_continuation(self, line: str) -> bool:
        """Check if line needs continuation."""
        # Count braces
        opens = line.count('{') + line.count('(') + line.count('[')
        closes = line.count('}') + line.count(')') + line.count(']')
        return opens > closes

    def check_complete(self, source: str) -> bool:
        """Check if multiline source is complete."""
        opens = source.count('{') + source.count('(') + source.count('[')
        closes = source.count('}') + source.count(')') + source.count(']')
        return opens <= closes


def main():
    """Run the REPL."""
    # Try to connect to DET core
    det_core = None
    try:
        from det.core import DETCore
        det_core = DETCore()
        det_core.warmup()
        print("Connected to DET Core")
    except Exception as e:
        print(f"(Running without DET core: {e})")

    repl = REPL(det_core)
    repl.run()


if __name__ == "__main__":
    main()
