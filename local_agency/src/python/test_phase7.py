#!/usr/bin/env python3
"""
Phase 7 Tests: Existence-Lang Implementation
============================================

Tests for the Existence-Lang agency-first programming language.
"""

import unittest
import math
import sys
import os

# Add det directory to path directly (bypass main det package)
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TEST_DIR)

# Create a minimal det package without the full dependencies
# by setting up det.lang as a standalone import
import importlib.util

def load_module_direct(name, path):
    """Load a module directly from path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    return spec, module

# Load lang modules directly
LANG_DIR = os.path.join(TEST_DIR, "det", "lang")
STDLIB_DIR = os.path.join(LANG_DIR, "stdlib")

# Create package namespaces
class DummyPackage:
    pass

# Set up det.lang namespace
sys.modules["det"] = DummyPackage()
sys.modules["det.lang"] = DummyPackage()
sys.modules["det.lang.stdlib"] = DummyPackage()

# Load modules in order
spec, errors_mod = load_module_direct("det.lang.errors", os.path.join(LANG_DIR, "errors.py"))
spec.loader.exec_module(errors_mod)

spec, tokens_mod = load_module_direct("det.lang.tokens", os.path.join(LANG_DIR, "tokens.py"))
spec.loader.exec_module(tokens_mod)

spec, ast_mod = load_module_direct("det.lang.ast_nodes", os.path.join(LANG_DIR, "ast_nodes.py"))
spec.loader.exec_module(ast_mod)

spec, parser_mod = load_module_direct("det.lang.parser", os.path.join(LANG_DIR, "parser.py"))
spec.loader.exec_module(parser_mod)

spec, runtime_mod = load_module_direct("det.lang.runtime", os.path.join(LANG_DIR, "runtime.py"))
spec.loader.exec_module(runtime_mod)

spec, transpiler_mod = load_module_direct("det.lang.transpiler", os.path.join(LANG_DIR, "transpiler.py"))
spec.loader.exec_module(transpiler_mod)

spec, semantic_mod = load_module_direct("det.lang.semantic", os.path.join(LANG_DIR, "semantic.py"))
spec.loader.exec_module(semantic_mod)

# Load stdlib
spec, primitives_mod = load_module_direct("det.lang.stdlib.primitives", os.path.join(STDLIB_DIR, "primitives.py"))
spec.loader.exec_module(primitives_mod)

spec, arithmetic_mod = load_module_direct("det.lang.stdlib.arithmetic", os.path.join(STDLIB_DIR, "arithmetic.py"))
spec.loader.exec_module(arithmetic_mod)

spec, grace_mod = load_module_direct("det.lang.stdlib.grace", os.path.join(STDLIB_DIR, "grace.py"))
spec.loader.exec_module(grace_mod)

# Import from loaded modules
Lexer = tokens_mod.Lexer
TokenType = tokens_mod.TokenType
Token = tokens_mod.Token

Creature = ast_mod.Creature
Kernel = ast_mod.Kernel
Presence = ast_mod.Presence
IfPast = ast_mod.IfPast
Program = ast_mod.Program

parse = parser_mod.parse
Parser = parser_mod.Parser

transpile = transpiler_mod.transpile
Transpiler = transpiler_mod.Transpiler

Register = runtime_mod.Register
TokenReg = runtime_mod.TokenReg
CreatureBase = runtime_mod.CreatureBase
KernelBase = runtime_mod.KernelBase
BondInstance = runtime_mod.BondInstance
ExistenceRuntime = runtime_mod.ExistenceRuntime
distinct = runtime_mod.distinct
transfer = runtime_mod.transfer
diffuse = runtime_mod.diffuse
compare = runtime_mod.compare
reconcile = runtime_mod.reconcile
CompareResult = runtime_mod.CompareResult
ReconcileResult = runtime_mod.ReconcileResult
TransferResult = runtime_mod.TransferResult

analyze = semantic_mod.analyze

Transfer = primitives_mod.Transfer
Diffuse = primitives_mod.Diffuse
CompareKernel = primitives_mod.Compare

AddSigned = arithmetic_mod.AddSigned
ReconcileKernel = arithmetic_mod.Reconcile

GraceFlow = grace_mod.GraceFlow


class TestLexer(unittest.TestCase):
    """Test Existence-Lang lexer."""

    def test_basic_tokens(self):
        """Test basic token recognition."""
        lexer = Lexer("creature Test { var x: float := 1.0; }")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        self.assertIn(TokenType.CREATURE, types)
        self.assertIn(TokenType.IDENTIFIER, types)
        self.assertIn(TokenType.VAR, types)
        self.assertIn(TokenType.ALIAS_EQ, types)
        self.assertIn(TokenType.FLOAT, types)

    def test_four_equalities(self):
        """Test the four equality operators."""
        lexer = Lexer(":= == = ===")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertEqual(types, [
            TokenType.ALIAS_EQ,     # :=
            TokenType.MEASURE_EQ,   # ==
            TokenType.RECONCILE_EQ, # =
            TokenType.COVENANT_EQ,  # ===
        ])

    def test_witness_binding(self):
        """Test witness binding operator ::=."""
        lexer = Lexer("x ::= compare(a, b)")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        self.assertIn(TokenType.WITNESS_BIND, types)

    def test_control_flow_keywords(self):
        """Test control flow keywords."""
        lexer = Lexer("if_past else_past repeat_past while_past")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens if t.type != TokenType.EOF]
        self.assertEqual(types, [
            TokenType.IF_PAST,
            TokenType.ELSE_PAST,
            TokenType.REPEAT_PAST,
            TokenType.WHILE_PAST,
        ])

    def test_numbers(self):
        """Test number literals."""
        lexer = Lexer("42 3.14 1e-5 1_000")
        tokens = lexer.tokenize()

        values = [(t.type, t.value) for t in tokens if t.type != TokenType.EOF]
        self.assertEqual(values[0], (TokenType.INTEGER, "42"))
        self.assertEqual(values[1], (TokenType.FLOAT, "3.14"))
        self.assertEqual(values[2], (TokenType.FLOAT, "1e-5"))
        self.assertEqual(values[3], (TokenType.INTEGER, "1000"))

    def test_strings(self):
        """Test string literals."""
        lexer = Lexer('"hello" \'world\'')
        tokens = lexer.tokenize()

        strings = [t.value for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(strings, ["hello", "world"])

    def test_comments(self):
        """Test comment handling."""
        lexer = Lexer("x // comment\ny /* multi\nline */ z")
        tokens = lexer.tokenize()

        idents = [t.value for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertEqual(idents, ["x", "y", "z"])


class TestParser(unittest.TestCase):
    """Test Existence-Lang parser."""

    def test_parse_creature(self):
        """Test parsing creature definition."""
        source = """
        creature Thermostat {
            var target: float := 22.0;
            sensor temp: Temperature @ channel(0);
            actuator heater: Switch @ channel(1);

            agency {
                this.sigma *= 0.5;
            }
        }
        """
        ast = parse(source)

        self.assertEqual(len(ast.declarations), 1)
        creature = ast.declarations[0]
        self.assertIsInstance(creature, Creature)
        self.assertEqual(creature.name, "Thermostat")
        self.assertEqual(len(creature.variables), 1)
        self.assertEqual(len(creature.sensors), 1)
        self.assertEqual(len(creature.actuators), 1)
        self.assertIsNotNone(creature.agency)

    def test_parse_kernel(self):
        """Test parsing kernel definition."""
        source = """
        kernel AddSigned {
            in xP: Register;
            in xM: Register;
            out oP: Register;
            out w: TokenReg;

            params {
                J_step = 0.01;
            }

            phase COMMIT {
                proposal RUN {
                    score = 1.0;
                    effect {
                        transfer(xP, oP);
                    }
                }
            }
        }
        """
        ast = parse(source)

        self.assertEqual(len(ast.declarations), 1)
        kernel = ast.declarations[0]
        self.assertIsInstance(kernel, Kernel)
        self.assertEqual(kernel.name, "AddSigned")
        self.assertEqual(len(kernel.ports), 4)

    def test_parse_presence(self):
        """Test parsing presence definition."""
        source = """
        presence Home {
            creatures {
                thermo: Thermostat;
            }
            init {
                inject_F(thermo, 100.0);
            }
        }
        """
        ast = parse(source)

        self.assertEqual(len(ast.declarations), 1)
        presence = ast.declarations[0]
        self.assertIsInstance(presence, Presence)
        self.assertEqual(presence.name, "Home")
        self.assertIsNotNone(presence.creatures_block)
        self.assertIn("thermo", presence.creatures_block.creatures)

    def test_parse_if_past(self):
        """Test parsing if_past control flow."""
        source = """
        creature Test {
            agency {
                comparison ::= compare(a, b);
                if_past(comparison == LT) {
                    x = 1;
                } else {
                    x = 2;
                }
            }
        }
        """
        ast = parse(source)
        creature = ast.declarations[0]
        stmts = creature.agency.body.statements

        # Find if_past statement
        if_past = None
        for stmt in stmts:
            if isinstance(stmt, IfPast):
                if_past = stmt
                break

        self.assertIsNotNone(if_past)
        self.assertIsNotNone(if_past.then_block)
        self.assertIsNotNone(if_past.else_block)

    def test_parse_expressions(self):
        """Test parsing various expressions."""
        source = """
        creature Test {
            var a: float := 1.0 + 2.0 * 3.0;
            var b: float := this.F - 10;
            var c: bool := true && false;
        }
        """
        ast = parse(source)
        creature = ast.declarations[0]

        self.assertEqual(len(creature.variables), 3)


class TestTranspiler(unittest.TestCase):
    """Test Existence-Lang transpiler."""

    def test_transpile_creature(self):
        """Test transpiling creature to Python."""
        source = """
        creature Counter {
            var count: int := 0;

            agency {
                this.count = this.count + 1;
            }
        }
        """
        python_code = transpile(source)

        self.assertIn("class Counter(CreatureBase)", python_code)
        self.assertIn("self.count = 0", python_code)
        self.assertIn("def agency_block(self)", python_code)

    def test_transpile_presence(self):
        """Test transpiling presence to setup function."""
        source = """
        creature Simple {
            var x: float := 1.0;
        }

        presence Test {
            creatures {
                s: Simple;
            }
            init {
                inject_F(s, 50.0);
            }
        }
        """
        python_code = transpile(source)

        self.assertIn("def setup_Test(runtime)", python_code)
        self.assertIn("spawn_creature", python_code)

    def test_transpile_if_past(self):
        """Test transpiling if_past to Python if."""
        source = """
        creature Test {
            agency {
                result ::= compare(1, 2);
                if_past(result == LT) {
                    x = 1;
                }
            }
        }
        """
        python_code = transpile(source)

        self.assertIn("if", python_code)
        self.assertIn("CompareResult.LT", python_code)


class TestRuntime(unittest.TestCase):
    """Test Existence-Lang runtime."""

    def test_register(self):
        """Test Register class."""
        r = Register()
        r.F = 100.0
        self.assertEqual(r.F, 100.0)
        self.assertEqual(r.value, 100.0)

        # Test signed representation
        r.F = -50.0
        self.assertEqual(r.F, -50.0)

    def test_token_reg(self):
        """Test TokenReg class."""
        t = TokenReg()
        t.token = CompareResult.EQ
        self.assertEqual(t, CompareResult.EQ)

    def test_creature_base(self):
        """Test CreatureBase class."""
        c = CreatureBase(name="test")
        self.assertEqual(c.name, "test")
        self.assertEqual(c.F, 100.0)
        self.assertEqual(c.a, 1.0)

        # Test agency ceiling
        c._q = 10.0  # High debt
        self.assertLess(c.a, 1.0)

    def test_primitive_distinct(self):
        """Test distinct() primitive."""
        a, b = distinct()
        self.assertIsNot(a, b)
        self.assertNotEqual(a.F, b.F)

    def test_primitive_transfer(self):
        """Test transfer() primitive."""
        src = Register()
        dst = Register()
        src.F = 100.0
        dst.F = 0.0

        result = transfer(src, dst, 30.0)

        self.assertEqual(result, TransferResult.TRANSFER_OK)
        self.assertEqual(src.F, 70.0)
        self.assertEqual(dst.F, 30.0)

    def test_primitive_compare(self):
        """Test compare() primitive."""
        self.assertEqual(compare(1, 2), CompareResult.LT)
        self.assertEqual(compare(2, 1), CompareResult.GT)
        self.assertEqual(compare(1, 1), CompareResult.EQ)

    def test_primitive_reconcile(self):
        """Test reconcile() primitive."""
        a = Register()
        b = Register()
        a.F = 100.0
        b.F = 100.001

        result = reconcile(a, b)
        self.assertEqual(result, ReconcileResult.EQ_OK)

    def test_existence_runtime(self):
        """Test ExistenceRuntime."""
        runtime = ExistenceRuntime()
        runtime.register_creature_type("Test", CreatureBase)
        c = runtime.spawn_creature("test1", "Test")

        self.assertEqual(len(runtime._creatures), 1)
        self.assertEqual(c.name, "test1")

        runtime.step()
        self.assertEqual(runtime._tick, 1)


class TestStdlib(unittest.TestCase):
    """Test standard library kernels."""

    def test_transfer_kernel(self):
        """Test Transfer kernel."""
        kernel = Transfer()
        kernel.src.F = 100.0
        kernel.dst.F = 0.0
        kernel.amount.F = 50.0

        kernel.execute()

        self.assertEqual(kernel.witness.token, TransferResult.TRANSFER_OK)
        self.assertEqual(kernel.src.F, 50.0)
        self.assertEqual(kernel.dst.F, 50.0)

    def test_diffuse_kernel(self):
        """Test Diffuse kernel."""
        kernel = Diffuse()
        kernel.a.F = 100.0
        kernel.b.F = 0.0

        kernel.execute()

        self.assertEqual(kernel.a.F, 50.0)
        self.assertEqual(kernel.b.F, 50.0)
        self.assertEqual(kernel.flux.F, 50.0)

    def test_compare_kernel(self):
        """Test Compare kernel."""
        kernel = CompareKernel()
        kernel.src_a.F = 10.0
        kernel.src_b.F = 20.0

        kernel.execute()

        self.assertEqual(kernel.token.token, CompareResult.LT)

    def test_add_signed(self):
        """Test AddSigned kernel convenience method."""
        result, witness = AddSigned.add(10.0, 20.0)
        self.assertEqual(result, 30.0)

        result, witness = AddSigned.add(-5.0, 3.0)
        self.assertEqual(result, -2.0)

    def test_reconcile_kernel(self):
        """Test Reconcile kernel."""
        a_val, b_val, result = ReconcileKernel.attempt(100.0, 100.001)
        self.assertEqual(result, ReconcileResult.EQ_OK)

    def test_grace_flow(self):
        """Test GraceFlow kernel."""
        kernel = GraceFlow()
        kernel.node_i.F = 100.0
        kernel.node_j.F = 20.0
        kernel.coherence.F = 0.5

        kernel.execute()

        # Grace should flow from high-F to low-F
        self.assertLess(kernel.node_i.F, 100.0)
        self.assertGreater(kernel.node_j.F, 20.0)


class TestSemantic(unittest.TestCase):
    """Test semantic analysis."""

    def test_undefined_variable(self):
        """Test detection of undefined variables."""
        source = """
        creature Test {
            agency {
                x = undefined_var + 1;
            }
        }
        """
        ast = parse(source)
        success, errors = analyze(ast)

        self.assertFalse(success)
        self.assertIn("undefined_var", errors.lower())

    def test_temporal_semantics_warning(self):
        """Test warning for potential temporal violations."""
        source = """
        creature Test {
            var x: float := 1.0;
            agency {
                if_past(x > 0) {
                    y = 1;
                }
            }
        }
        """
        ast = parse(source)
        success, report = analyze(ast)

        # Should produce warning about direct comparison
        self.assertIn("temporal", report.lower())


class TestIntegration(unittest.TestCase):
    """Integration tests."""

    def test_full_thermostat(self):
        """Test full thermostat example."""
        source = """
        creature Thermostat {
            var target: float := 22.0;

            agency {
                comparison ::= compare(this.F, target);
                if_past(comparison == LT) {
                    this.sigma = 1.5;
                } else {
                    this.sigma = 0.5;
                }
            }
        }

        presence Home {
            creatures {
                thermo: Thermostat;
            }
            init {
                inject_F(thermo, 100.0);
            }
        }
        """

        # Parse and transpile
        python_code = transpile(source)

        # Execute
        runtime = ExistenceRuntime()
        local_vars = {
            "runtime": runtime,
            "math": math,
            "random": __import__("random"),
        }
        exec(python_code, local_vars)

        # Run setup
        local_vars["setup_Home"](runtime)

        # Verify creature was created
        self.assertIn("thermo", runtime._creatures)

        # Step simulation
        runtime.step()
        runtime.step()

        # Check state
        thermo = runtime._creatures["thermo"]
        self.assertGreater(thermo.F, 0)

    def test_kernel_execution(self):
        """Test kernel-style execution."""
        source = """
        kernel MyAdd {
            in a: Register;
            in b: Register;
            out result: Register;
            out w: TokenReg;

            phase COMMIT {
                proposal ADD {
                    score = 1.0;
                    effect {
                        result.F = a.F + b.F;
                    }
                }
                commit ADD;
                w ::= "OK";
            }
        }
        """

        python_code = transpile(source)
        local_vars = {"math": math, "random": __import__("random")}
        exec(python_code, local_vars)

        # Create and execute kernel
        MyAdd = local_vars["MyAdd"]
        kernel = MyAdd()
        kernel.a.F = 10.0
        kernel.b.F = 20.0
        kernel.execute()

        self.assertEqual(kernel.result.F, 30.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_empty_creature(self):
        """Test parsing empty creature."""
        source = "creature Empty { }"
        ast = parse(source)
        self.assertEqual(len(ast.declarations), 1)

    def test_nested_expressions(self):
        """Test deeply nested expressions."""
        source = """
        creature Test {
            var x: float := ((1 + 2) * (3 - 4)) / 5;
        }
        """
        ast = parse(source)
        self.assertEqual(len(ast.declarations), 1)

    def test_unicode_covenant(self):
        """Test unicode covenant equality."""
        lexer = Lexer("a ≡ b")
        tokens = lexer.tokenize()

        types = [t.type for t in tokens]
        self.assertIn(TokenType.COVENANT_EQ, types)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLexer))
    suite.addTests(loader.loadTestsFromTestCase(TestParser))
    suite.addTests(loader.loadTestsFromTestCase(TestTranspiler))
    suite.addTests(loader.loadTestsFromTestCase(TestRuntime))
    suite.addTests(loader.loadTestsFromTestCase(TestStdlib))
    suite.addTests(loader.loadTestsFromTestCase(TestSemantic))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())
