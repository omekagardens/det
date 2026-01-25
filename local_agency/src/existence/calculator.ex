/**
 * CalculatorCreature - Math Expression Evaluator
 * ===============================================
 *
 * A simple creature that evaluates mathematical expressions.
 *
 * Kernels:
 *   - Eval: Evaluate a math expression string
 *   - Add, Sub, Mul, Div: Basic binary operations
 *
 * Usage (in det_os_boot):
 *   load calculator
 *   bond calculator
 *   send calculator eval_math "2 + 3 * 4"
 *
 * Supported operations: +, -, *, /, **, (), sqrt, sin, cos, tan, log, abs
 */

creature CalculatorCreature {
    var result: Register = 0.0;
    var last_expr: TokenReg = "";

    /**
     * Eval kernel - Evaluate a math expression string
     *
     * Input: expr (TokenReg) - Math expression like "2 + 3 * 4"
     * Output: result (TokenReg) - Computed result or error message
     */
    kernel Eval {
        in expr: TokenReg;
        out result: TokenReg;

        phase READ {
            // Read the input expression
            input_expr ::= witness(expr);
        }

        phase PROPOSE {
            proposal EvalExpr {
                score = 1.0;
                effect {
                    result ::= primitive("eval_math", expr);
                }
            }
        }

        phase CHOOSE {
            choice := choose({EvalExpr});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Add kernel - Add two numbers
     */
    kernel Add {
        in a: Register;
        in b: Register;
        out result: Register;

        phase READ {
            val_a ::= witness(a);
            val_b ::= witness(b);
        }

        phase PROPOSE {
            proposal DoAdd {
                score = 1.0;
                effect {
                    result = a + b;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DoAdd});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Sub kernel - Subtract two numbers
     */
    kernel Sub {
        in a: Register;
        in b: Register;
        out result: Register;

        phase READ {
            val_a ::= witness(a);
            val_b ::= witness(b);
        }

        phase PROPOSE {
            proposal DoSub {
                score = 1.0;
                effect {
                    result = a - b;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DoSub});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Mul kernel - Multiply two numbers
     */
    kernel Mul {
        in a: Register;
        in b: Register;
        out result: Register;

        phase READ {
            val_a ::= witness(a);
            val_b ::= witness(b);
        }

        phase PROPOSE {
            proposal DoMul {
                score = 1.0;
                effect {
                    result = a * b;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DoMul});
        }

        phase COMMIT {
            commit choice;
        }
    }

    /**
     * Div kernel - Divide two numbers
     */
    kernel Div {
        in a: Register;
        in b: Register;
        out result: Register;

        phase READ {
            val_a ::= witness(a);
            val_b ::= witness(b);
        }

        phase PROPOSE {
            proposal DoDiv {
                score = 1.0;
                effect {
                    result = a / b;
                }
            }
        }

        phase CHOOSE {
            choice := choose({DoDiv});
        }

        phase COMMIT {
            commit choice;
        }
    }
}
