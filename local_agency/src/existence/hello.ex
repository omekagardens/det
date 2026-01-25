/**
 * Hello World Creature - Minimal Example
 */

creature HelloCreature {
    // DET state
    var F: Register := 10.0;
    var a: float := 0.5;

    // State
    var greeting_count: int := 0;

    /**
     * Greet kernel - simple greeting
     */
    kernel Greet {
        in  target_name: TokenReg;
        out greeting: TokenReg;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal DO_GREET {
                score = a;
                effect {
                    F := F - 0.1;
                    greeting_count := greeting_count + 1;
                    greeting ::= witness("Hello!");
                }
            }
        }
    }

    /**
     * Status kernel
     */
    kernel Status {
        out status: TokenReg;

        phase READ {
            current_F ::= witness(F);
        }

        phase PROPOSE {
            proposal REPORT {
                score = 1.0;
                effect {
                    status ::= witness(current_F);
                }
            }
        }
    }
}
