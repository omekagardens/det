"""
Existence-Lang Arithmetic Kernels
=================================

Arithmetic as ledger of agency: AddSigned, SubSigned, MulByPastToken, Reconcile.
These operations are derived from the primitives, not axioms.
"""

from ..runtime import KernelBase, Register, TokenReg, ReconcileResult, CompareResult
from .primitives import Pump, Diffuse, Compare


class AddSigned(KernelBase):
    """
    Signed addition kernel.

    Addition is repetition of agency - combining distinctions.
    Uses two-component signed representation (positive, negative).
    Cost: k += 1

    Ports:
        in  xP: Register - X positive component
        in  xM: Register - X negative component
        in  yP: Register - Y positive component
        in  yM: Register - Y negative component
        out oP: Register - output positive
        out oM: Register - output negative
        out w: TokenReg - witness
    """

    def __init__(self):
        super().__init__()
        self.xP = Register()
        self.xM = Register()
        self.yP = Register()
        self.yM = Register()
        self.oP = Register()
        self.oM = Register()
        self.w = TokenReg()
        self._params["J_step"] = 0.01

    def phase_COMMIT(self):
        """Execute addition via flux transfer."""
        j_step = self._params.get("J_step", 0.01)

        # Create pump for transfers
        pump = Pump(j_step)

        # Transfer positive components to output positive
        pump.move(self.xP, self.oP)
        pump.move(self.yP, self.oP)

        # Transfer negative components to output negative
        pump.move(self.xM, self.oM)
        pump.move(self.yM, self.oM)

        self.w.token = "OK"

    @staticmethod
    def add(x: float, y: float) -> tuple[float, TokenReg]:
        """Convenience static method for simple addition."""
        kernel = AddSigned()

        # Set up signed representation
        if x >= 0:
            kernel.xP.F = x
        else:
            kernel.xM.F = -x

        if y >= 0:
            kernel.yP.F = y
        else:
            kernel.yM.F = -y

        # Direct transfer for convenience method (bypass incremental Pump)
        kernel.oP.F = kernel.xP.F + kernel.yP.F
        kernel.oM.F = kernel.xM.F + kernel.yM.F
        kernel.w.token = "OK"

        result = kernel.oP.F - kernel.oM.F
        return (result, kernel.w)


class SubSigned(KernelBase):
    """
    Signed subtraction kernel.

    Subtraction is reconciliation attempt - trying to undo distinction.
    May fail if reconciliation impossible.
    Cost: k += reconciliation_cost

    Ports:
        in  xP: Register - X positive
        in  xM: Register - X negative
        in  yP: Register - Y positive (subtracted)
        in  yM: Register - Y negative (subtracted)
        out oP: Register - output positive
        out oM: Register - output negative
        out w: TokenReg - SUB_OK, SUB_FAIL, SUB_REFUSE
    """

    def __init__(self):
        super().__init__()
        self.xP = Register()
        self.xM = Register()
        self.yP = Register()
        self.yM = Register()
        self.oP = Register()
        self.oM = Register()
        self.w = TokenReg()
        self._params["J_step"] = 0.01

    def phase_COMMIT(self):
        """Execute subtraction via reconciliation."""
        j_step = self._params.get("J_step", 0.01)
        pump = Pump(j_step)

        # Subtraction: X - Y = (xP - xM) - (yP - yM)
        #            = (xP + yM) - (xM + yP)
        # So: oP gets xP and yM
        #     oM gets xM and yP

        pump.move(self.xP, self.oP)
        pump.move(self.yM, self.oP)
        pump.move(self.xM, self.oM)
        pump.move(self.yP, self.oM)

        self.w.token = "SUB_OK"

    @staticmethod
    def subtract(x: float, y: float) -> tuple[float, TokenReg]:
        """Convenience static method."""
        kernel = SubSigned()

        if x >= 0:
            kernel.xP.F = x
        else:
            kernel.xM.F = -x

        if y >= 0:
            kernel.yP.F = y
        else:
            kernel.yM.F = -y

        kernel.execute()

        result = kernel.oP.F - kernel.oM.F
        return (result, kernel.w)


class MulByPastToken(KernelBase):
    """
    Multiplication by past token kernel.

    Multiplication is nested repetition of agency.
    The multiplier must be a past token (integer).
    Cost: k += count (one per repetition)

    Ports:
        in  base: Register - value to multiply
        in  count: TokenReg - multiplier (must be past token integer)
        out result: Register - output
        out w: TokenReg - witness
    """

    def __init__(self):
        super().__init__()
        self.base = Register()
        self.count = TokenReg()
        self.count.token = 0
        self.result = Register()
        self.w = TokenReg()
        self._params["J_step"] = 0.01

    def phase_COMMIT(self):
        """Execute multiplication via repeated addition."""
        count = int(self.count.token) if self.count.token else 0
        j_step = self._params.get("J_step", 0.01)

        # Initialize result
        self.result.F = 0

        if count <= 0:
            self.w.token = "OK"
            return

        # Create temporary registers for signed addition
        base_val = self.base.F

        # Repeat addition count times
        for _ in range(count):
            self.result.F += base_val

        self.w.token = "OK"

    @staticmethod
    def multiply(base: float, count: int) -> tuple[float, TokenReg]:
        """Convenience static method."""
        kernel = MulByPastToken()
        kernel.base.F = base
        kernel.count.token = count
        kernel.execute()
        return (kernel.result.F, kernel.w)


class DivByPastToken(KernelBase):
    """
    Division by past token kernel.

    Division asks: how many times can X reconcile with Y?
    Returns quotient and remainder.

    Ports:
        in  numerator: Register
        in  denominator: Register
        out quotient: TokenReg - integer count of reconciliations
        out remainder: Register - unreconciled portion
        out w: TokenReg
    """

    def __init__(self):
        super().__init__()
        self.numerator = Register()
        self.denominator = Register()
        self.quotient = TokenReg()
        self.remainder = Register()
        self.w = TokenReg()

    def phase_COMMIT(self):
        """Execute division via repeated reconciliation."""
        num = self.numerator.F
        denom = self.denominator.F

        if denom <= 0:
            self.quotient.token = 0
            self.remainder.F = num
            self.w.token = "DIV_BY_ZERO"
            return

        # Count reconciliations
        count = int(num / denom)
        self.quotient.token = count
        self.remainder.F = num - (count * denom)
        self.w.token = "OK"


class Reconcile(KernelBase):
    """
    Reconciliation kernel (= operator).

    Equality is not a logical assertion - it's an attempted act of unification.
    May fail or be refused based on agency.

    Ports:
        inout A: Register
        inout B: Register
        out w: TokenReg - EQ_OK, EQ_FAIL, EQ_REFUSE
    """

    def __init__(self):
        super().__init__()
        self.A = Register()
        self.B = Register()
        self.w = TokenReg()
        self._params["epsilon"] = 1e-3
        self._params["J_step"] = 0.01
        self._params["max_steps"] = 100

    def phase_COMMIT(self):
        """Attempt reconciliation."""
        epsilon = self._params.get("epsilon", 1e-3)
        j_step = self._params.get("J_step", 0.01)
        max_steps = self._params.get("max_steps", 100)

        # Check initial difference
        diff = abs(self.A.F - self.B.F)

        if diff < epsilon:
            self.w.token = ReconcileResult.EQ_OK
            return

        # Attempt diffusion toward equality
        for _ in range(max_steps):
            # Diffuse
            gradient = self.A.F - self.B.F
            flux = min(j_step, abs(gradient) * 0.5)

            if gradient > 0:
                self.A.F -= flux
                self.B.F += flux
            else:
                self.A.F += flux
                self.B.F -= flux

            # Check convergence
            if abs(self.A.F - self.B.F) < epsilon:
                self.w.token = ReconcileResult.EQ_OK
                return

        # Failed to reconcile within max steps
        self.w.token = ReconcileResult.EQ_FAIL

    @staticmethod
    def attempt(a: float, b: float) -> tuple[float, float, ReconcileResult]:
        """Convenience static method."""
        kernel = Reconcile()
        kernel.A.F = a
        kernel.B.F = b
        kernel.execute()
        return (kernel.A.F, kernel.B.F, kernel.w.token)


class ConvergeTo(KernelBase):
    """
    Converge to target kernel (<= operator).

    Attempts to make X converge to target value over time.
    This is a convergent kernel - may take multiple ticks.

    Ports:
        inout X: Register - value to adjust
        in  target: Register - target value
        out w: TokenReg - EQ when converged, NEQ otherwise
    """

    def __init__(self):
        super().__init__()
        self.X = Register()
        self.target = Register()
        self.w = TokenReg()
        self._params["J_step"] = 0.01
        self._params["epsilon"] = 1e-3

    def phase_COMMIT(self):
        """Single step toward target."""
        j_step = self._params.get("J_step", 0.01)
        epsilon = self._params.get("epsilon", 1e-3)

        diff = self.target.F - self.X.F
        step = min(j_step, abs(diff))

        if diff > 0:
            self.X.F += step
        else:
            self.X.F -= step

        # Check convergence
        if abs(self.X.F - self.target.F) < epsilon:
            self.w.token = CompareResult.EQ
        else:
            self.w.token = CompareResult.LT if self.X.F < self.target.F else CompareResult.GT
