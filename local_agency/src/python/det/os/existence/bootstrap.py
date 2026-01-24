"""
DET-OS Bootstrap - Loads and runs the Existence-Lang kernel
============================================================

The bootstrap is the minimal code that:
1. Initializes the DET Core (physics engine)
2. Initializes the EIS VM (instruction execution)
3. Loads the Existence-Lang kernel
4. Compiles kernel.ex to EIS bytecode
5. Executes the kernel on the VM

With DET-native hardware, steps 2-5 become direct silicon execution.
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable
import os
import time

# Import the architecture VM layer (minimal C)
try:
    from det.core import DETCore
    DETCORE_AVAILABLE = True
except ImportError:
    DETCORE_AVAILABLE = False
    DETCore = None

# Import Existence-Lang compiler
try:
    from det.lang import parse as parse_existence, transpile as transpile_existence
    from det.lang.runtime import ExistenceRuntime, CreatureBase, KernelBase, Register
    LANG_AVAILABLE = True
except ImportError:
    LANG_AVAILABLE = False
    parse_existence = None
    transpile_existence = None
    ExistenceRuntime = None

# Import EIS VM
try:
    from det.lang.eis import EISVM, EISProgram
    from det.lang.eis.compiler import EISCompiler
    EIS_AVAILABLE = True
except ImportError:
    EIS_AVAILABLE = False
    EISVM = None
    EISProgram = None
    EISCompiler = None

# Import the runtime (must be after other imports to avoid circular)
from .runtime import ExistenceKernelRuntime


class BootState(Enum):
    """Bootstrap state machine."""
    INIT = auto()           # Just created
    LOADING = auto()        # Loading kernel source
    COMPILING = auto()      # Compiling to EIS
    LINKING = auto()        # Linking with runtime
    BOOTING = auto()        # Kernel initializing
    RUNNING = auto()        # Kernel running
    HALTED = auto()         # Kernel stopped
    PANIC = auto()          # Unrecoverable error


@dataclass
class BootConfig:
    """Bootstrap configuration."""
    kernel_path: str = ""                    # Path to kernel.ex
    total_memory: int = 1024 * 1024 * 1024   # 1GB
    total_F: float = 1000000.0               # Total resource
    grace_pool: float = 10000.0              # Initial grace
    tick_rate: float = 50.0                  # Hz
    max_creatures: int = 4096
    debug: bool = False


@dataclass
class KernelState:
    """Runtime kernel state."""
    tick: int = 0
    num_creatures: int = 1
    total_F: float = 0.0
    grace_pool: float = 0.0
    scheduled_cid: int = -1


class DETOSBootstrap:
    """
    DET-OS Bootstrap Loader

    Boots the Existence-Lang kernel on the architecture VM layer.

    The architecture is:
        Existence-Lang Kernel (kernel.ex)
              │
              ▼ compiles to
        EIS Bytecode
              │
              ▼ executes on
        Architecture VM (EIS VM + DET Core)
              │
              ▼ future: direct silicon
        DET-Native Hardware
    """

    def __init__(self, config: Optional[BootConfig] = None):
        """Initialize bootstrap."""
        self.config = config or BootConfig()
        self.state = BootState.INIT

        # Architecture VM layer
        self.det_core: Optional[Any] = None
        self.eis_vm: Optional[Any] = None

        # Kernel state
        self.kernel_source: str = ""
        self.kernel_ast: Optional[Any] = None
        self.kernel_eis: Optional[Any] = None
        self.kernel_state = KernelState()

        # Runtime
        self.runtime: Optional[Any] = None

        # Callbacks for syscalls (bridge to host)
        self.syscall_handlers: Dict[str, Callable] = {}

        # Boot log
        self.boot_log: List[str] = []

    def _log(self, msg: str):
        """Log boot message."""
        timestamp = time.time()
        entry = f"[{timestamp:.3f}] {msg}"
        self.boot_log.append(entry)
        if self.config.debug:
            print(f"BOOT: {msg}")

    def boot(self) -> bool:
        """
        Boot the DET-OS kernel.

        Returns True if boot successful, False otherwise.
        """
        try:
            self._log("Starting DET-OS bootstrap...")

            # 1. Initialize architecture VM layer
            self._init_arch_vm()

            # 2. Load kernel source
            self._load_kernel()

            # 3. Compile to EIS
            self._compile_kernel()

            # 4. Link with runtime
            self._link_runtime()

            # 5. Boot kernel
            self._boot_kernel()

            self.state = BootState.RUNNING
            self._log("DET-OS kernel running")
            return True

        except Exception as e:
            self.state = BootState.PANIC
            self._log(f"PANIC: {e}")
            return False

    def _init_arch_vm(self):
        """Initialize the architecture VM layer (minimal C)."""
        self.state = BootState.INIT
        self._log("Initializing architecture VM layer...")

        # Initialize DET Core (physics engine)
        if DETCORE_AVAILABLE and DETCore:
            self.det_core = DETCore()
            self._log(f"  DET Core initialized ({self.det_core.num_nodes} nodes)")
        else:
            self._log("  DET Core not available - using stub")
            self.det_core = StubDETCore()

        # Initialize EIS VM
        if EIS_AVAILABLE and EISVM:
            self.eis_vm = EISVM(num_registers=256)
            self._log("  EIS VM initialized (256 registers)")
        else:
            self._log("  EIS VM not available - using interpreted mode")
            self.eis_vm = None

    def _load_kernel(self):
        """Load the Existence-Lang kernel source."""
        self.state = BootState.LOADING
        self._log("Loading kernel source...")

        # Find kernel.ex
        if self.config.kernel_path:
            kernel_path = self.config.kernel_path
        else:
            # Default: src/existence/kernel.ex (relative to project root)
            # From src/python/det/os/existence/ go up to src/existence/
            this_dir = os.path.dirname(__file__)
            kernel_path = os.path.join(this_dir, "..", "..", "..", "..", "existence", "kernel.ex")
            kernel_path = os.path.normpath(kernel_path)

        if os.path.exists(kernel_path):
            with open(kernel_path, 'r') as f:
                self.kernel_source = f.read()
            self._log(f"  Loaded {len(self.kernel_source)} bytes from {kernel_path}")
        else:
            self._log(f"  kernel.ex not found at {kernel_path}")
            # Use embedded minimal kernel
            self.kernel_source = MINIMAL_KERNEL
            self._log("  Using embedded minimal kernel")

    def _compile_kernel(self):
        """Compile kernel to EIS bytecode."""
        self.state = BootState.COMPILING
        self._log("Compiling kernel to EIS...")

        if LANG_AVAILABLE and parse_existence:
            try:
                # Parse Existence-Lang
                self.kernel_ast = parse_existence(self.kernel_source)
                self._log(f"  Parsed {len(self.kernel_ast.creatures)} creatures")

                # Transpile to Python (intermediate)
                python_code = transpile_existence(self.kernel_ast)
                self._log(f"  Transpiled to {len(python_code)} bytes Python")

                # Compile to EIS if available
                if EIS_AVAILABLE and EISCompiler:
                    compiler = EISCompiler()
                    self.kernel_eis = compiler.compile(self.kernel_ast)
                    self._log(f"  Compiled to {len(self.kernel_eis.instructions)} EIS instructions")
                else:
                    self._log("  EIS compiler not available - using interpreted mode")

            except Exception as e:
                self._log(f"  Compilation failed: {e}")
                self._log("  Falling back to Python runtime")
        else:
            self._log("  Existence-Lang compiler not available")
            self._log("  Using Python kernel implementation")

    def _link_runtime(self):
        """Link kernel with runtime support."""
        self.state = BootState.LINKING
        self._log("Linking with runtime...")

        # Create runtime environment
        self.runtime = ExistenceKernelRuntime(
            det_core=self.det_core,
            eis_vm=self.eis_vm,
            config=self.config
        )

        # Register syscall handlers
        self._register_syscalls()

        self._log("  Runtime linked")

    def _register_syscalls(self):
        """Register syscall handlers for host bridge."""
        # These allow the kernel to interact with the host system
        self.syscall_handlers = {
            'host_read': self._syscall_host_read,
            'host_write': self._syscall_host_write,
            'host_time': self._syscall_host_time,
            'host_random': self._syscall_host_random,
        }
        self._log(f"  Registered {len(self.syscall_handlers)} syscall handlers")

    def _boot_kernel(self):
        """Boot the kernel creature."""
        self.state = BootState.BOOTING
        self._log("Booting kernel creature...")

        # Initialize kernel state
        self.kernel_state.total_F = self.config.total_F
        self.kernel_state.grace_pool = self.config.grace_pool
        self.kernel_state.num_creatures = 1  # Kernel itself

        # Create kernel creature in runtime
        self.runtime.boot_kernel(
            total_F=self.config.total_F,
            grace_pool=self.config.grace_pool,
            max_creatures=self.config.max_creatures
        )

        self._log("  Kernel creature initialized")
        self._log(f"  Total F: {self.kernel_state.total_F}")
        self._log(f"  Grace pool: {self.kernel_state.grace_pool}")

    # =========================================================================
    # Public API
    # =========================================================================

    def spawn(self, name: str, initial_f: float = 1.0,
              initial_a: float = 0.5, program: Optional[bytes] = None) -> int:
        """
        Spawn a new creature.

        Returns creature ID (cid).
        """
        if self.state != BootState.RUNNING:
            raise RuntimeError(f"Cannot spawn: kernel state is {self.state}")

        return self.runtime.spawn(name, initial_f, initial_a, program)

    def kill(self, cid: int, reason: str = ""):
        """Kill a creature."""
        if self.state != BootState.RUNNING:
            raise RuntimeError(f"Cannot kill: kernel state is {self.state}")

        self.runtime.kill(cid, reason)

    def tick(self) -> KernelState:
        """Execute one kernel tick."""
        if self.state != BootState.RUNNING:
            raise RuntimeError(f"Cannot tick: kernel state is {self.state}")

        self.runtime.tick(1.0 / self.config.tick_rate)
        self.kernel_state = self.runtime.get_kernel_state()
        return self.kernel_state

    def run(self, max_ticks: Optional[int] = None):
        """Run kernel main loop."""
        if self.state != BootState.RUNNING:
            raise RuntimeError(f"Cannot run: kernel state is {self.state}")

        ticks = 0
        dt = 1.0 / self.config.tick_rate

        while self.state == BootState.RUNNING:
            self.tick()
            ticks += 1

            if max_ticks and ticks >= max_ticks:
                break

            time.sleep(dt)

    def halt(self):
        """Halt the kernel."""
        self._log("Halting kernel...")
        self.state = BootState.HALTED
        self.runtime.halt()

    def get_stats(self) -> Dict:
        """Get kernel statistics."""
        return {
            'state': self.state.name,
            'tick': self.kernel_state.tick,
            'num_creatures': self.kernel_state.num_creatures,
            'total_F': self.kernel_state.total_F,
            'grace_pool': self.kernel_state.grace_pool,
            'boot_log': self.boot_log[-10:],  # Last 10 entries
        }

    # =========================================================================
    # Syscall Handlers (Host Bridge)
    # =========================================================================

    def _syscall_host_read(self, path: str) -> bytes:
        """Read file from host filesystem."""
        with open(path, 'rb') as f:
            return f.read()

    def _syscall_host_write(self, path: str, data: bytes):
        """Write file to host filesystem."""
        with open(path, 'wb') as f:
            f.write(data)

    def _syscall_host_time(self) -> float:
        """Get host system time."""
        return time.time()

    def _syscall_host_random(self) -> float:
        """Get random number from host."""
        import random
        return random.random()


class StubDETCore:
    """Stub DET Core when C library not available."""

    def __init__(self):
        self.tick_count = 0

    def num_nodes(self) -> int:
        return 256

    def step(self, dt: float):
        self.tick_count += 1


# Minimal embedded kernel for when kernel.ex can't be loaded
MINIMAL_KERNEL = '''
/**
 * Minimal DET-OS Kernel
 * Used when kernel.ex cannot be loaded.
 */

creature KernelCreature {
    var F: Register := 1000000.0;
    var a: float := 1.0;
    var tick: int := 0;

    kernel Tick {
        in dt: float;
        phase COMMIT {
            proposal RUN {
                score = 1.0;
                effect { tick := tick + 1; }
            }
            commit choose({RUN});
        }
    }

    agency {
        repeat_past(INFINITY) {
            Tick(0.02);
        }
    }
}

presence DET_OS {
    creatures { kernel: KernelCreature; }
    init { inject_F(kernel, 1000000.0); }
}
'''


__all__ = ['DETOSBootstrap', 'BootState', 'BootConfig', 'KernelState']
