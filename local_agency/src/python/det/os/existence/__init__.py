"""
DET-OS Existence-Lang Kernel
============================

The operating system kernel written in Existence-Lang.
This is not a simulation - this IS the kernel, expressed in agency-first semantics.

Architecture:
    ┌─────────────────────────────────────────────┐
    │         Existence-Lang Kernel               │
    │   (kernel.ex - Schedule, Allocate, etc.)    │
    └─────────────────────────────────────────────┘
                        │
                        ▼ compiles to
    ┌─────────────────────────────────────────────┐
    │              EIS Bytecode                   │
    │   (Existence Instruction Set)               │
    └─────────────────────────────────────────────┘
                        │
                        ▼ executes on
    ┌─────────────────────────────────────────────┐
    │         Minimal C Layer                     │
    │   (EIS VM + DET Core + Hardware Abstraction)│
    └─────────────────────────────────────────────┘
                        │
                        ▼ future
    ┌─────────────────────────────────────────────┐
    │         DET-Native Hardware                 │
    │   (Direct EIS execution on silicon)         │
    └─────────────────────────────────────────────┘

Usage:
    from det.os.existence import DETOSBootstrap

    # Boot the kernel
    os = DETOSBootstrap()
    os.boot()

    # Spawn a creature
    creature = os.spawn("my_app", initial_f=10.0)

    # Run
    os.run()
"""

from .bootstrap import DETOSBootstrap, BootState
from .runtime import ExistenceKernelRuntime

__all__ = [
    'DETOSBootstrap',
    'BootState',
    'ExistenceKernelRuntime',
]
