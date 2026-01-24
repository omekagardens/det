"""
DET-OS Existence-Lang Kernel
============================

The operating system kernel written in Existence-Lang.
This is not a simulation - this IS the kernel, expressed in agency-first semantics.

Architecture:
    ┌─────────────────────────────────────────────┐
    │         Existence-Lang (kernel.ex)          │
    │   Schedule, Allocate, Send, Gate, Grace     │
    └─────────────────────────────────────────────┘
                        │ imports
    ┌─────────────────────────────────────────────┐
    │         Existence-Lang (physics.ex)         │
    │   Transfer, Diffuse, Compare, GraceFlow     │
    └─────────────────────────────────────────────┘
                        │
                        ▼ bridges via
    ┌─────────────────────────────────────────────┐
    │         Physics Bridge (Python)             │
    │   physics_bridge.py → PhysicsKernels        │
    └─────────────────────────────────────────────┘
                        │
                        ▼ executes on
    ┌─────────────────────────────────────────────┐
    │         Substrate v2 (C)                    │
    │   Phase-based: READ→PROPOSE→CHOOSE→COMMIT   │
    │   Effects: XFER_F, DIFFUSE, SET_F, etc.     │
    └─────────────────────────────────────────────┘
                        │
                        ▼ future
    ┌─────────────────────────────────────────────┐
    │         DET-Native Hardware                 │
    │   (Direct substrate execution on silicon)   │
    └─────────────────────────────────────────────┘

Usage:
    from det.os.existence import DETOSBootstrap, PhysicsKernels

    # Boot the kernel
    os = DETOSBootstrap()
    os.boot()

    # Or use physics directly
    from det.os.existence import create_physics_runtime
    physics = create_physics_runtime()
    witness = physics.transfer(src_id=0, dst_id=1, amount=10.0)
"""

from .bootstrap import DETOSBootstrap, BootState
from .runtime import ExistenceKernelRuntime
from .physics_bridge import (
    PhysicsKernels,
    SubstrateInterface,
    WitnessToken,
    create_physics_runtime,
    EffectId,
    TokenValue,
    NodeFieldId,
    BondFieldId,
)

__all__ = [
    # Bootstrap
    'DETOSBootstrap',
    'BootState',
    'ExistenceKernelRuntime',
    # Physics
    'PhysicsKernels',
    'SubstrateInterface',
    'WitnessToken',
    'create_physics_runtime',
    'EffectId',
    'TokenValue',
    'NodeFieldId',
    'BondFieldId',
]
