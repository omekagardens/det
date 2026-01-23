"""
DET Web Application
===================

Phase 6.2 - Local webapp for DET visualization and control.

Provides:
- 3D visualization of DET mind state (nodes, bonds, self-cluster)
- Real-time data feeds via WebSocket
- Stats dashboard and log viewer
- Integration with CLI harness for control
"""

from .server import create_app, run_server, FASTAPI_AVAILABLE
from .api import DETStateAPI

__all__ = [
    "create_app",
    "run_server",
    "DETStateAPI",
    "FASTAPI_AVAILABLE",
]
