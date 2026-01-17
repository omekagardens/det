#!/usr/bin/env python3
"""
DET Local Agency - CLI Entry Point
==================================

Run with: python det_cli.py
"""

import sys
from pathlib import Path

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from det.cli import main

if __name__ == "__main__":
    sys.exit(main())
