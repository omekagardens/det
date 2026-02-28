"""
DET Validation Harness
======================

Tools for testing DET predictions against real-world data.

Modules:
--------
- det_validation_harness: Main CLI tool and validation tests
- gps_data_loader: Load and parse GPS ephemeris data (future)
- bell_data_loader: Load open Bell test datasets (future)

Usage:
------
    from validation.det_validation_harness import run_all_suites
    reports = run_all_suites(verbose=True)
"""

from .det_validation_harness import (
    DETClockModel,
    ValidationResult,
    ValidationReport,
    TestStatus,
    run_all_suites,
    run_gravity_suite,
    run_kepler_suite,
    run_bell_suite,
)

__all__ = [
    'DETClockModel',
    'ValidationResult',
    'ValidationReport',
    'TestStatus',
    'run_all_suites',
    'run_gravity_suite',
    'run_kepler_suite',
    'run_bell_suite',
]
