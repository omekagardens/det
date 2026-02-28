"""F_BH-Drag-3D - Black-hole thermodynamic scaling in 3D collider."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "calibration"))

from black_hole_thermodynamics import BlackHoleThermodynamicsAnalyzer


REQUIRED_FOR_CANONIZATION = True


def test_bh_drag_scaling_3d():
    analyzer = BlackHoleThermodynamicsAnalyzer(grid_size=32, kappa=5.0, verbose=False)
    comparison = analyzer.analyze_mass_scaling(
        masses=[25.0, 35.0, 50.0, 70.0],
        radiation_steps=60,
    )

    # Target: Hawking-like inverse temperature trend (T ~ 1/M) and
    # area-like entropy trend in 3D. Keep tolerances explicit and broad.
    assert -1.8 <= comparison.T_exponent_det <= -0.2
    assert 0.5 <= comparison.entropy_exponent_det <= 3.5

