#!/usr/bin/env python3
"""
Test that PFS metric handles degenerate cases without SVD errors.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.objective import pareto_front_smoothness


def test_degenerate_cases():
    """Test PFS with cases that previously caused SVD errors."""
    print("\n" + "="*80)
    print("Testing PFS Metric - Degenerate Cases")
    print("="*80)

    # Case 1: All points have same x-coordinate
    print("\n1. All points with same x-coordinate:")
    objectives = np.array([
        [0.5, 0.1],
        [0.5, 0.3],
        [0.5, 0.5],
        [0.5, 0.7],
    ])
    pfs = pareto_front_smoothness(objectives)
    print(f"   Objectives:\n{objectives}")
    print(f"   PFS: {pfs:.6f} ✓ (should be 0.0, no SVD error)")
    assert pfs == 0.0, f"Expected 0.0, got {pfs}"

    # Case 2: All points have same y-coordinate
    print("\n2. All points with same y-coordinate:")
    objectives = np.array([
        [0.1, 0.5],
        [0.3, 0.5],
        [0.5, 0.5],
        [0.7, 0.5],
    ])
    pfs = pareto_front_smoothness(objectives)
    print(f"   Objectives:\n{objectives}")
    print(f"   PFS: {pfs:.6f} ✓ (should be 0.0, no SVD error)")
    assert pfs == 0.0, f"Expected 0.0, got {pfs}"

    # Case 3: Only 2 unique x-values (not enough for degree 3 polynomial)
    print("\n3. Only 2 unique x-values:")
    objectives = np.array([
        [0.2, 0.1],
        [0.2, 0.3],
        [0.8, 0.5],
        [0.8, 0.7],
    ])
    pfs = pareto_front_smoothness(objectives)
    print(f"   Objectives:\n{objectives}")
    print(f"   PFS: {pfs:.6f} ✓ (should be 0.0, not enough unique x)")
    assert pfs == 0.0, f"Expected 0.0, got {pfs}"

    # Case 4: Very small variance (near-degenerate)
    print("\n4. Very small variance:")
    objectives = np.array([
        [0.5, 0.5],
        [0.5 + 1e-12, 0.5],
        [0.5 + 2e-12, 0.5],
        [0.5 + 3e-12, 0.5],
    ])
    pfs = pareto_front_smoothness(objectives)
    print(f"   Objectives:\n{objectives}")
    print(f"   PFS: {pfs:.6f} ✓ (should be 0.0, variance too small)")
    assert pfs == 0.0, f"Expected 0.0, got {pfs}"

    # Case 5: Valid case (should compute normally)
    print("\n5. Valid case with proper spread:")
    objectives = np.array([
        [0.1, 0.9],
        [0.3, 0.7],
        [0.5, 0.5],
        [0.7, 0.3],
        [0.9, 0.1],
    ])
    pfs = pareto_front_smoothness(objectives)
    print(f"   Objectives:\n{objectives}")
    print(f"   PFS: {pfs:.6f} ✓ (should be > 0, computed successfully)")
    # Should compute successfully (any value is OK)
    assert not np.isnan(pfs), f"Got NaN for valid case"
    assert not np.isinf(pfs), f"Got Inf for valid case"

    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED - PFS handles degenerate cases correctly")
    print("="*80 + "\n")


if __name__ == '__main__':
    try:
        test_degenerate_cases()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
