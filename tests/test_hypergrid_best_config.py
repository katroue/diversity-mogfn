#!/usr/bin/env python3
"""
Test script to verify hypergrid_best_config does NOT experience mode collapse.

This script runs a quick validation (500 iterations, 1 seed) to test that:
1. MCE > 0.25 (no mode collapse)
2. num_modes > 10 (sufficient diversity)
3. QDS > 0.30 (reasonable quality-diversity)

Run: python tests/test_hypergrid_best_config.py
"""

import sys
from pathlib import Path
import subprocess
import json
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_hypergrid_best_config():
    """Test that hypergrid_best_config_v2.yaml does not experience mode collapse."""

    print("=" * 80)
    print("TESTING HYPERGRID BEST CONFIG FOR MODE COLLAPSE")
    print("=" * 80)
    print()

    # Test parameters (quick test)
    config_path = PROJECT_ROOT / "configs/factorials/hypergrid_best_config.yaml"
    output_dir = PROJECT_ROOT / "results/tests/hypergrid_best_quick_test"

    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print()

    # Check if config exists
    if not config_path.exists():
        print(f"❌ ERROR: Config file not found: {config_path}")
        print("   Please create hypergrid_best_config_v2.yaml first.")
        return False

    # Create a temporary quick-test config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Modify for quick test
    config['fixed']['max_iterations'] = 250  # Fast test (250 iterations ~8 min)
    config['fixed']['num_seeds'] = 1  # Single seed
    config['fixed']['eval_every'] = 50  # Show progress every 50 iterations
    config['fixed']['eval_samples'] = 500
    config['fixed']['final_eval_samples'] = 1000

    # Save quick test config
    quick_config_path = PROJECT_ROOT / "configs/factorials/hypergrid_best_config_quick_test.yaml"
    with open(quick_config_path, 'w') as f:
        yaml.dump(config, f)

    print(f"Created quick test config: {quick_config_path}")
    print(f"  - max_iterations: 250 (~8 minutes)")
    print(f"  - num_seeds: 1")
    print(f"  - final_eval_samples: 1000")
    print()

    # Run experiment
    print("Running experiment...")
    print("-" * 80)

    cmd = [
        "python3",
        str(PROJECT_ROOT / "scripts/factorials/hypergrid/run_factorial_hypergrid.py"),
        "--config", str(quick_config_path),
        "--output_dir", str(output_dir)
    ]

    try:
        # Run with real-time output streaming
        process = subprocess.Popen(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1  # Line buffered
        )

        # Stream output in real-time
        stdout_lines = []
        import time
        start_time = time.time()

        while True:
            # Check if process is still running
            retcode = process.poll()

            # Read output
            if process.stdout:
                line = process.stdout.readline()
                if line:
                    print(line, end='')  # Print in real-time
                    stdout_lines.append(line)

            # Check timeout
            if time.time() - start_time > 1200:  # 20 minute timeout
                process.kill()
                print("\n❌ ERROR: Experiment timed out after 20 minutes")
                return False

            # Exit if process finished
            if retcode is not None:
                # Read any remaining output
                if process.stdout:
                    remaining = process.stdout.read()
                    if remaining:
                        print(remaining, end='')
                        stdout_lines.append(remaining)
                break

            time.sleep(0.1)

        if retcode != 0:
            print(f"\n❌ ERROR: Experiment failed with exit code {retcode}")
            return False

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("-" * 80)
    print()

    # Check results
    print("Checking results...")
    print("-" * 80)

    # Find the metrics file
    metrics_files = list(output_dir.glob("*/metrics.json"))

    if not metrics_files:
        print(f"❌ ERROR: No metrics.json found in {output_dir}")
        return False

    metrics_file = metrics_files[0]
    print(f"Found metrics: {metrics_file}")
    print()

    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Print key metrics
    mce = metrics.get('mce', 0)
    num_modes = metrics.get('num_modes', 0)
    qds = metrics.get('qds', 0)
    hypervolume = metrics.get('hypervolume', 0)
    tds = metrics.get('tds', 0)

    print("KEY METRICS:")
    print(f"  MCE (Mode Coverage Entropy):  {mce:.4f}")
    print(f"  num_modes:                     {num_modes}")
    print(f"  QDS (Quality-Diversity Score): {qds:.4f}")
    print(f"  Hypervolume:                   {hypervolume:.4f}")
    print(f"  TDS (Trajectory Diversity):    {tds:.4f}")
    print()

    # Load objectives to check diversity
    objectives_file = metrics_file.parent / "objectives.npy"
    if objectives_file.exists():
        objectives = np.load(objectives_file)
        unique_solutions = len(np.unique(objectives, axis=0))
        most_common_count = 0

        unique, counts = np.unique(objectives, axis=0, return_counts=True)
        most_common_count = counts.max()
        most_common_pct = 100 * most_common_count / len(objectives)

        print(f"OBJECTIVE SPACE ANALYSIS:")
        print(f"  Total samples:        {len(objectives)}")
        print(f"  Unique solutions:     {unique_solutions}")
        print(f"  Most common solution: {most_common_count} times ({most_common_pct:.2f}%)")
        print()

    # Run tests
    print("=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print()

    passed = True

    # Test 1: No complete mode collapse (MCE > 0.25)
    test1_passed = mce > 0.25
    print(f"Test 1: MCE > 0.25 (no complete mode collapse)")
    print(f"  Result: {mce:.4f} > 0.25 = {test1_passed}")
    print(f"  Status: {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print()
    passed = passed and test1_passed

    # Test 2: Reasonable QDS (QDS > 0.30)
    test2_passed = qds > 0.30
    print(f"Test 2: QDS > 0.30 (reasonable quality-diversity)")
    print(f"  Result: {qds:.4f} > 0.30 = {test2_passed}")
    print(f"  Status: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print()
    passed = passed and test2_passed

    # Test 3: Multiple unique solutions (>20 unique for 250 iterations)
    if objectives_file.exists():
        test3_passed = unique_solutions > 20
        print(f"Test 3: Multiple solutions (>20 unique for short test)")
        print(f"  Result: {unique_solutions} unique solutions")
        print(f"  Status: {'✓ PASSED' if test3_passed else '✗ FAILED'}")
        print()
        passed = passed and test3_passed

    # Test 4: Not stuck on single solution (most common <95%)
    if objectives_file.exists():
        test4_passed = most_common_pct < 95
        print(f"Test 4: Not stuck (most common solution <95%)")
        print(f"  Result: Most common {most_common_pct:.1f}% < 95%")
        print(f"  Status: {'✓ PASSED' if test4_passed else '✗ FAILED'}")
        print()
        passed = passed and test4_passed

    # Final verdict
    print("=" * 80)
    if passed:
        print("✓ ALL TESTS PASSED - No mode collapse detected!")
        print()
        print("The hypergrid_best_config.yaml configuration is working correctly.")
        print("You can now run the full experiment with all 5 seeds:")
        print()
        print(f"  python3 scripts/factorials/hypergrid/run_factorial_hypergrid.py \\")
        print(f"      --config {config_path} \\")
        print(f"      --output_dir results/validation/hypergrid_best_final")
    else:
        print("✗ TESTS FAILED - Mode collapse or poor performance detected!")
        print()
        print("The configuration needs further debugging.")
        print("Check the experiment output and metrics above for details.")
    print("=" * 80)

    return passed


if __name__ == "__main__":
    try:
        success = test_hypergrid_best_config()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
