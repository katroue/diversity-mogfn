#!/usr/bin/env python3
"""
Test script for sampling ablation study.

This script tests the sampling ablation fixes by running a subset of experiments
with reduced iterations to verify that different configurations produce different results.

Usage:
    python tests/test_sampling_ablation.py
"""
import sys
import yaml
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_ablation_study import run_single_experiment, load_config


def create_test_config():
    """Create a test configuration with reduced iterations."""
    # Load full config
    config_path = project_root / 'configs' / 'ablations' / 'sampling_ablation.yaml'
    config = load_config(str(config_path))

    # Modify for testing
    config['fixed']['num_iterations'] = 500  # Reduced from 4000
    config['fixed']['num_samples'] = 200  # Reduced from 1000
    config['fixed']['eval_every'] = 100  # More frequent evaluation
    config['fixed']['batch_size'] = 64  # Smaller batch for speed
    config['seeds'] = [42]  # Single seed for testing

    # Select a subset of experiments to test (one from each category)
    test_experiments = [
        # 1. Temperature test
        {
            'name': 'temp_low',
            'temperature': 0.5,
            'sampling_strategy': 'categorical',
            'description': 'Low temperature - exploit mode'
        },
        {
            'name': 'temp_high',
            'temperature': 2.0,
            'sampling_strategy': 'categorical',
            'description': 'High temperature - explore mode'
        },
        # 2. Sampling strategy test
        {
            'name': 'greedy',
            'temperature': 1.0,
            'sampling_strategy': 'greedy',
            'description': 'Greedy sampling - deterministic'
        },
        {
            'name': 'top_k',
            'temperature': 1.0,
            'sampling_strategy': 'top_k',
            'top_k': 3,
            'description': 'Top-K sampling with K=3'
        },
        # 3. Off-policy test
        {
            'name': 'on_policy_pure',
            'off_policy_ratio': 0.0,
            'temperature': 1.0,
            'sampling_strategy': 'categorical',
            'description': 'Pure on-policy training'
        },
        {
            'name': 'off_policy_25',
            'off_policy_ratio': 0.25,
            'temperature': 1.0,
            'sampling_strategy': 'categorical',
            'description': '25% off-policy exploration'
        },
        # 4. Preference distribution test
        {
            'name': 'pref_uniform',
            'preference_distribution': 'uniform',
            'temperature': 1.0,
            'sampling_strategy': 'categorical',
            'description': 'Uniform preference distribution'
        },
        {
            'name': 'pref_dirichlet',
            'preference_distribution': 'dirichlet',
            'dirichlet_alpha': 0.5,
            'temperature': 1.0,
            'sampling_strategy': 'categorical',
            'description': 'Dirichlet alpha=0.5 (corner preferences)'
        },
    ]

    config['experiments'] = test_experiments

    return config


def run_tests():
    """Run sampling ablation tests."""
    print("="*70)
    print("SAMPLING ABLATION TEST")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create test config
    print("1. Creating test configuration...")
    config = create_test_config()
    print(f"   ✓ Testing {len(config['experiments'])} experiments with {config['fixed']['num_iterations']} iterations")
    print(f"   ✓ Using seed: {config['seeds'][0]}")

    # Create temporary output directory
    temp_dir = tempfile.mkdtemp(prefix='sampling_ablation_test_')
    output_dir = Path(temp_dir)
    print(f"   ✓ Output directory: {output_dir}")
    print()

    try:
        # Run experiments
        print("2. Running experiments...")
        results = []

        for i, exp_config in enumerate(config['experiments'], 1):
            print(f"\n   [{i}/{len(config['experiments'])}] Testing: {exp_config['name']}")
            print(f"       Description: {exp_config.get('description', 'N/A')}")

            # Run experiment
            result = run_single_experiment(
                exp_config=exp_config,
                fixed_config=config['fixed'],
                seed=config['seeds'][0],
                output_dir=output_dir,
                device='cpu'
            )

            results.append({
                'name': exp_config['name'],
                'hypervolume': result.get('hypervolume', 0),
                'tds': result.get('tds', 0),
                'pas': result.get('pas', 0),
                'final_loss': result.get('final_loss', 0),
                'config': exp_config
            })

            print(f"       ✓ HV: {result.get('hypervolume', 0):.4f}, TDS: {result.get('tds', 0):.4f}, PAS: {result.get('pas', 0):.4f}")

        print()
        print("="*70)
        print("3. VALIDATING RESULTS")
        print("="*70)

        # Validate that different configurations produce different results
        validation_passed = True

        # Test 1: Temperature effect (temp_low vs temp_high)
        print("\n✓ Test 1: Temperature Effect")
        temp_low = next(r for r in results if r['name'] == 'temp_low')
        temp_high = next(r for r in results if r['name'] == 'temp_high')

        hv_diff = abs(temp_low['hypervolume'] - temp_high['hypervolume'])
        tds_diff = abs(temp_low['tds'] - temp_high['tds'])

        print(f"  temp_low  (T=0.5): HV={temp_low['hypervolume']:.4f}, TDS={temp_low['tds']:.4f}")
        print(f"  temp_high (T=2.0): HV={temp_high['hypervolume']:.4f}, TDS={temp_high['tds']:.4f}")
        print(f"  Difference: HV={hv_diff:.4f}, TDS={tds_diff:.4f}")

        if hv_diff > 0.001 or tds_diff > 0.001:
            print("  ✓ PASS: Temperature produces different results")
        else:
            print("  ✗ FAIL: Temperature does not affect results (values identical)")
            validation_passed = False

        # Test 2: Sampling strategy effect (greedy vs top_k)
        print("\n✓ Test 2: Sampling Strategy Effect")
        greedy = next(r for r in results if r['name'] == 'greedy')
        top_k = next(r for r in results if r['name'] == 'top_k')

        hv_diff = abs(greedy['hypervolume'] - top_k['hypervolume'])
        tds_diff = abs(greedy['tds'] - top_k['tds'])

        print(f"  greedy: HV={greedy['hypervolume']:.4f}, TDS={greedy['tds']:.4f}")
        print(f"  top_k:  HV={top_k['hypervolume']:.4f}, TDS={top_k['tds']:.4f}")
        print(f"  Difference: HV={hv_diff:.4f}, TDS={tds_diff:.4f}")

        if hv_diff > 0.001 or tds_diff > 0.001:
            print("  ✓ PASS: Sampling strategy produces different results")
        else:
            print("  ✗ FAIL: Sampling strategy does not affect results")
            validation_passed = False

        # Test 3: Off-policy effect
        print("\n✓ Test 3: Off-Policy Ratio Effect")
        on_policy = next(r for r in results if r['name'] == 'on_policy_pure')
        off_policy = next(r for r in results if r['name'] == 'off_policy_25')

        hv_diff = abs(on_policy['hypervolume'] - off_policy['hypervolume'])
        tds_diff = abs(on_policy['tds'] - off_policy['tds'])

        print(f"  on_policy  (0%):  HV={on_policy['hypervolume']:.4f}, TDS={on_policy['tds']:.4f}")
        print(f"  off_policy (25%): HV={off_policy['hypervolume']:.4f}, TDS={off_policy['tds']:.4f}")
        print(f"  Difference: HV={hv_diff:.4f}, TDS={tds_diff:.4f}")

        if hv_diff > 0.001 or tds_diff > 0.001:
            print("  ✓ PASS: Off-policy ratio produces different results")
        else:
            print("  ✗ FAIL: Off-policy ratio does not affect results")
            validation_passed = False

        # Test 4: Preference distribution effect
        print("\n✓ Test 4: Preference Distribution Effect")
        uniform = next(r for r in results if r['name'] == 'pref_uniform')
        dirichlet = next(r for r in results if r['name'] == 'pref_dirichlet')

        hv_diff = abs(uniform['hypervolume'] - dirichlet['hypervolume'])
        pas_diff = abs(uniform['pas'] - dirichlet['pas'])

        print(f"  uniform:   HV={uniform['hypervolume']:.4f}, PAS={uniform['pas']:.4f}")
        print(f"  dirichlet: HV={dirichlet['hypervolume']:.4f}, PAS={dirichlet['pas']:.4f}")
        print(f"  Difference: HV={hv_diff:.4f}, PAS={pas_diff:.4f}")

        if hv_diff > 0.001 or pas_diff > 0.001:
            print("  ✓ PASS: Preference distribution produces different results")
        else:
            print("  ✗ FAIL: Preference distribution does not affect results")
            validation_passed = False

        # Summary
        print()
        print("="*70)
        print("TEST SUMMARY")
        print("="*70)

        if validation_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("\nAll sampling ablation fixes are working correctly:")
            print("  ✓ Temperature parameter affects exploration")
            print("  ✓ Sampling strategies produce different behaviors")
            print("  ✓ Off-policy ratio affects sampling")
            print("  ✓ Preference distribution affects results")
            return_code = 0
        else:
            print("\n⚠️  SOME TESTS FAILED")
            print("\nPlease review the failed tests above.")
            return_code = 1

        print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test output saved to: {output_dir}")
        print()

        return return_code

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        # Optional: Clean up temp directory
        # Uncomment to automatically delete test results
        # shutil.rmtree(temp_dir, ignore_errors=True)
        pass


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
