#!/usr/bin/env python3
"""
Test script for loss ablation entropy regularization.

Tests that different entropy regularization beta values produce different results,
validating the fix for the bug where all configurations returned identical metrics.

Usage:
    python tests/ablation_test/test_loss_ablation_entropy.py
"""

import sys
import json
import yaml
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ablations.run_loss_ablation_group import generate_group_configs, run_group


def create_test_config() -> dict:
    """Create a minimal test configuration for entropy ablation."""
    return {
        'experiment_name': 'test_loss_ablation',
        'study_type': 'ablation',

        'fixed': {
            'task': 'hypergrid',
            'grid_size': [8, 8],  # Smaller grid for faster testing
            'capacity': 'small',
            'arch_type': 'concat',
            'hidden_dim': 32,  # Smaller for faster training
            'num_layers': 2,
            'activation': 'relu',
            'preference_distribution': 'dirichlet',
            'temperature': 2.0,
            'sampling_strategy': 'categorical',
            'dirichlet_alpha': 1.5,
            'num_preferences_per_batch': 8,
            'sampling_schedule': 'uniform',
            'num_iterations': 1000,  # Short training for testing
            'batch_size': 64,  # Smaller batch
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'lr_schedule': 'constant',
            'gradient_clip': 10.0,
            'eval_every': 500,
            'eval_samples': 200,  # Fewer samples for testing
            'final_eval_samples': 500,
            'num_seeds': 2,  # Just 2 seeds for testing
            'base_seed': 42,
        },

        'ablation_factors': {
            'base_loss': {
                'description': 'Core GFlowNet training objective',
                'options': [
                    {
                        'name': 'trajectory_balance',
                        'label': 'TB',
                        'type': 'trajectory_balance',
                        'params': {'log_reward_clip': 10.0}
                    },
                    {
                        'name': 'subtrajectory_balance_09',
                        'label': 'SubTB(λ=0.9)',
                        'type': 'subtrajectory_balance',
                        'params': {'lambda_': 0.9, 'log_reward_clip': 10.0}
                    }
                ]
            },
            'regularization': {
                'description': 'Regularization terms',
                'options': [
                    {
                        'name': 'none',
                        'label': 'No Reg',
                        'type': 'none',
                        'params': {}
                    },
                    {
                        'name': 'entropy_001',
                        'label': 'Entropy(β=0.01)',
                        'type': 'entropy',
                        'params': {'beta': 0.01, 'entropy_type': 'policy'}
                    },
                    {
                        'name': 'entropy_01',
                        'label': 'Entropy(β=0.1)',
                        'type': 'entropy',
                        'params': {'beta': 0.1, 'entropy_type': 'policy'}
                    },
                    {
                        'name': 'entropy_05',
                        'label': 'Entropy(β=0.5)',
                        'type': 'entropy',
                        'params': {'beta': 0.5, 'entropy_type': 'policy'}
                    }
                ]
            },
            'modifications': {
                'description': 'Loss modifications',
                'options': [
                    {
                        'name': 'standard',
                        'label': 'Standard',
                        'type': 'none',
                        'params': {}
                    }
                ]
            }
        },

        'experiments': [
            {
                'group': 'test_entropy_regularization',
                'description': 'Test entropy regularization with different β values',
                'fixed': {
                    'base_loss': 'subtrajectory_balance_09',
                    'modifications': 'standard'
                },
                'vary': {
                    'regularization': [
                        'none',
                        'entropy_001',
                        'entropy_01',
                        'entropy_05'
                    ]
                }
            }
        ]
    }


def test_config_generation():
    """Test that configs are generated correctly with all required keys."""
    print("\n" + "="*80)
    print("TEST 1: Configuration Generation")
    print("="*80)

    config = create_test_config()
    group_name = 'test_entropy_regularization'

    configs = generate_group_configs(config, group_name)

    print(f"\nGenerated {len(configs)} configurations")
    expected_configs = 4  # 4 regularization options
    expected_total = expected_configs * 2  # 2 seeds

    assert len(configs) == expected_total, \
        f"Expected {expected_total} configs (4 options × 2 seeds), got {len(configs)}"
    print(f"✓ Correct number of configs: {len(configs)}")

    # Check first config has all required keys
    first_config = configs[0]
    required_keys = [
        'name', 'group', 'seed',
        'regularization', 'regularization_type', 'regularization_params',
        'base_loss', 'base_loss_type', 'base_loss_params',
        'modifications', 'modifications_type', 'modifications_params'
    ]

    missing_keys = [k for k in required_keys if k not in first_config]
    if missing_keys:
        print(f"✗ Missing keys in config: {missing_keys}")
        print(f"  Available keys: {list(first_config.keys())}")
        raise AssertionError(f"Missing required keys: {missing_keys}")

    print(f"✓ All required keys present in configs")

    # Verify different beta values
    beta_values = set()
    for cfg in configs:
        if cfg['regularization_type'] == 'entropy':
            beta = cfg['regularization_params'].get('beta')
            beta_values.add(beta)

    print(f"✓ Found {len(beta_values)} different beta values: {sorted(beta_values)}")

    # Verify base loss is correctly configured
    for cfg in configs:
        assert cfg['base_loss_type'] == 'subtrajectory_balance', \
            f"Expected base_loss_type='subtrajectory_balance', got '{cfg['base_loss_type']}'"
        assert cfg['base_loss_params']['lambda_'] == 0.9, \
            f"Expected lambda_=0.9, got {cfg['base_loss_params'].get('lambda_')}"

    print(f"✓ Base loss correctly configured (SubTB λ=0.9)")

    print("\n✓ TEST 1 PASSED: Configuration generation works correctly\n")
    return config


def test_experiments_produce_different_results():
    """Test that experiments with different beta values produce different results."""
    print("\n" + "="*80)
    print("TEST 2: Different Beta Values Produce Different Results")
    print("="*80)

    config = create_test_config()

    # Create temporary output directory
    output_dir = Path('results/test_entropy_ablation_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning experiments in: {output_dir}")
    print("This will take a few minutes...\n")

    try:
        # Run the experiment group
        run_group(
            config=config,
            group_name='test_entropy_regularization',
            output_dir=output_dir,
            resume=False,
            dry_run=False,
            device='cpu'
        )

        # Load results
        results_file = output_dir / 'test_entropy_regularization' / 'results.csv'

        if not results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        import pandas as pd
        df = pd.read_csv(results_file)

        print("\n" + "-"*80)
        print("RESULTS SUMMARY")
        print("-"*80)

        # Group by configuration (ignore seed)
        df['config_name'] = df['exp_name'].str.replace(r'_seed\d+', '', regex=True)

        # Check key metrics across configurations
        metrics_to_check = ['hypervolume', 'mce', 'tds', 'qds']

        print("\nMetrics by configuration:")
        for metric in metrics_to_check:
            if metric not in df.columns:
                print(f"  Warning: Metric '{metric}' not found in results")
                continue

            print(f"\n  {metric.upper()}:")
            grouped = df.groupby('config_name')[metric].agg(['mean', 'std', 'count'])
            for config_name, row in grouped.iterrows():
                print(f"    {config_name:20s}: {row['mean']:8.4f} ± {row['std']:7.4f} (n={int(row['count'])})")

        # Check training history for regularization terms
        print("\n" + "-"*80)
        print("REGULARIZATION TERM VALIDATION")
        print("-"*80)

        group_dir = output_dir / 'test_entropy_regularization'
        reg_terms = {}

        for exp_dir in group_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            history_file = exp_dir / 'training_history.json'
            if not history_file.exists():
                continue

            with open(history_file, 'r') as f:
                history = json.load(f)

            exp_name = exp_dir.name
            config_name = exp_name.replace('_seed42', '').replace('_seed153', '')

            if 'reg_term' in history:
                reg_values = history['reg_term']
                # Take absolute values since entropy reg is negative
                abs_reg_values = [abs(x) for x in reg_values]
                mean_reg = np.mean(abs_reg_values)
                max_reg = np.max(abs_reg_values)

                if config_name not in reg_terms:
                    reg_terms[config_name] = []
                reg_terms[config_name].append(mean_reg)

        print("\nRegularization term statistics:")
        for config_name in sorted(reg_terms.keys()):
            values = reg_terms[config_name]
            mean_val = np.mean(values)
            print(f"  {config_name:20s}: mean={mean_val:10.6f}")

            if 'none' in config_name:
                if mean_val > 1e-6:
                    print(f"    ✗ WARNING: 'none' config should have ~0 regularization, got {mean_val}")
                else:
                    print(f"    ✓ Correctly zero for 'none' config")
            else:
                if mean_val < 1e-6:
                    print(f"    ✗ ERROR: Entropy config has zero regularization!")
                    raise AssertionError(f"Entropy regularization not applied for {config_name}")
                else:
                    print(f"    ✓ Non-zero regularization applied")

        # Check that different beta values produce different results
        print("\n" + "-"*80)
        print("UNIQUENESS CHECK")
        print("-"*80)

        # Get unique configurations
        unique_configs = df['config_name'].unique()
        print(f"\nFound {len(unique_configs)} unique configurations: {list(unique_configs)}")

        # For each metric, check if all values are identical (within floating point tolerance)
        all_identical = True
        for metric in metrics_to_check:
            if metric not in df.columns:
                continue

            grouped_means = df.groupby('config_name')[metric].mean()

            # Check if all means are identical (within 1e-6 tolerance)
            if len(grouped_means) > 1:
                variance = grouped_means.std()
                if variance < 1e-6:
                    print(f"  ✗ {metric}: All values identical (std={variance:.2e})")
                    all_identical = True
                else:
                    print(f"  ✓ {metric}: Values differ (std={variance:.6f})")
                    all_identical = False

        if all_identical:
            print("\n✗ ERROR: All configurations produced identical results!")
            print("  This suggests the entropy regularization is not being applied.")
            raise AssertionError("Different beta values produced identical results")

        print("\n✓ TEST 2 PASSED: Different beta values produce different results\n")

        return True

    finally:
        # Clean up test directory
        print(f"\nCleaning up test directory: {output_dir}")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        print("✓ Cleanup complete")


def test_config_files():
    """Test that generated config files have correct structure."""
    print("\n" + "="*80)
    print("TEST 3: Generated Config Files Structure")
    print("="*80)

    config = create_test_config()

    # Create temporary output directory
    output_dir = Path('results/test_config_check_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate configs and save them (without running)
        configs = generate_group_configs(config, 'test_entropy_regularization')

        group_dir = output_dir / 'test_entropy_regularization'
        group_dir.mkdir(parents=True, exist_ok=True)

        # Save a few sample configs
        for i, exp_config in enumerate(configs[:4]):  # Just check first 4
            exp_config_full = {**config['fixed'], **exp_config}
            exp_name = f"{exp_config['name']}_seed{exp_config['seed']}"
            config_file = group_dir / f'{exp_name}_config.json'

            with open(config_file, 'w') as f:
                json.dump(exp_config_full, f, indent=2)

            print(f"\nChecking config {i+1}: {exp_name}")

            # Verify required keys
            required_trainer_keys = [
                'base_loss_type', 'base_loss_params',
                'regularization_type', 'regularization_params'
            ]

            for key in required_trainer_keys:
                if key not in exp_config_full:
                    print(f"  ✗ Missing key: {key}")
                    raise AssertionError(f"Config missing required key: {key}")
                else:
                    print(f"  ✓ {key}: {exp_config_full[key]}")

            # Verify types
            assert exp_config_full['base_loss_type'] == 'subtrajectory_balance', \
                "Base loss type incorrect"

            if exp_config['name'] != 'none':
                assert exp_config_full['regularization_type'] == 'entropy', \
                    f"Expected entropy, got {exp_config_full['regularization_type']}"
                assert 'beta' in exp_config_full['regularization_params'], \
                    "Beta parameter missing"
                print(f"  ✓ Regularization beta: {exp_config_full['regularization_params']['beta']}")

        print("\n✓ TEST 3 PASSED: Config files have correct structure\n")

    finally:
        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("ENTROPY REGULARIZATION TEST SUITE")
    print("="*80)
    print("\nThis test validates that the fix for run_loss_ablation_group.py works correctly.")
    print("It tests that different entropy beta values produce different results.\n")

    all_passed = True

    try:
        # Test 1: Config generation
        test_config_generation()

        # Test 3: Config file structure
        test_config_files()

        # Test 2: Full experiment (this takes the longest)
        test_experiments_produce_different_results()

        print("\n" + "="*80)
        print("✓ ALL TESTS PASSED")
        print("="*80)
        print("\nThe entropy regularization is working correctly:")
        print("  • Configs are generated with all required keys")
        print("  • Base loss parameters are correctly propagated")
        print("  • Different beta values produce different results")
        print("  • Regularization terms are non-zero when expected")
        print("\n")

    except AssertionError as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}\n")
        all_passed = False
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST ERROR")
        print("="*80)
        print(f"\nUnexpected error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        all_passed = False
        sys.exit(1)


if __name__ == '__main__':
    main()
