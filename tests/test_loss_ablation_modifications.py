#!/usr/bin/env python3
"""
Test script for loss ablation modifications group.

This test checks if the modifications parameter is properly handled in the
loss ablation pipeline, including config generation and trainer integration.

NOTE: As of this test creation, modifications are NOT YET IMPLEMENTED in the
trainer. This test documents what needs to be implemented.

Usage:
    python tests/test_loss_ablation_modifications.py
"""

import sys
import json
import yaml
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
import inspect

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.ablations.run_loss_ablation_group import generate_group_configs


def test_config_generation():
    """Test that modification configs are generated correctly."""
    print("\n" + "="*80)
    print("TEST 1: Configuration Generation for Loss Modifications")
    print("="*80)

    # Load the actual loss ablation config
    config_path = project_root / 'configs/ablations/loss_ablation.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    group_name = 'loss_modifications'

    configs = generate_group_configs(config, group_name)

    print(f"\nGenerated {len(configs)} configurations")

    # Get expected values from config
    num_seeds = config['fixed'].get('num_seeds', 5)
    group_config = None
    for exp_group in config['experiments']:
        if exp_group['group'] == group_name:
            group_config = exp_group
            break

    if not group_config:
        raise ValueError(f"Group '{group_name}' not found in config")

    num_modifications = len(group_config['vary']['modifications'])
    expected_total = num_modifications * num_seeds

    assert len(configs) == expected_total, \
        f"Expected {expected_total} configs ({num_modifications} mods × {num_seeds} seeds), got {len(configs)}"
    print(f"✓ Correct number of configs: {len(configs)}")

    # Check first config has all required keys
    first_config = configs[0]
    required_keys = [
        'name', 'group', 'seed',
        'modifications', 'modifications_type', 'modifications_params',
        'base_loss', 'base_loss_type', 'base_loss_params',
        'regularization', 'regularization_type', 'regularization_params'
    ]

    missing_keys = [k for k in required_keys if k not in first_config]
    if missing_keys:
        print(f"✗ Missing keys in config: {missing_keys}")
        print(f"  Available keys: {list(first_config.keys())}")
        raise AssertionError(f"Missing required keys: {missing_keys}")

    print(f"✓ All required keys present in configs")

    # Verify different modification types
    mod_types = set()
    mod_configs = {}

    for cfg in configs:
        mod_name = cfg['modifications']
        mod_type = cfg['modifications_type']
        mod_params = cfg['modifications_params']

        mod_types.add(mod_type)

        if mod_name not in mod_configs:
            mod_configs[mod_name] = {
                'type': mod_type,
                'params': mod_params
            }

    print(f"\n✓ Found {len(mod_types)} different modification types:")
    for mod_name, mod_info in sorted(mod_configs.items()):
        print(f"  • {mod_name:30s}: type={mod_info['type']:20s} params={mod_info['params']}")

    # Verify expected modification types
    expected_mods = group_config['vary']['modifications']
    for expected_mod in expected_mods:
        if expected_mod not in mod_configs:
            raise AssertionError(f"Expected modification '{expected_mod}' not found in generated configs")

    print(f"\n✓ All expected modifications present in configs")

    # Verify fixed parameters are correct
    expected_base_loss = group_config['fixed']['base_loss']
    expected_reg = group_config['fixed']['regularization']

    for cfg in configs:
        # Check base_loss
        if cfg['base_loss'] != expected_base_loss:
            raise AssertionError(
                f"Expected base_loss='{expected_base_loss}', got '{cfg['base_loss']}'"
            )

        # Check regularization
        if cfg['regularization'] != expected_reg:
            raise AssertionError(
                f"Expected regularization='{expected_reg}', got '{cfg['regularization']}'"
            )

    print(f"✓ Fixed parameters correctly configured:")
    print(f"  • base_loss: {expected_base_loss}")
    print(f"  • regularization: {expected_reg}")

    print("\n✓ TEST 1 PASSED: Configuration generation works correctly\n")
    return config, configs


def test_trainer_support():
    """Test if MOGFNTrainer supports modifications."""
    print("\n" + "="*80)
    print("TEST 2: Trainer Support for Modifications")
    print("="*80)

    try:
        from src.models.mogfn_pc import MOGFNTrainer

        # Check MOGFNTrainer __init__ signature
        init_signature = inspect.signature(MOGFNTrainer.__init__)
        params = list(init_signature.parameters.keys())

        print(f"\nMOGFNTrainer.__init__ parameters:")
        for param in params:
            param_obj = init_signature.parameters[param]
            if param == 'self':
                continue
            default = param_obj.default
            if default == inspect.Parameter.empty:
                print(f"  • {param:25s} (required)")
            else:
                print(f"  • {param:25s} = {default}")

        # Check if modifications-related parameters exist
        modifications_params = [p for p in params if 'modif' in p.lower()]

        if not modifications_params:
            print("\n" + "!"*80)
            print("⚠ WARNING: MOGFNTrainer does NOT support modifications!")
            print("!"*80)
            print("\nThe trainer is missing the following parameters:")
            print("  • modifications: str = 'none'")
            print("  • modifications_params: Optional[Dict] = None")
            print("\nThis means modifications configs will be generated but NOT USED!")
            print("\n" + "!"*80)
            return False
        else:
            print(f"\n✓ Found modifications parameters: {modifications_params}")
            return True

    except Exception as e:
        print(f"✗ ERROR: {str(e)}")
        raise


def test_run_ablation_study_support():
    """Test if run_ablation_study.py passes modifications to trainer."""
    print("\n" + "="*80)
    print("TEST 3: run_ablation_study.py Support for Modifications")
    print("="*80)

    script_path = project_root / 'scripts/ablations/run_ablation_study.py'

    with open(script_path, 'r') as f:
        script_content = f.read()

    # Check if modifications is passed to MOGFNTrainer
    print("\nChecking if run_ablation_study.py passes modifications to trainer...")

    if 'modifications_type' in script_content or 'modifications_params' in script_content:
        print("✓ Script references modifications_type or modifications_params")
        return True
    else:
        print("\n" + "!"*80)
        print("⚠ WARNING: run_ablation_study.py does NOT pass modifications to trainer!")
        print("!"*80)
        print("\nThe script needs to be updated to pass:")
        print("  modifications=config.get('modifications_type', 'none'),")
        print("  modifications_params=config.get('modifications_params', {}),")
        print("\nto the MOGFNTrainer constructor.")
        print("\n" + "!"*80)
        return False


def test_config_file_structure():
    """Test that generated config files have correct structure."""
    print("\n" + "="*80)
    print("TEST 4: Generated Config Files Structure")
    print("="*80)

    # Load the actual loss ablation config
    config_path = project_root / 'configs/ablations/loss_ablation.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create temporary output directory
    output_dir = Path('results/test_modifications_check_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate configs and save them
        configs = generate_group_configs(config, 'loss_modifications')

        group_dir = output_dir / 'loss_modifications'
        group_dir.mkdir(parents=True, exist_ok=True)

        # Save a sample from each modification type
        saved_configs = {}
        for exp_config in configs:
            mod_name = exp_config['modifications']
            if mod_name not in saved_configs:
                exp_config_full = {**config['fixed'], **exp_config}
                exp_name = f"{exp_config['name']}_seed{exp_config['seed']}"
                config_file = group_dir / f'{exp_name}_config.json'

                with open(config_file, 'w') as f:
                    json.dump(exp_config_full, f, indent=2)

                saved_configs[mod_name] = exp_config_full

        print(f"\nSaved {len(saved_configs)} sample configs (one per modification type)")

        for mod_name, exp_config_full in saved_configs.items():
            print(f"\nChecking config: {mod_name}")

            # Verify required keys
            required_trainer_keys = [
                'base_loss_type', 'base_loss_params',
                'regularization_type', 'regularization_params',
                'modifications_type', 'modifications_params'
            ]

            for key in required_trainer_keys:
                if key not in exp_config_full:
                    print(f"  ✗ Missing key: {key}")
                    raise AssertionError(f"Config missing required key: {key}")
                else:
                    value = exp_config_full[key]
                    if isinstance(value, dict):
                        print(f"  ✓ {key}: {value}")
                    else:
                        print(f"  ✓ {key}: {value}")

        print("\n✓ TEST 4 PASSED: Config files have correct structure\n")

    finally:
        # Clean up
        if output_dir.exists():
            shutil.rmtree(output_dir)


def check_modification_implementations():
    """Check what needs to be implemented for modifications support."""
    print("\n" + "="*80)
    print("IMPLEMENTATION STATUS: Loss Modifications")
    print("="*80)

    print("\nModifications defined in config:")
    print("  1. temperature_scaling - Scale logits before softmax")
    print("  2. reward_shaping - Add diversity bonus to rewards")

    print("\n" + "-"*80)
    print("WHAT'S IMPLEMENTED:")
    print("-"*80)
    print("  ✓ Config generation (run_loss_ablation_group.py)")
    print("  ✓ modifications_type and modifications_params created correctly")
    print("  ✓ Fixed parameters (base_loss, regularization) handled properly")

    print("\n" + "-"*80)
    print("WHAT'S MISSING:")
    print("-"*80)

    # Check trainer
    from src.models.mogfn_pc import MOGFNTrainer
    init_signature = inspect.signature(MOGFNTrainer.__init__)
    has_modifications = 'modifications' in init_signature.parameters

    if not has_modifications:
        print("  ✗ MOGFNTrainer needs to accept:")
        print("      modifications: str = 'none'")
        print("      modifications_params: Optional[Dict] = None")
        print("\n  ✗ MOGFNTrainer needs methods:")
        print("      apply_temperature_scaling(logits, params)")
        print("      apply_reward_shaping(reward, state, params)")
    else:
        print("  ✓ MOGFNTrainer has modifications parameters")

    # Check run_ablation_study
    script_path = project_root / 'scripts/ablations/run_ablation_study.py'
    with open(script_path, 'r') as f:
        script_content = f.read()

    if 'modifications_type' not in script_content:
        print("\n  ✗ run_ablation_study.py needs to pass modifications to trainer:")
        print("      trainer = MOGFNTrainer(")
        print("          ...,")
        print("          modifications=config.get('modifications_type', 'none'),")
        print("          modifications_params=config.get('modifications_params', {})")
        print("      )")
    else:
        print("  ✓ run_ablation_study.py passes modifications to trainer")

    print("\n" + "="*80)


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("LOSS MODIFICATIONS TEST SUITE")
    print("="*80)
    print("\nThis test validates the loss modifications group configuration.")
    print("It checks if modifications are properly generated and identifies missing implementations.\n")

    all_tests_passed = True
    trainer_supports_modifications = False
    script_supports_modifications = False

    try:
        # Test 1: Config generation
        config, configs = test_config_generation()

        # Test 2: Trainer support
        trainer_supports_modifications = test_trainer_support()

        # Test 3: Script support
        script_supports_modifications = test_run_ablation_study_support()

        # Test 4: Config file structure
        test_config_file_structure()

        # Summary
        check_modification_implementations()

        print("\n" + "="*80)

        if trainer_supports_modifications and script_supports_modifications:
            print("✓ ALL TESTS PASSED - Modifications Fully Implemented")
            print("="*80)
            print("\nModifications are properly implemented and ready to use!")
        else:
            print("⚠ TESTS PASSED BUT IMPLEMENTATION INCOMPLETE")
            print("="*80)
            print("\nConfiguration generation works correctly, but:")

            if not trainer_supports_modifications:
                print("  • MOGFNTrainer needs to be updated to support modifications")
            if not script_supports_modifications:
                print("  • run_ablation_study.py needs to pass modifications to trainer")

            print("\nUntil these are implemented, the loss_modifications group will:")
            print("  • Generate correct config files")
            print("  • Run experiments successfully")
            print("  • BUT modifications will have NO EFFECT (all configs will behave identically)")

            print("\nRecommendation:")
            print("  Wait to run loss_modifications group until modifications are implemented.")
            print("  Or run it knowing all configs will produce identical results.")

        print("\n")

    except AssertionError as e:
        print("\n" + "="*80)
        print("✗ TEST FAILED")
        print("="*80)
        print(f"\nError: {str(e)}\n")
        all_tests_passed = False
        sys.exit(1)

    except Exception as e:
        print("\n" + "="*80)
        print("✗ TEST ERROR")
        print("="*80)
        print(f"\nUnexpected error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
        sys.exit(1)


if __name__ == '__main__':
    main()
