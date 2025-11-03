#!/usr/bin/env python3
"""
Run loss ablation study by experiment group.

This script runs the loss ablation study one group at a time, allowing for
better analysis and resource management. Results from earlier groups can
inform decisions about later groups.

Usage:
    # Run a specific group
    python scripts/ablations/run_loss_ablation_group.py --group kl_regularization

    # Run all groups sequentially
    python scripts/ablations/run_loss_ablation_group.py --all

    # List available groups
    python scripts/ablations/run_loss_ablation_group.py --list

    # Resume a group that was interrupted
    python scripts/ablations/run_loss_ablation_group.py --group base_loss_comparison --resume

    # Dry run to see what would be executed
    python scripts/ablations/run_loss_ablation_group.py --group base_loss_comparison --dry-run

Available Groups (in recommended order):
    1. base_loss_comparison     - Compare base GFlowNet losses (6 configs × 5 seeds = 30 runs)
    2. entropy_regularization   - Test entropy regularization (5 configs × 5 seeds = 25 runs)
    3. kl_regularization        - Test KL divergence regularization (3 configs × 5 seeds = 15 runs)
    4. subtb_entropy_sweep      - SubTB + entropy combinations (4 configs × 5 seeds = 20 runs)
    5. loss_modifications       - Test loss modifications (3 configs × 5 seeds = 15 runs)
"""

import sys
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Add project root to path
# __file__ is in scripts/ablations/, so go up 2 levels to project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.ablations.run_ablation_study import run_single_experiment

def load_loss_ablation_config() -> dict:
    """Load the loss ablation configuration."""
    config_path = project_root / 'configs/ablations/loss_ablation.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def list_experiment_groups(config: dict) -> None:
    """List all available experiment groups."""
    print("\n" + "="*80)
    print("LOSS ABLATION STUDY - EXPERIMENT GROUPS")
    print("="*80)

    experiments = config.get('experiments', [])

    for i, exp_group in enumerate(experiments, 1):
        group_name = exp_group['group']
        description = exp_group['description']
        vary_factor = list(exp_group.get('vary', {}).keys())[0]
        vary_options = exp_group['vary'][vary_factor]
        num_configs = len(vary_options)
        num_seeds = config['fixed'].get('num_seeds', 5)
        total_runs = num_configs * num_seeds

        print(f"\n{i}. {group_name}")
        print(f"   Description: {description}")
        print(f"   Varying: {vary_factor}")
        print(f"   Options: {', '.join(vary_options)}")
        print(f"   Total runs: {num_configs} configs × {num_seeds} seeds = {total_runs} runs")

        # Show fixed parameters for this group
        fixed_params = exp_group.get('fixed', {})
        if fixed_params:
            print(f"   Fixed: {', '.join([f'{k}={v}' for k, v in fixed_params.items()])}")

    print("\n" + "="*80)
    print(f"Total groups: {len(experiments)}")
    total_runs = sum(len(exp['vary'][list(exp['vary'].keys())[0]]) * num_seeds
                    for exp in experiments)
    print(f"Total runs across all groups: {total_runs}")
    print("="*80 + "\n")


def get_experiment_group(config: dict, group_name: str) -> Optional[dict]:
    """Get a specific experiment group by name."""
    experiments = config.get('experiments', [])
    for exp_group in experiments:
        if exp_group['group'] == group_name:
            return exp_group
    return None


def generate_group_configs(config: dict, group_name: str) -> List[Dict]:
    """
    Generate individual experiment configurations for a group.

    Returns:
        List of experiment configurations ready to run
    """
    exp_group = get_experiment_group(config, group_name)
    if not exp_group:
        raise ValueError(f"Group '{group_name}' not found in configuration")

    # Get base configuration
    fixed_config = config['fixed']
    group_fixed = exp_group.get('fixed', {})
    ablation_factors = config['ablation_factors']

    # Get the varying parameter
    vary = exp_group['vary']
    vary_param = list(vary.keys())[0]
    vary_values = vary[vary_param]

    # Generate seeds
    num_seeds = fixed_config.get('num_seeds', 5)
    base_seed = fixed_config.get('base_seed', 42)
    seeds = [base_seed + i * 111 for i in range(num_seeds)]

    # Generate configurations
    configs = []
    for value in vary_values:
        for seed in seeds:
            # Get the option configuration from ablation_factors
            option_config = None
            for option in ablation_factors[vary_param]['options']:
                if option['name'] == value:
                    option_config = option
                    break

            if not option_config:
                print(f"Warning: Could not find configuration for {vary_param}={value}")
                continue

            # Build experiment config
            exp_config = {
                'name': value,
                'group': group_name,
                'seed': seed,
                vary_param: value,
                f'{vary_param}_type': option_config['type'],
                f'{vary_param}_params': option_config.get('params', {}),
                f'{vary_param}_label': option_config.get('label', value),
            }

            # Add fixed parameters from group
            # Need to process each fixed parameter through ablation_factors lookup
            for param_name, param_value in group_fixed.items():
                # Add the raw value
                exp_config[param_name] = param_value

                # Look up the configuration in ablation_factors if it exists
                if param_name in ablation_factors:
                    for option in ablation_factors[param_name]['options']:
                        if option['name'] == param_value:
                            exp_config[f'{param_name}_type'] = option['type']
                            exp_config[f'{param_name}_params'] = option.get('params', {})
                            exp_config[f'{param_name}_label'] = option.get('label', param_value)
                            break

            configs.append(exp_config)

    return configs


def run_group(config: dict,
             group_name: str,
             output_dir: Path,
             resume: bool = False,
             dry_run: bool = False,
             device: str = 'cpu') -> None:
    """
    Run all experiments in a group.

    Args:
        config: Full loss ablation configuration
        group_name: Name of the group to run
        output_dir: Output directory for results
        resume: Whether to skip already completed experiments
        dry_run: If True, only print what would be run without executing
        device: Device to use for training ('cpu' or 'cuda')
    """
    from tqdm import tqdm

    # Get group info
    exp_group = get_experiment_group(config, group_name)
    if not exp_group:
        raise ValueError(f"Group '{group_name}' not found")

    # Create output directory
    group_dir = output_dir / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    # Generate configurations
    print(f"\n{'='*80}")
    print(f"RUNNING GROUP: {group_name}")
    print(f"Description: {exp_group['description']}")
    print(f"{'='*80}\n")

    configs = generate_group_configs(config, group_name)
    print(f"Generated {len(configs)} experiment configurations")

    # Check for existing results if resuming
    completed = set()
    results_file = group_dir / 'results.csv'
    if resume and results_file.exists():
        df = pd.read_csv(results_file)
        # Handle both formats: with and without group prefix
        # CSV might have: "base_loss_comparison_trajectory_balance_seed42"
        # Script uses: "trajectory_balance_seed42"
        group_prefix = f"{group_name}_"
        completed_names = df['exp_name'].values
        completed = set()
        for name in completed_names:
            # Strip group prefix if present
            if name.startswith(group_prefix):
                completed.add(name[len(group_prefix):])
            else:
                completed.add(name)
        print(f"Found {len(completed)} completed experiments, will skip these")

    # Save group configuration
    group_config_file = group_dir / 'group_config.yaml'
    with open(group_config_file, 'w') as f:
        yaml.dump({
            'group': exp_group,
            'fixed': config['fixed'],
            'timestamp': datetime.now().isoformat()
        }, f)

    if dry_run:
        print("\n[DRY RUN] Would execute the following experiments:")
        for i, exp_config in enumerate(configs, 1):
            exp_name = f"{exp_config['name']}_seed{exp_config['seed']}"
            status = "[SKIP - COMPLETED]" if exp_name in completed else "[RUN]"
            print(f"  {i:3d}. {status} {exp_name}")
        print(f"\nTotal to run: {len([c for c in configs if f'{c['name']}_seed{c['seed']}' not in completed])}")
        return

    # Run experiments
    results = []
    failed = []

    for exp_config in tqdm(configs, desc=f"Running {group_name}"):
        exp_name = f"{exp_config['name']}_seed{exp_config['seed']}"

        # Skip if already completed
        if exp_name in completed:
            tqdm.write(f"  ✓ Skipping {exp_name} (already completed)")
            continue

        # Save individual experiment config
        exp_config_full = {**config['fixed'], **exp_config}
        exp_config_file = group_dir / f'{exp_name}_config.json'
        with open(exp_config_file, 'w') as f:
            json.dump(exp_config_full, f, indent=2)

        # Create experiment directory
        exp_dir = group_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        tqdm.write(f"  Running {exp_name}...")

        try:
            # Run the actual experiment using run_single_experiment
            result_metrics = run_single_experiment(
                exp_config=exp_config,
                fixed_config=config['fixed'],
                seed=exp_config['seed'],
                output_dir=group_dir,
                device=device
            )

            # Add group information to results
            result_metrics['group'] = group_name
            results.append(result_metrics)

            tqdm.write(f"  ✓ Completed {exp_name}")

        except Exception as e:
            tqdm.write(f"  ✗ Failed {exp_name}: {str(e)}")
            failed.append({'exp_name': exp_name, 'error': str(e)})

    # Save results
    if results:
        df_results = pd.DataFrame(results)

        # Merge with existing results if resuming
        if resume and results_file.exists():
            df_existing = pd.read_csv(results_file)
            df_results = pd.concat([df_existing, df_results], ignore_index=True)

        df_results.to_csv(results_file, index=False)
        print(f"\n✓ Saved results to {results_file}")

    # Save failed experiments
    if failed:
        failed_file = group_dir / 'failed.json'
        with open(failed_file, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f"⚠ {len(failed)} experiments failed, see {failed_file}")

    # Print summary
    print(f"\n{'='*80}")
    print(f"GROUP SUMMARY: {group_name}")
    print(f"{'='*80}")
    print(f"  Total experiments: {len(configs)}")
    print(f"  Completed: {len(results)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Skipped (already done): {len(completed)}")
    print(f"{'='*80}\n")


def run_all_groups(config: dict,
                  output_dir: Path,
                  resume: bool = False,
                  dry_run: bool = False,
                  device: str = 'cpu') -> None:
    """Run all experiment groups sequentially."""
    experiments = config.get('experiments', [])

    print("\n" + "="*80)
    print("RUNNING ALL GROUPS SEQUENTIALLY")
    print("="*80)

    for i, exp_group in enumerate(experiments, 1):
        group_name = exp_group['group']
        print(f"\n[{i}/{len(experiments)}] Starting group: {group_name}")

        run_group(config, group_name, output_dir, resume, dry_run, device)

        if i < len(experiments):
            print(f"\n✓ Completed group {i}/{len(experiments)}")
            print("  Moving to next group...\n")

    print("\n" + "="*80)
    print("ALL GROUPS COMPLETED")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run loss ablation study by experiment group',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--group',
        type=str,
        help='Experiment group to run (e.g., base_loss_comparison)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all groups sequentially'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available experiment groups'
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('results/ablations/loss'),
        help='Output directory for results (default: results/ablations/loss)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from existing results (skip completed experiments)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be run without executing'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device to use for training (cpu or cuda, default: cpu)'
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_loss_ablation_config()
    except FileNotFoundError:
        print("Error: Could not find configs/ablations/loss_ablation.yaml")
        print("Please ensure the configuration file exists.")
        sys.exit(1)

    # Handle commands
    if args.list:
        list_experiment_groups(config)
        return

    if args.all:
        run_all_groups(config, args.output_dir, args.resume, args.dry_run, args.device)
        return

    if args.group:
        run_group(config, args.group, args.output_dir, args.resume, args.dry_run, args.device)
        return

    # No command specified
    parser.print_help()
    print("\nNo action specified. Use --list to see available groups,")
    print("--group <name> to run a specific group, or --all to run all groups.")


if __name__ == '__main__':
    main()
