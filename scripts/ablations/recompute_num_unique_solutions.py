#!/usr/bin/env python3
"""
Recompute num_unique_solutions metric for loss ablation experiments.

This script recomputes the num_unique_solutions metric for all experiments
by loading objectives.npy files and counting unique solutions.

Usage:
    # For base_loss_comparison
    python scripts/ablations/recompute_num_unique_solutions.py \
        --group base_loss_comparison

    # For all loss ablation groups
    python scripts/ablations/recompute_num_unique_solutions.py --all

    # Dry run
    python scripts/ablations/recompute_num_unique_solutions.py \
        --group base_loss_comparison \
        --dry-run
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def num_unique_solutions(objectives: np.ndarray, tolerance: float = 1e-9) -> int:
    """
    Count the number of unique solutions in objective space.

    Args:
        objectives: Array of shape (N, num_objectives)
        tolerance: Tolerance for considering solutions as unique (default: 1e-9)

    Returns:
        Number of unique solutions
    """
    # Convert tensor to numpy if needed
    if hasattr(objectives, 'cpu'):
        objectives = objectives.cpu().numpy()

    objectives = np.atleast_2d(objectives)

    if len(objectives) == 0:
        return 0

    if len(objectives) == 1:
        return 1

    # Round to handle floating point precision
    if tolerance > 0:
        decimals = max(0, int(-np.log10(tolerance)))
        objectives_rounded = np.round(objectives, decimals=decimals)
        unique_objectives = np.unique(objectives_rounded, axis=0)
    else:
        unique_objectives = np.unique(objectives, axis=0)

    return int(len(unique_objectives))


def recompute_num_unique_for_experiment(exp_dir: Path, dry_run: bool = False):
    """
    Recompute num_unique_solutions for a single experiment.

    Args:
        exp_dir: Experiment directory
        dry_run: If True, don't save changes

    Returns:
        Dictionary with results
    """
    objectives_file = exp_dir / 'objectives.npy'
    metrics_file = exp_dir / 'metrics.json'

    if not objectives_file.exists():
        return {
            'exp_name': exp_dir.name,
            'status': 'missing_objectives',
            'old_nus': None,
            'new_nus': None
        }

    if not metrics_file.exists():
        return {
            'exp_name': exp_dir.name,
            'status': 'missing_metrics',
            'old_nus': None,
            'new_nus': None
        }

    try:
        # Load objectives and metrics
        objectives = np.load(objectives_file)
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        old_nus = metrics.get('num_unique_solutions', None)

        # Recompute num_unique_solutions
        new_nus = num_unique_solutions(objectives, tolerance=1e-9)

        # Update metrics if not dry run
        if not dry_run:
            metrics['num_unique_solutions'] = int(new_nus)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        return {
            'exp_name': exp_dir.name,
            'status': 'success',
            'old_nus': old_nus,
            'new_nus': int(new_nus),
            'total_solutions': len(objectives),
            'uniqueness_pct': (new_nus / len(objectives) * 100) if len(objectives) > 0 else 0
        }

    except Exception as e:
        return {
            'exp_name': exp_dir.name,
            'status': 'error',
            'error': str(e)
        }


def regenerate_results_csv(group_dir: Path, dry_run: bool = False):
    """
    Regenerate results.csv from all metrics.json files.

    Args:
        group_dir: Directory containing experiment subdirectories
        dry_run: If True, don't actually save the CSV
    """
    results_csv = group_dir / 'results.csv'

    # Find all metrics.json files
    metrics_files = sorted(group_dir.glob('*/metrics.json'))

    if not metrics_files:
        print(f"  ⚠️  No metrics.json files found")
        return

    print(f"\n  Regenerating results.csv from {len(metrics_files)} experiments...")

    # Load all metrics
    all_metrics = []
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        except Exception as e:
            print(f"  ⚠️  Could not load {metrics_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Sort by exp_name
    if 'exp_name' in df.columns:
        df = df.sort_values('exp_name')

    print(f"  Generated DataFrame with {len(df)} rows, {len(df.columns)} columns")

    # Check num_unique_solutions coverage
    if 'num_unique_solutions' in df.columns:
        non_null = df['num_unique_solutions'].notna().sum()
        print(f"  num_unique_solutions: {non_null}/{len(df)} non-null")

    # Save
    if not dry_run:
        df.to_csv(results_csv, index=False)
        print(f"  ✓ Saved: {results_csv}")
    else:
        print(f"  [DRY RUN] Would save to: {results_csv}")


def recompute_for_group(group_name: str, dry_run: bool = False):
    """
    Recompute num_unique_solutions for all experiments in a group.

    Args:
        group_name: Name of the loss ablation group
        dry_run: If True, preview changes without saving
    """
    base_dir = Path('results/ablations/loss')
    group_dir = base_dir / group_name

    if not group_dir.exists():
        print(f"✗ Error: Group directory does not exist: {group_dir}")
        return

    print("="*80)
    print(f"RECOMPUTING NUM_UNIQUE_SOLUTIONS: {group_name}")
    print("="*80)
    print(f"Directory: {group_dir}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()

    # Find all experiment directories
    exp_dirs = sorted([d for d in group_dir.iterdir()
                      if d.is_dir() and not d.name.startswith('.')])

    print(f"Found {len(exp_dirs)} experiment directories\n")

    # Process each experiment
    results = []
    num_success = 0
    num_errors = 0
    num_missing = 0

    for exp_dir in exp_dirs:
        result = recompute_num_unique_for_experiment(exp_dir, dry_run=dry_run)
        results.append(result)

        if result['status'] == 'success':
            num_success += 1
            # Show a sample of results
            if num_success <= 5 or (result['old_nus'] != result['new_nus']):
                print(f"  {'[DRY RUN] ' if dry_run else ''}✓ {result['exp_name']}")
                if result['old_nus'] is not None:
                    print(f"      Old: {result['old_nus']}")
                print(f"      New: {result['new_nus']}/{result['total_solutions']} "
                      f"({result['uniqueness_pct']:.1f}% unique)")
        elif result['status'] in ['missing_objectives', 'missing_metrics']:
            num_missing += 1
            print(f"  ⚠️  {result['exp_name']}: {result['status']}")
        else:
            num_errors += 1
            print(f"  ✗ {result['exp_name']}: {result.get('error', 'Unknown error')}")

    if num_success > 5:
        print(f"\n  ... and {num_success - 5} more successfully processed")

    print()
    print("-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Total experiments: {len(exp_dirs)}")
    print(f"Successfully recomputed: {num_success}")
    print(f"Missing files: {num_missing}")
    print(f"Errors: {num_errors}")

    # Statistics on uniqueness
    success_results = [r for r in results if r['status'] == 'success']
    if success_results:
        nus_values = [r['new_nus'] for r in success_results]
        uniqueness_pcts = [r['uniqueness_pct'] for r in success_results]

        print(f"\nUniqueness Statistics:")
        print(f"  Mean unique solutions: {np.mean(nus_values):.1f}")
        print(f"  Range: [{np.min(nus_values)}, {np.max(nus_values)}]")
        print(f"  Mean uniqueness: {np.mean(uniqueness_pcts):.1f}%")

        # Group by loss type if exp_name contains loss type
        from collections import defaultdict
        by_loss_type = defaultdict(list)
        for r in success_results:
            for loss_type in ['trajectory_balance', 'detailed_balance', 'flow_matching',
                            'subtrajectory_balance_05', 'subtrajectory_balance_09',
                            'subtrajectory_balance_095']:
                if loss_type in r['exp_name']:
                    by_loss_type[loss_type].append(r['new_nus'])
                    break

        if by_loss_type:
            print(f"\nBy Loss Type:")
            for loss_type, values in sorted(by_loss_type.items()):
                print(f"  {loss_type:30s}: mean={np.mean(values):.1f}, "
                      f"range=[{np.min(values)}, {np.max(values)}]")

    # Regenerate results.csv
    if num_success > 0:
        regenerate_results_csv(group_dir, dry_run=dry_run)

    print(f"\n{'✓ Done! (DRY RUN - no changes made)' if dry_run else '✓ Done!'}")


def main():
    parser = argparse.ArgumentParser(
        description='Recompute num_unique_solutions for loss ablation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--group',
        type=str,
        help='Loss ablation group to process (e.g., base_loss_comparison)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all loss ablation groups'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    args = parser.parse_args()

    if args.all:
        # Process all groups
        base_dir = Path('results/ablations/loss')
        existing_groups = [d.name for d in base_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.') and d.name != 'report']

        print(f"Processing all groups: {', '.join(existing_groups)}\n")

        for group in existing_groups:
            recompute_for_group(group, dry_run=args.dry_run)
            print("\n")

    elif args.group:
        # Process single group
        recompute_for_group(args.group, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\nError: Must specify either --group <name> or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
