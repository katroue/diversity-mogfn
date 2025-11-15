#!/usr/bin/env python3
"""
Recompute hypervolume metric for loss ablation experiments.

This script recomputes the hypervolume metric for all experiments in the loss ablation
study directories and updates both metrics.json files and results.csv.

IMPORTANT: Run this if you suspect hypervolume calculations were incorrect due to bugs
in the original implementation.

Usage:
    # Recompute for base_loss_comparison
    python scripts/ablations/recompute_hypervolume_loss.py \
        --group base_loss_comparison

    # Recompute for all loss ablation groups
    python scripts/ablations/recompute_hypervolume_loss.py --all

    # Dry run to preview changes
    python scripts/ablations/recompute_hypervolume_loss.py \
        --group base_loss_comparison \
        --dry-run
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.traditional import hypervolume


def get_reference_point(objectives: np.ndarray) -> np.ndarray:
    """
    Get reference point for hypervolume computation.

    Uses the nadir point (worst values across all objectives) plus a small margin.

    Args:
        objectives: Array of shape (n_samples, n_objectives)

    Returns:
        Reference point array
    """
    # For maximization problems, reference point should be below nadir
    # Add small margin (5%) below minimum values
    ref_point = np.min(objectives, axis=0) * 0.95
    return ref_point


def recompute_hypervolume_for_experiment(exp_dir: Path, dry_run: bool = False) -> Dict:
    """
    Recompute hypervolume for a single experiment.

    Args:
        exp_dir: Experiment directory containing objectives.npy and metrics.json
        dry_run: If True, don't actually save changes

    Returns:
        Dictionary with old_hv, new_hv, and change information
    """
    objectives_file = exp_dir / 'objectives.npy'
    metrics_file = exp_dir / 'metrics.json'

    if not objectives_file.exists():
        return {
            'exp_name': exp_dir.name,
            'status': 'missing_objectives',
            'old_hv': None,
            'new_hv': None,
            'change': None
        }

    if not metrics_file.exists():
        return {
            'exp_name': exp_dir.name,
            'status': 'missing_metrics',
            'old_hv': None,
            'new_hv': None,
            'change': None
        }

    try:
        # Load objectives
        objectives = np.load(objectives_file)

        # Load current metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        old_hv = metrics.get('hypervolume', None)

        # Compute reference point
        ref_point = get_reference_point(objectives)

        # Recompute hypervolume
        new_hv = hypervolume(objectives, ref_point, maximize=True)

        # Calculate change
        if old_hv is not None:
            change = new_hv - old_hv
            pct_change = (change / old_hv * 100) if old_hv != 0 else 0
        else:
            change = None
            pct_change = None

        # Update metrics if not dry run
        if not dry_run:
            metrics['hypervolume'] = float(new_hv)
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)

        return {
            'exp_name': exp_dir.name,
            'status': 'success',
            'old_hv': old_hv,
            'new_hv': float(new_hv),
            'change': change,
            'pct_change': pct_change
        }

    except Exception as e:
        return {
            'exp_name': exp_dir.name,
            'status': 'error',
            'old_hv': None,
            'new_hv': None,
            'change': None,
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
        print(f"  ⚠️  No metrics.json files found in {group_dir}")
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

    # Save
    if not dry_run:
        df.to_csv(results_csv, index=False)
        print(f"  ✓ Saved: {results_csv}")
    else:
        print(f"  [DRY RUN] Would save to: {results_csv}")


def recompute_hypervolume_for_group(group_name: str, dry_run: bool = False):
    """
    Recompute hypervolume for all experiments in a loss ablation group.

    Args:
        group_name: Name of the group (e.g., 'base_loss_comparison')
        dry_run: If True, preview changes without saving
    """
    base_dir = Path('results/ablations/loss')
    group_dir = base_dir / group_name

    if not group_dir.exists():
        print(f"✗ Error: Group directory does not exist: {group_dir}")
        return

    print("="*80)
    print(f"RECOMPUTING HYPERVOLUME: {group_name}")
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
        result = recompute_hypervolume_for_experiment(exp_dir, dry_run=dry_run)
        results.append(result)

        if result['status'] == 'success':
            num_success += 1
            if result['old_hv'] is not None and result['change'] is not None:
                if abs(result['change']) > 0.001:  # Significant change
                    print(f"  {'[DRY RUN] ' if dry_run else ''}✓ {result['exp_name']}")
                    print(f"      Old HV: {result['old_hv']:.6f}")
                    print(f"      New HV: {result['new_hv']:.6f}")
                    print(f"      Change: {result['change']:+.6f} ({result['pct_change']:+.2f}%)")
        elif result['status'] in ['missing_objectives', 'missing_metrics']:
            num_missing += 1
            print(f"  ⚠️  {result['exp_name']}: {result['status']}")
        else:
            num_errors += 1
            print(f"  ✗ {result['exp_name']}: {result.get('error', 'Unknown error')}")

    print()
    print("-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Total experiments: {len(exp_dirs)}")
    print(f"Successfully recomputed: {num_success}")
    print(f"Missing files: {num_missing}")
    print(f"Errors: {num_errors}")

    # Statistics on changes
    changes = [r['change'] for r in results if r['status'] == 'success' and r['change'] is not None]
    if changes:
        print(f"\nHypervolume changes:")
        print(f"  Mean change: {np.mean(changes):+.6f}")
        print(f"  Max increase: {np.max(changes):+.6f}")
        print(f"  Max decrease: {np.min(changes):+.6f}")
        print(f"  Std dev: {np.std(changes):.6f}")

    # Regenerate results.csv
    if num_success > 0:
        regenerate_results_csv(group_dir, dry_run=dry_run)

    print(f"\n{'✓ Done! (DRY RUN - no changes made)' if dry_run else '✓ Done!'}")


def main():
    parser = argparse.ArgumentParser(
        description='Recompute hypervolume metric for loss ablation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--group',
        type=str,
        help='Loss ablation group to process (e.g., base_loss_comparison, loss_modifications)'
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

    # Define all known loss ablation groups
    all_groups = [
        'base_loss_comparison',
        'loss_modifications',
        'entropy_regularization',
        'kl_regularization'
    ]

    if args.all:
        # Process all groups
        base_dir = Path('results/ablations/loss')
        existing_groups = [d.name for d in base_dir.iterdir()
                          if d.is_dir() and not d.name.startswith('.') and d.name != 'report']

        print(f"Processing all groups: {', '.join(existing_groups)}\n")

        for group in existing_groups:
            recompute_hypervolume_for_group(group, dry_run=args.dry_run)
            print("\n")

    elif args.group:
        # Process single group
        recompute_hypervolume_for_group(args.group, dry_run=args.dry_run)

    else:
        parser.print_help()
        print("\nError: Must specify either --group <name> or --all")
        sys.exit(1)


if __name__ == '__main__':
    main()
