#!/usr/bin/env python3
"""
Populate missing experiments in results.csv for base_loss_comparison.

This script scans the base_loss_comparison directory for experiment directories
that have metrics.json files but are not present in results.csv, and adds them.

Usage:
    python scripts/ablations/populate_missing_results.py

    # Or specify custom paths:
    python scripts/ablations/populate_missing_results.py \
        --results_dir results/ablations/loss/base_loss_comparison
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


def find_experiment_directories(base_dir: Path) -> list:
    """Find all experiment directories with metrics.json."""
    exp_dirs = []

    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            metrics_file = item / 'metrics.json'
            if metrics_file.exists():
                exp_dirs.append(item)

    return sorted(exp_dirs)


def load_experiment_metrics(exp_dir: Path) -> dict:
    """Load metrics from an experiment directory."""
    metrics_file = exp_dir / 'metrics.json'

    if not metrics_file.exists():
        return None

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    return metrics


def populate_missing_experiments(results_dir: Path, dry_run: bool = False):
    """
    Add missing experiments to results.csv.

    Args:
        results_dir: Directory containing experiment subdirectories and results.csv
        dry_run: If True, only print what would be added without modifying files
    """
    results_csv = results_dir / 'results.csv'

    print("="*80)
    print("POPULATING MISSING EXPERIMENTS IN results.csv")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Results CSV: {results_csv}")
    print()

    # Load existing results
    if results_csv.exists():
        existing_df = pd.read_csv(results_csv)
        existing_exp_names = set(existing_df['exp_name'].values)
        print(f"Existing results.csv has {len(existing_df)} experiments")
    else:
        existing_df = pd.DataFrame()
        existing_exp_names = set()
        print("No existing results.csv found - will create new file")

    print()

    # Find all experiment directories
    exp_dirs = find_experiment_directories(results_dir)
    print(f"Found {len(exp_dirs)} experiment directories")
    print()

    # Check which are missing
    missing_experiments = []

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name

        # Check if already in CSV
        if exp_name in existing_exp_names:
            continue

        # Try to load metrics
        metrics = load_experiment_metrics(exp_dir)
        if metrics is None:
            print(f"⚠️  Skipping {exp_name}: No valid metrics.json")
            continue

        missing_experiments.append({
            'exp_dir': exp_dir,
            'exp_name': exp_name,
            'metrics': metrics
        })

    print("-"*80)
    print(f"SUMMARY")
    print("-"*80)
    print(f"Total experiment directories: {len(exp_dirs)}")
    print(f"Already in results.csv: {len(existing_exp_names)}")
    print(f"Missing from results.csv: {len(missing_experiments)}")
    print()

    if not missing_experiments:
        print("✓ All experiments are already in results.csv")
        return

    print("Missing experiments:")
    for i, exp in enumerate(missing_experiments, 1):
        print(f"  {i}. {exp['exp_name']}")
    print()

    if dry_run:
        print("[DRY RUN] Would add these experiments to results.csv")
        return

    # Create DataFrame for missing experiments
    missing_rows = []
    for exp in missing_experiments:
        missing_rows.append(exp['metrics'])

    missing_df = pd.DataFrame(missing_rows)

    # Ensure columns match existing CSV
    if not existing_df.empty:
        # Reorder columns to match existing CSV
        missing_df = missing_df.reindex(columns=existing_df.columns, fill_value=np.nan)

    # Concatenate with existing results
    if existing_df.empty:
        updated_df = missing_df
    else:
        updated_df = pd.concat([existing_df, missing_df], ignore_index=True)

    # Sort by exp_name for consistency
    updated_df = updated_df.sort_values('exp_name')

    # Save updated results
    updated_df.to_csv(results_csv, index=False)

    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"✓ Added {len(missing_experiments)} experiments to results.csv")
    print(f"✓ Updated results.csv now has {len(updated_df)} experiments")
    print(f"✓ Saved to: {results_csv}")


def main():
    parser = argparse.ArgumentParser(
        description='Populate missing experiments in results.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add missing experiments to base_loss_comparison
  python scripts/ablations/populate_missing_results.py

  # Dry run to see what would be added
  python scripts/ablations/populate_missing_results.py --dry-run

  # Specify custom directory
  python scripts/ablations/populate_missing_results.py \\
      --results_dir results/ablations/loss/entropy_regularization
        """
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        default=project_root / 'results' / 'ablations' / 'loss' / 'base_loss_comparison',
        help='Directory containing experiment subdirectories and results.csv'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be added without modifying files'
    )

    args = parser.parse_args()

    # Check that directory exists
    if not args.results_dir.exists():
        print(f"Error: Directory does not exist: {args.results_dir}")
        sys.exit(1)

    # Populate missing experiments
    populate_missing_experiments(args.results_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
