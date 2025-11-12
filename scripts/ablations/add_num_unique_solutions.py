#!/usr/bin/env python3
"""
Add num_unique_solutions metric to results.csv for loss ablation experiments.

This script computes the num_unique_solutions metric for all experiments
that have objectives.npy files and adds it to results.csv.

Usage:
    python scripts/ablations/add_num_unique_solutions.py

    # Or specify custom path:
    python scripts/ablations/add_num_unique_solutions.py \
        --results_dir results/ablations/loss/base_loss_comparison
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.spatial import num_unique_solutions


def compute_num_unique_solutions_for_experiment(exp_dir: Path) -> int:
    """
    Compute num_unique_solutions for a single experiment.

    Args:
        exp_dir: Experiment directory containing objectives.npy

    Returns:
        num_unique_solutions value, or None if computation failed
    """
    objectives_file = exp_dir / 'objectives.npy'

    if not objectives_file.exists():
        print(f"  ⚠️  No objectives.npy found in {exp_dir.name}")
        return None

    try:
        # Load objectives
        objectives = np.load(objectives_file)

        # Compute num_unique_solutions
        nus = num_unique_solutions(objectives, tolerance=1e-9)

        return nus

    except Exception as e:
        print(f"  ✗ Error computing num_unique_solutions for {exp_dir.name}: {e}")
        return None


def add_num_unique_solutions_to_csv(results_dir: Path, dry_run: bool = False):
    """
    Add num_unique_solutions metric to results.csv.

    Args:
        results_dir: Directory containing experiment subdirectories and results.csv
        dry_run: If True, only print what would be done without modifying files
    """
    results_csv = results_dir / 'results.csv'

    print("="*80)
    print("ADDING num_unique_solutions TO results.csv")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Results CSV: {results_csv}")
    print()

    # Load existing results
    if not results_csv.exists():
        print(f"✗ Error: results.csv not found at {results_csv}")
        return

    df = pd.read_csv(results_csv)
    print(f"Loaded results.csv with {len(df)} experiments")

    # Check if num_unique_solutions already exists
    if 'num_unique_solutions' in df.columns:
        print("\n⚠️  Column 'num_unique_solutions' already exists in results.csv")
        response = input("Do you want to recompute it? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        print("\nRecomputing num_unique_solutions for all experiments...")
    else:
        print("\n✓ Column 'num_unique_solutions' not found - will add it")

    print()

    # Process each experiment
    num_computed = 0
    num_failed = 0

    nus_values = []

    for _, row in df.iterrows():
        exp_name = row['exp_name']

        # Find experiment directory (try with and without group prefix)
        exp_dir = results_dir / exp_name

        if not exp_dir.exists():
            # Try removing common prefixes
            for prefix in ['base_loss_comparison_', 'entropy_regularization_',
                          'kl_regularization_', 'loss_modifications_']:
                alt_name = exp_name.replace(prefix, '')
                alt_dir = results_dir / alt_name
                if alt_dir.exists():
                    exp_dir = alt_dir
                    break

        if not exp_dir.exists():
            print(f"  ⚠️  Directory not found for: {exp_name}")
            nus_values.append(np.nan)
            num_failed += 1
            continue

        # Compute num_unique_solutions
        nus = compute_num_unique_solutions_for_experiment(exp_dir)

        if nus is not None:
            nus_values.append(nus)
            num_computed += 1
            if (num_computed % 5) == 0 or num_computed <= 3:
                print(f"  ✓ {exp_name}: {nus} unique solutions")
        else:
            nus_values.append(np.nan)
            num_failed += 1

    print()
    print("-"*80)
    print("SUMMARY")
    print("-"*80)
    print(f"Total experiments: {len(df)}")
    print(f"Successfully computed: {num_computed}")
    print(f"Failed: {num_failed}")
    print()

    if num_computed == 0:
        print("✗ No metrics were computed - aborting")
        return

    if dry_run:
        print("[DRY RUN] Would add num_unique_solutions column with these values:")
        for i, (exp_name, nus) in enumerate(zip(df['exp_name'], nus_values)):
            if i < 5 or i >= len(nus_values) - 2:
                print(f"  {exp_name}: {nus}")
            elif i == 5:
                print(f"  ... ({len(nus_values) - 7} more) ...")
        return

    # Add num_unique_solutions column
    df['num_unique_solutions'] = nus_values

    # Move num_unique_solutions to be near other diversity metrics (after mce)
    cols = df.columns.tolist()
    if 'num_unique_solutions' in cols and 'mce' in cols:
        cols.remove('num_unique_solutions')
        mce_idx = cols.index('mce')
        cols.insert(mce_idx + 1, 'num_unique_solutions')
        df = df[cols]

    # Save updated results
    df.to_csv(results_csv, index=False)

    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"✓ Added num_unique_solutions column to results.csv")
    print(f"✓ Saved to: {results_csv}")
    print()
    print("📊 Statistics:")
    print(f"  Mean: {df['num_unique_solutions'].mean():.2f}")
    print(f"  Median: {df['num_unique_solutions'].median():.2f}")
    print(f"  Min: {df['num_unique_solutions'].min():.0f}")
    print(f"  Max: {df['num_unique_solutions'].max():.0f}")
    print(f"  Std: {df['num_unique_solutions'].std():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Add num_unique_solutions metric to results.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add num_unique_solutions to base_loss_comparison
  python scripts/ablations/add_num_unique_solutions.py

  # Dry run to see what would be computed
  python scripts/ablations/add_num_unique_solutions.py --dry-run

  # Specify custom directory
  python scripts/ablations/add_num_unique_solutions.py \\
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
        help='Show what would be computed without modifying files'
    )

    args = parser.parse_args()

    # Check that directory exists
    if not args.results_dir.exists():
        print(f"Error: Directory does not exist: {args.results_dir}")
        sys.exit(1)

    # Add num_unique_solutions metric
    add_num_unique_solutions_to_csv(args.results_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
