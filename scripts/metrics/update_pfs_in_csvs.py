#!/usr/bin/env python3
"""
Update PFS values in CSV results files after recalculating with extended implementation.

This script:
1. Scans all experiment directories to get updated PFS values from metrics.json
2. Updates all_results.csv, results.csv, and summary CSVs with new PFS values
3. Creates backups before modifying CSV files

Usage:
    # Update capacity ablation results
    python scripts/metrics/update_pfs_in_csvs.py \
        --results_dir results/ablations/capacity

    # Update sampling ablation results
    python scripts/metrics/update_pfs_in_csvs.py \
        --results_dir results/ablations/sampling

    # Update all ablations
    python scripts/metrics/update_pfs_in_csvs.py \
        --results_dir results/ablations

    # Dry run (preview changes)
    python scripts/metrics/update_pfs_in_csvs.py \
        --results_dir results/ablations/capacity \
        --dry_run
"""

import argparse
import json
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict


def load_pfs_from_experiments(results_dir: Path) -> Dict[str, float]:
    """
    Load PFS values from all experiment metrics.json files.

    Args:
        results_dir: Root directory containing experiments

    Returns:
        pfs_values: Dict mapping exp_name/identifier to PFS value
    """
    pfs_values = {}

    # Find all experiments with metrics.json
    for metrics_file in results_dir.rglob("metrics.json"):
        exp_dir = metrics_file.parent

        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)

            pfs = metrics.get('pfs')
            if pfs is None:
                continue

            # Try to get exp_name (used in ablations)
            exp_name = metrics.get('exp_name')

            # If no exp_name, try to construct identifier from algorithm + seed (used in baselines)
            if not exp_name:
                algorithm = metrics.get('algorithm')
                seed = metrics.get('seed')

                if algorithm and seed is not None:
                    # Create composite key for matching CSV rows
                    exp_name = f"{algorithm}_{seed}"
                    # Also store with just directory name for backup matching
                    dir_name = exp_dir.name
                    pfs_values[dir_name] = pfs
                else:
                    # Fall back to directory name
                    exp_name = exp_dir.name

            if exp_name:
                pfs_values[exp_name] = pfs

        except Exception as e:
            print(f"Warning: Could not load {metrics_file}: {e}")
            continue

    return pfs_values


def update_csv_pfs(csv_file: Path, pfs_values: Dict[str, float], dry_run: bool = False) -> dict:
    """
    Update PFS values in a CSV file.

    Args:
        csv_file: Path to CSV file
        pfs_values: Dict mapping exp_name to PFS value
        dry_run: If True, don't save changes

    Returns:
        stats: Dictionary with update statistics
    """
    if not csv_file.exists():
        return {
            'status': 'skipped',
            'reason': 'file does not exist'
        }

    # Load CSV
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

    # Check if pfs column exists
    if 'pfs' not in df.columns:
        return {
            'status': 'skipped',
            'reason': 'no pfs column'
        }

    # Determine identifier columns
    has_exp_name = 'exp_name' in df.columns
    has_algo_seed = 'algorithm' in df.columns and 'seed' in df.columns

    if not has_exp_name and not has_algo_seed:
        return {
            'status': 'skipped',
            'reason': 'no identifier columns (exp_name or algorithm+seed)'
        }

    # Track changes
    num_updated = 0
    old_values = []
    new_values = []

    # Update PFS values
    for idx, row in df.iterrows():
        # Try to match experiment
        exp_key = None

        if has_exp_name:
            exp_key = row['exp_name']
        elif has_algo_seed:
            # Construct composite key from algorithm + seed
            algorithm = row['algorithm']
            seed = row['seed']
            exp_key = f"{algorithm}_{seed}"

        if exp_key and exp_key in pfs_values:
            old_pfs = row['pfs']
            new_pfs = pfs_values[exp_key]

            # Check if changed
            if abs(new_pfs - old_pfs) > 1e-6:
                df.at[idx, 'pfs'] = new_pfs
                num_updated += 1
                old_values.append(old_pfs)
                new_values.append(new_pfs)

    if num_updated == 0:
        return {
            'status': 'unchanged',
            'num_rows': len(df)
        }

    # Save changes
    if not dry_run:
        # Backup original
        backup_file = csv_file.parent / f"{csv_file.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        shutil.copy(csv_file, backup_file)

        # Save updated CSV
        df.to_csv(csv_file, index=False)

    return {
        'status': 'updated',
        'num_rows': len(df),
        'num_updated': num_updated,
        'old_mean': sum(old_values) / len(old_values) if old_values else 0,
        'new_mean': sum(new_values) / len(new_values) if new_values else 0,
        'saved': not dry_run
    }


def main():
    parser = argparse.ArgumentParser(
        description="Update PFS values in CSV results files"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Root directory containing experiment results"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Preview changes without saving"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"❌ Error: Results directory not found: {results_dir}")
        return 1

    print("=" * 80)
    print("UPDATING PFS VALUES IN CSV FILES")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Dry run: {args.dry_run}")
    print()

    # Load PFS values from experiments
    print("Loading PFS values from experiment metrics.json files...")
    pfs_values = load_pfs_from_experiments(results_dir)
    print(f"Found {len(pfs_values)} experiments with PFS values")
    print()

    if len(pfs_values) == 0:
        print("❌ No experiments with PFS values found!")
        return 1

    # Find all CSV files to update
    csv_patterns = [
        "all_results.csv",
        "results.csv",
        "results_temp.csv",
        "summary*.csv",          # Root-level summary files
        "**/summary*.csv",       # Subdirectory summary files (e.g., summary_by_algorithm.csv)
        "**/all_results.csv",
        "**/results.csv"
    ]

    csv_files = set()
    for pattern in csv_patterns:
        csv_files.update(results_dir.glob(pattern))

    csv_files = sorted(csv_files)

    print(f"Found {len(csv_files)} CSV files to update")
    print("-" * 80)
    print()

    # Update each CSV file
    results = {}
    num_updated_files = 0

    for csv_file in csv_files:
        rel_path = csv_file.relative_to(results_dir)
        print(f"Processing: {rel_path}")

        result = update_csv_pfs(csv_file, pfs_values, dry_run=args.dry_run)
        results[rel_path] = result

        if result['status'] == 'updated':
            num_updated_files += 1
            saved_str = "✓ SAVED" if result['saved'] else "(DRY RUN)"
            print(f"  ✓ Updated {result['num_updated']}/{result['num_rows']} rows")
            print(f"     Old PFS mean: {result['old_mean']:.6f}")
            print(f"     New PFS mean: {result['new_mean']:.6f}")
            print(f"     {saved_str}")

        elif result['status'] == 'unchanged':
            print(f"  - No changes needed ({result['num_rows']} rows)")

        elif result['status'] == 'skipped':
            print(f"  ⊘ Skipped: {result['reason']}")

        elif result['status'] == 'error':
            print(f"  ❌ Error: {result['error']}")

        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total CSV files:      {len(csv_files)}")
    print(f"Updated:              {num_updated_files}")
    print(f"Unchanged:            {sum(1 for r in results.values() if r['status'] == 'unchanged')}")
    print(f"Skipped:              {sum(1 for r in results.values() if r['status'] == 'skipped')}")
    print(f"Errors:               {sum(1 for r in results.values() if r['status'] == 'error')}")
    print()

    if args.dry_run:
        print("🔍 DRY RUN - No files were modified")
        print("   Remove --dry_run flag to save changes")
    elif num_updated_files > 0:
        print("✅ CSV files updated successfully!")
        print(f"   Updated {num_updated_files} files")
        print(f"   Backups saved with timestamp")
    else:
        print("✅ All CSV files already up to date!")

    print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
