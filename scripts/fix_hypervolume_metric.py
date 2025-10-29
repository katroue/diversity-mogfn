#!/usr/bin/env python3
"""
Fix hypervolume metric for ablation study results.

This script recalculates hypervolume values using the corrected implementation
in src/metrics/traditional.py. The original implementation had a bug where only
the first point contributed to the hypervolume calculation.

Usage:
    # Fix capacity ablation results
    python scripts/fix_hypervolume_metric.py \
        --results_dir results/ablations/capacity

    # Fix sampling ablation results
    python scripts/fix_hypervolume_metric.py \
        --results_dir results/ablations/sampling

    # Dry run (show changes without saving)
    python scripts/fix_hypervolume_metric.py \
        --results_dir results/ablations/capacity \
        --dry_run
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.traditional import hypervolume


def fix_hypervolume_for_experiment(exp_dir: Path, reference_point: np.ndarray) -> dict:
    """
    Recalculate hypervolume for a single experiment.

    Args:
        exp_dir: Experiment directory containing objectives.npy
        reference_point: Reference point for hypervolume calculation

    Returns:
        dict with old_hv, new_hv, exp_name
    """
    objectives_file = exp_dir / 'objectives.npy'

    if not objectives_file.exists():
        print(f"  ⚠ Warning: No objectives.npy found in {exp_dir.name}")
        return None

    # Load objectives
    objectives = np.load(objectives_file)

    # Load old metrics to get old hypervolume
    metrics_file = exp_dir / 'metrics.json'
    old_hv = None
    if metrics_file.exists():
        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            old_hv = metrics.get('hypervolume', None)

    # Compute new hypervolume with fixed implementation
    new_hv = hypervolume(objectives, reference_point)

    return {
        'exp_name': exp_dir.name,
        'old_hv': old_hv,
        'new_hv': new_hv,
        'difference': abs(new_hv - old_hv) if old_hv is not None else None,
        'num_objectives': len(objectives)
    }


def fix_hypervolume_in_csv(results_csv: Path,
                           results_dir: Path,
                           reference_point: np.ndarray,
                           dry_run: bool = False) -> pd.DataFrame:
    """
    Fix hypervolume values in the results CSV.

    Args:
        results_csv: Path to all_results.csv or results_temp.csv
        results_dir: Directory containing experiment subdirectories
        reference_point: Reference point for hypervolume
        dry_run: If True, don't save changes

    Returns:
        Updated DataFrame
    """
    # Load results CSV
    df = pd.read_csv(results_csv)
    print(f"\nLoaded {len(df)} experiments from {results_csv.name}")

    if 'hypervolume' not in df.columns:
        print("✗ Error: No 'hypervolume' column found in CSV")
        return df

    old_hv_values = df['hypervolume'].copy()

    # Recalculate hypervolume for each experiment
    print("\nRecalculating hypervolume for each experiment...")

    recalculated = []
    failed = []

    for idx, row in df.iterrows():
        exp_name = row['exp_name']
        exp_dir = results_dir / exp_name

        if not exp_dir.exists():
            print(f"  ⚠ Warning: Directory not found for {exp_name}")
            failed.append(exp_name)
            continue

        result = fix_hypervolume_for_experiment(exp_dir, reference_point)

        if result is None:
            failed.append(exp_name)
            continue

        # Update dataframe
        df.at[idx, 'hypervolume'] = result['new_hv']
        recalculated.append(result)

    # Print statistics
    print(f"\n{'='*70}")
    print("HYPERVOLUME FIX RESULTS")
    print(f"{'='*70}")
    print(f"Total experiments: {len(df)}")
    print(f"Successfully recalculated: {len(recalculated)}")
    print(f"Failed: {len(failed)}")

    if recalculated:
        new_hv_values = [r['new_hv'] for r in recalculated]
        old_hv_mean = old_hv_values.mean()
        new_hv_mean = np.mean(new_hv_values)

        print(f"\nHypervolume statistics:")
        print(f"  Old HV (buggy):  mean={old_hv_mean:.6f}, unique={old_hv_values.nunique()}")
        print(f"  New HV (fixed):  mean={new_hv_mean:.6f}, unique={len(set(new_hv_values))}")
        print(f"  Mean change: {abs(new_hv_mean - old_hv_mean):.6f}")

        # Show sample of changes
        print(f"\nSample of hypervolume changes (first 10):")
        print(f"{'Experiment':<30} {'Old HV':>12} {'New HV':>12} {'Difference':>12}")
        print("-" * 70)
        for result in recalculated[:10]:
            exp_name = result['exp_name'][:28]
            old_hv = result['old_hv'] or 0.0
            new_hv = result['new_hv']
            diff = result['difference'] or 0.0
            print(f"{exp_name:<30} {old_hv:>12.6f} {new_hv:>12.6f} {diff:>12.6f}")

        if len(recalculated) > 10:
            print(f"... and {len(recalculated) - 10} more")

    if failed:
        print(f"\n⚠ Failed experiments:")
        for exp_name in failed[:10]:
            print(f"  - {exp_name}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Fix hypervolume metric in ablation study results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix capacity ablation
  python scripts/fix_hypervolume_metric.py --results_dir results/ablations/capacity

  # Fix sampling ablation
  python scripts/fix_hypervolume_metric.py --results_dir results/ablations/sampling

  # Dry run to preview changes
  python scripts/fix_hypervolume_metric.py --results_dir results/ablations/capacity --dry_run
        """
    )

    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing ablation study results')
    parser.add_argument('--reference_point', type=float, nargs='+', default=None,
                       help='Reference point for hypervolume (default: auto-detect from objectives)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Show changes without saving')
    parser.add_argument('--no_backup', action='store_true',
                       help='Skip creating backup of original CSV')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"✗ Error: Results directory not found: {results_dir}")
        return 1

    print("="*70)
    print("HYPERVOLUME METRIC FIX")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Mode: {'DRY RUN (no changes saved)' if args.dry_run else 'LIVE (will update files)'}")
    print()

    # Find results CSV (check both all_results.csv and results_temp.csv)
    results_csv = results_dir / 'all_results.csv'
    if not results_csv.exists():
        results_csv = results_dir / 'results_temp.csv'
        if not results_csv.exists():
            print(f"✗ Error: No results CSV found in {results_dir}")
            print("  Expected: all_results.csv or results_temp.csv")
            return 1

    print(f"Found results file: {results_csv.name}")

    # Determine reference point
    if args.reference_point is not None:
        reference_point = np.array(args.reference_point)
        print(f"Using provided reference point: {reference_point}")
    else:
        # Auto-detect from first experiment's objectives
        print("Auto-detecting reference point from objectives...")
        exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.endswith('seed42')]
        if not exp_dirs:
            exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]

        if not exp_dirs:
            print("✗ Error: No experiment directories found")
            return 1

        objectives_file = exp_dirs[0] / 'objectives.npy'
        if not objectives_file.exists():
            print(f"✗ Error: No objectives.npy found in {exp_dirs[0].name}")
            return 1

        objectives = np.load(objectives_file)
        num_objectives = objectives.shape[1]
        reference_point = np.array([1.1] * num_objectives)
        print(f"  Detected {num_objectives} objectives")
        print(f"  Using reference point: {reference_point}")

    # Backup original CSV
    if not args.dry_run and not args.no_backup:
        backup_path = results_csv.parent / f"{results_csv.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        print(f"\nCreating backup: {backup_path.name}")
        shutil.copy2(results_csv, backup_path)

    # Fix hypervolume values
    df_updated = fix_hypervolume_in_csv(
        results_csv=results_csv,
        results_dir=results_dir,
        reference_point=reference_point,
        dry_run=args.dry_run
    )

    # Save updated CSV
    if not args.dry_run:
        print(f"\n{'='*70}")
        print("SAVING RESULTS")
        print(f"{'='*70}")
        df_updated.to_csv(results_csv, index=False)
        print(f"✓ Updated CSV saved to: {results_csv}")

        # Also update individual metrics.json files
        print(f"\nUpdating individual metrics.json files...")
        updated_count = 0
        for idx, row in df_updated.iterrows():
            exp_name = row['exp_name']
            exp_dir = results_dir / exp_name
            metrics_file = exp_dir / 'metrics.json'

            if metrics_file.exists():
                import json
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)

                metrics['hypervolume'] = float(row['hypervolume'])

                with open(metrics_file, 'w') as f:
                    json.dump(metrics, f, indent=2)

                updated_count += 1

        print(f"✓ Updated {updated_count} metrics.json files")
        print(f"\n✓ Hypervolume fix completed successfully!")
    else:
        print(f"\n{'='*70}")
        print("DRY RUN - No changes saved")
        print(f"{'='*70}")
        print("Run without --dry_run to apply changes")

    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
