#!/usr/bin/env python3
"""
Fix missing factor level columns in sequences_sampling_loss experiment.

This script:
1. Parses exp_name to extract temperature_level and loss_level
2. Updates each metrics.json file with these fields
3. Regenerates results.csv from all metrics.json files

Usage:
    python scripts/factorials/sequences/fix_missing_factor_levels.py \
        --results_dir results/factorials/sequences_sampling_loss
"""

import sys
import json
import argparse
from pathlib import Path
import pandas as pd
import re

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_exp_name(exp_name: str) -> dict:
    """
    Parse experiment name to extract factor levels.

    Examples:
        - "high_subtb_entropy_seed375" → {temperature_level: 'high', loss_level: 'subtb_entropy'}
        - "veryhigh_tb_seed42" → {temperature_level: 'very_high', loss_level: 'tb'}
        - "low_subtb_seed153" → {temperature_level: 'low', loss_level: 'subtb'}

    Args:
        exp_name: Experiment name string

    Returns:
        Dictionary with temperature_level and loss_level
    """
    # Remove seed suffix
    name_no_seed = re.sub(r'_seed\d+$', '', exp_name)

    # Parse temperature and loss
    # Temperature comes first, loss comes second
    parts = name_no_seed.split('_')

    # Handle temperature (can be 'low', 'high', 'veryhigh')
    if parts[0] == 'veryhigh':
        temperature = 'very_high'
        loss_parts = parts[1:]
    elif parts[0] in ['low', 'high']:
        temperature = parts[0]
        loss_parts = parts[1:]
    else:
        # Fallback - couldn't parse
        return {}

    # Handle loss function (can be 'tb', 'subtb', 'subtb_entropy')
    loss = '_'.join(loss_parts)

    return {
        'temperature_level': temperature,
        'loss_level': loss
    }


def fix_metrics_file(metrics_path: Path, dry_run: bool = False) -> bool:
    """
    Fix a single metrics.json file by adding missing factor levels.

    Args:
        metrics_path: Path to metrics.json file
        dry_run: If True, don't actually save changes

    Returns:
        True if file was updated, False otherwise
    """
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        exp_name = metrics.get('exp_name', '')
        if not exp_name:
            print(f"⚠️  No exp_name in {metrics_path}")
            return False

        # Check if factor levels already exist
        has_temp = 'temperature_level' in metrics and metrics['temperature_level'] is not None
        has_loss = 'loss_level' in metrics and metrics['loss_level'] is not None

        if has_temp and has_loss:
            # Already has both factor levels
            return False

        # Parse exp_name to get factor levels
        factor_levels = parse_exp_name(exp_name)

        if not factor_levels:
            print(f"⚠️  Could not parse: {exp_name}")
            return False

        # Update metrics
        metrics['temperature_level'] = factor_levels['temperature_level']
        metrics['loss_level'] = factor_levels['loss_level']

        if not dry_run:
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)

        return True

    except Exception as e:
        print(f"✗ Error processing {metrics_path}: {e}")
        return False


def regenerate_results_csv(results_dir: Path, dry_run: bool = False):
    """
    Regenerate results.csv from all metrics.json files.

    Args:
        results_dir: Directory containing experiment subdirectories
        dry_run: If True, don't actually save the CSV
    """
    # Find all metrics.json files
    metrics_files = sorted(results_dir.glob('*/metrics.json'))

    if not metrics_files:
        print("No metrics.json files found!")
        return

    print(f"\nRegenerating results.csv from {len(metrics_files)} metrics files...")

    # Load all metrics
    all_metrics = []
    for metrics_file in metrics_files:
        try:
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        except Exception as e:
            print(f"⚠️  Could not load {metrics_file}: {e}")

    # Create DataFrame
    df = pd.DataFrame(all_metrics)

    # Sort by exp_name
    if 'exp_name' in df.columns:
        df = df.sort_values('exp_name')

    print(f"\nResulting DataFrame:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")

    # Check factor level coverage
    if 'temperature_level' in df.columns and 'loss_level' in df.columns:
        print(f"\nFactor level coverage:")
        print(f"  temperature_level NaN: {df['temperature_level'].isna().sum()}")
        print(f"  loss_level NaN: {df['loss_level'].isna().sum()}")

        print(f"\nFactor combinations:")
        combos = df.groupby(['temperature_level', 'loss_level']).size()
        print(combos)

    # Save
    if not dry_run:
        csv_path = results_dir / 'results.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results.csv to: {csv_path}")
    else:
        print(f"\n[DRY RUN] Would save results.csv")


def main():
    parser = argparse.ArgumentParser(
        description='Fix missing factor level columns in factorial experiment data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        default=Path('results/factorials/sequences_sampling_loss'),
        help='Results directory containing experiment subdirectories'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without modifying files'
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Directory does not exist: {args.results_dir}")
        sys.exit(1)

    print("="*80)
    print("FIXING MISSING FACTOR LEVELS")
    print("="*80)
    print(f"Results directory: {args.results_dir}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")
    print()

    # Find all metrics.json files
    metrics_files = sorted(args.results_dir.glob('*/metrics.json'))
    print(f"Found {len(metrics_files)} metrics.json files\n")

    # Fix each file
    num_updated = 0
    num_skipped = 0

    for metrics_file in metrics_files:
        was_updated = fix_metrics_file(metrics_file, dry_run=args.dry_run)
        if was_updated:
            num_updated += 1
            exp_name = metrics_file.parent.name
            print(f"  ✓ Updated: {exp_name}")
        else:
            num_skipped += 1

    print(f"\nSummary:")
    print(f"  Updated: {num_updated}")
    print(f"  Skipped (already had factor levels): {num_skipped}")

    # Regenerate results.csv
    regenerate_results_csv(args.results_dir, dry_run=args.dry_run)

    print("\n✓ Done!")


if __name__ == '__main__':
    main()
