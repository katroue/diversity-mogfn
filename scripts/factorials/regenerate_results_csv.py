#!/usr/bin/env python3
"""
Regenerate results.csv from individual metrics.json files.

This script reconstructs the results.csv file for factorial experiments by reading
all metrics.json files from experiment directories. This is useful when:
- results.csv has corrupted data
- Metrics were updated in metrics.json but not propagated to results.csv
- You need to verify data integrity

Usage:
    python scripts/factorials/regenerate_results_csv.py --results_dir results/factorials/sequences_sampling_loss
    python scripts/factorials/regenerate_results_csv.py --results_dir results/factorials --recursive
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def parse_experiment_name(exp_name: str) -> Dict[str, str]:
    """
    Parse experiment name to extract condition information.

    Examples:
        'low_tb_seed42' -> {'temperature_level': 'low', 'loss_level': 'tb', 'seed': 42}
        'medium_high_seed153' -> {'capacity_level': 'medium', 'temperature_level': 'high', 'seed': 153}
        'high_subtb_entropy_seed264' -> {'temperature_level': 'high', 'loss_level': 'subtb_entropy', 'seed': 264}
    """
    conditions = {}

    # Extract seed (always at the end)
    seed_match = re.search(r'seed(\d+)$', exp_name)
    if seed_match:
        conditions['seed'] = int(seed_match.group(1))
        # Remove seed from name for further parsing
        name_without_seed = exp_name[:seed_match.start()].rstrip('_')
    else:
        name_without_seed = exp_name

    # Split by underscore
    parts = name_without_seed.split('_')

    # Capacity levels
    capacity_levels = {'small', 'medium', 'large', 'xlarge'}
    # Temperature levels
    temp_levels = {'low', 'high', 'veryhigh', 'very_high'}
    # Loss functions
    loss_levels = {'tb', 'subtb', 'subtb_entropy', 'flowmatching', 'fm'}

    # Parse parts
    remaining_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]

        # Check for capacity
        if part in capacity_levels:
            conditions['capacity_level'] = part
        # Check for temperature (including very_high as two parts)
        elif part == 'very' and i + 1 < len(parts) and parts[i + 1] == 'high':
            conditions['temperature_level'] = 'very_high'
            i += 1  # Skip next part
        elif part == 'veryhigh':
            conditions['temperature_level'] = 'very_high'
        elif part in temp_levels:
            conditions['temperature_level'] = part
        # Check for loss (including subtb_entropy as two parts)
        elif part == 'subtb' and i + 1 < len(parts) and parts[i + 1] == 'entropy':
            conditions['loss_level'] = 'subtb_entropy'
            i += 1  # Skip next part
        elif part in loss_levels:
            conditions['loss_level'] = part
        else:
            remaining_parts.append(part)

        i += 1

    # Store original experiment name
    conditions['exp_name'] = exp_name

    # Create condition_name (name without seed)
    conditions['condition_name'] = name_without_seed

    return conditions


def load_metrics_from_json(metrics_file: Path) -> Dict:
    """Load metrics from a metrics.json file."""
    with open(metrics_file, 'r') as f:
        return json.load(f)


def collect_experiment_data(results_dir: Path) -> List[Dict]:
    """
    Collect all experiment data from metrics.json files.

    Args:
        results_dir: Directory containing experiment subdirectories

    Returns:
        List of dictionaries, each containing metrics and conditions for one experiment
    """
    data = []

    # Find all experiment directories (those containing metrics.json)
    experiment_dirs = []
    for path in results_dir.iterdir():
        if path.is_dir() and (path / 'metrics.json').exists():
            experiment_dirs.append(path)

    if not experiment_dirs:
        print(f"Warning: No experiment directories found in {results_dir}")
        return data

    print(f"Found {len(experiment_dirs)} experiment directories")

    for exp_dir in sorted(experiment_dirs):
        exp_name = exp_dir.name
        metrics_file = exp_dir / 'metrics.json'

        try:
            # Load metrics
            metrics = load_metrics_from_json(metrics_file)

            # Parse experiment conditions from directory name
            conditions = parse_experiment_name(exp_name)

            # Combine metrics and conditions
            row = {**metrics, **conditions}
            data.append(row)

        except Exception as e:
            print(f"Error processing {exp_dir.name}: {e}")
            continue

    return data


def regenerate_results_csv(results_dir: Path, output_file: Path = None, backup: bool = True) -> pd.DataFrame:
    """
    Regenerate results.csv from metrics.json files.

    Args:
        results_dir: Directory containing experiment subdirectories
        output_file: Path to save results.csv (default: results_dir/results.csv)
        backup: Whether to backup existing results.csv before overwriting

    Returns:
        DataFrame containing all experiment results
    """
    if output_file is None:
        output_file = results_dir / 'results.csv'

    # Backup existing file if requested
    if backup and output_file.exists():
        backup_file = output_file.with_suffix('.csv.backup')
        print(f"Backing up existing results.csv to {backup_file}")
        output_file.rename(backup_file)

    # Collect data from all experiments
    print(f"\nCollecting data from {results_dir}...")
    data = collect_experiment_data(results_dir)

    if not data:
        print("No data collected. Exiting.")
        return None

    # Create DataFrame
    df = pd.DataFrame(data)

    # Reorder columns: metrics first, then condition columns
    metric_cols = [col for col in df.columns if col not in
                   {'seed', 'exp_name', 'condition_name', 'capacity_level',
                    'temperature_level', 'loss_level'}]
    condition_cols = [col for col in ['seed', 'exp_name', 'condition_name',
                                      'capacity_level', 'temperature_level', 'loss_level']
                     if col in df.columns]

    df = df[metric_cols + condition_cols]

    # Sort by condition columns
    sort_cols = [col for col in condition_cols if col != 'exp_name']
    df = df.sort_values(sort_cols, ignore_index=True)

    # Save to CSV
    print(f"\nSaving results to {output_file}")
    df.to_csv(output_file, index=False)

    # Print summary
    print(f"\n✓ Regenerated results.csv with {len(df)} experiments")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())

    # Print validation info
    if 'num_modes' in df.columns:
        print(f"\nnum_modes statistics:")
        print(df['num_modes'].describe())

    return df


def process_recursive(base_dir: Path, pattern: str = '**/results.csv'):
    """
    Recursively process all factorial experiment directories.

    Args:
        base_dir: Base directory to search
        pattern: Pattern to find results directories
    """
    # Find all directories that either contain results.csv or have experiment subdirs
    results_dirs = []

    for path in base_dir.rglob('experiment_config.yaml'):
        results_dir = path.parent
        if results_dir not in results_dirs:
            results_dirs.append(results_dir)

    print(f"Found {len(results_dirs)} factorial experiment directories:\n")
    for rd in results_dirs:
        print(f"  - {rd.relative_to(base_dir)}")

    print("\n" + "="*80 + "\n")

    for results_dir in results_dirs:
        print(f"\nProcessing: {results_dir.relative_to(base_dir)}")
        print("="*80)
        regenerate_results_csv(results_dir)
        print("\n")


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate results.csv from metrics.json files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Regenerate for a single factorial experiment
  python scripts/factorials/regenerate_results_csv.py \\
      --results_dir results/factorials/sequences_sampling_loss

  # Regenerate for all factorial experiments recursively
  python scripts/factorials/regenerate_results_csv.py \\
      --results_dir results/factorials \\
      --recursive

  # Don't backup existing file
  python scripts/factorials/regenerate_results_csv.py \\
      --results_dir results/factorials/sequences_sampling_loss \\
      --no-backup
        """
    )

    parser.add_argument(
        '--results_dir',
        type=Path,
        required=True,
        help='Directory containing experiment subdirectories with metrics.json files'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output CSV file path (default: results_dir/results.csv)'
    )

    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Recursively process all factorial experiments under results_dir'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Do not backup existing results.csv file'
    )

    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"Error: Directory {args.results_dir} does not exist")
        return 1

    if args.recursive:
        process_recursive(args.results_dir)
    else:
        regenerate_results_csv(
            args.results_dir,
            output_file=args.output,
            backup=not args.no_backup
        )

    return 0


if __name__ == '__main__':
    exit(main())
