#!/usr/bin/env python3
"""
Recompute MCE (Mode Coverage Entropy) for all existing experiments.

This script fixes the MCE metric for all baseline, ablation, and factorial
experiments by using the corrected auto-tuning algorithm that avoids outliers.

The original MCE implementation had a bug where the elbow detection could pick
outliers in the k-distance distribution, leading to incorrect eps values and
mode collapse detection (e.g., seed 153 incorrectly reported MCE=0, modes=1).

Usage:
    # Dry run (preview changes without saving)
    python scripts/fix_mce_all_results.py --dry_run

    # Apply fixes
    python scripts/fix_mce_all_results.py

    # Fix specific directory only
    python scripts/fix_mce_all_results.py --results_dir results/baselines/ngrams
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Set
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.spatial import mode_coverage_entropy

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def find_all_experiments(base_dir: Path) -> List[Path]:
    """
    Find all experiment directories with metrics.json and objectives.npy.

    Args:
        base_dir: Base directory to search (e.g., results/)

    Returns:
        List of paths to experiment directories
    """
    experiments = []

    # Find all metrics.json files
    for metrics_file in base_dir.rglob('metrics.json'):
        exp_dir = metrics_file.parent

        # Check if objectives.npy exists
        objectives_file = exp_dir / 'objectives.npy'
        if objectives_file.exists():
            experiments.append(exp_dir)

    return sorted(experiments)


def recompute_mce_for_experiment(
    exp_dir: Path,
    eps: str = 'auto',
    min_samples: int = 5,
    max_samples_for_metrics: int = 5000
) -> Tuple[float, int, float, int]:
    """
    Recompute MCE for a single experiment.

    Args:
        exp_dir: Experiment directory path
        eps: DBSCAN eps parameter ('auto' for auto-tuning)
        min_samples: DBSCAN min_samples parameter
        max_samples_for_metrics: Maximum samples to use for MCE computation (default: 5000)

    Returns:
        Tuple of (old_mce, old_modes, new_mce, new_modes)
    """
    # Load current metrics
    metrics_file = exp_dir / 'metrics.json'
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    old_mce = metrics.get('mce', None)
    old_modes = metrics.get('num_modes', None)

    # Get seed for reproducible subsampling (if available)
    seed = metrics.get('seed', None)

    # Load objectives
    objectives_file = exp_dir / 'objectives.npy'
    objectives = np.load(objectives_file)

    # Subsample if dataset is too large (for computational efficiency)
    original_size = len(objectives)
    if original_size > max_samples_for_metrics:
        logger.info(f"  Subsampling {original_size} → {max_samples_for_metrics} for MCE computation")
        rng = np.random.RandomState(seed) if seed is not None else np.random
        indices = rng.choice(original_size, size=max_samples_for_metrics, replace=False)
        objectives = objectives[indices]

    # Recompute MCE with fixed implementation
    new_mce, new_modes = mode_coverage_entropy(objectives, eps=eps, min_samples=min_samples)

    return old_mce, old_modes, new_mce, new_modes


def update_metrics_file(exp_dir: Path, new_mce: float, new_modes: int) -> None:
    """
    Update metrics.json with corrected MCE values.

    Args:
        exp_dir: Experiment directory path
        new_mce: New MCE value
        new_modes: New number of modes
    """
    metrics_file = exp_dir / 'metrics.json'

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    # Update values
    metrics['mce'] = new_mce
    metrics['num_modes'] = new_modes

    # Save updated metrics
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)


def find_csv_files(base_dir: Path) -> List[Path]:
    """
    Find all CSV files in the results directory.

    Args:
        base_dir: Base directory to search

    Returns:
        List of CSV file paths
    """
    csv_files = []
    for csv_file in base_dir.rglob('*.csv'):
        csv_files.append(csv_file)
    return sorted(csv_files)


def update_csv_file(
    csv_file: Path,
    updates: Dict[Tuple[int, str], Tuple[float, int]]
) -> int:
    """
    Update MCE values in a CSV file.

    Args:
        csv_file: Path to CSV file
        updates: Dictionary mapping (seed, algorithm) -> (new_mce, new_modes)

    Returns:
        Number of rows updated
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Check if MCE columns exist
        if 'mce' not in df.columns or 'num_modes' not in df.columns:
            return 0

        # Check if required columns exist for matching
        if 'seed' not in df.columns:
            logger.warning(f"  {csv_file.name}: No 'seed' column, skipping")
            return 0
        if 'algorithm' not in df.columns:
            logger.warning(f"  {csv_file.name}: No 'algorithm' column, skipping")
            return 0

        # Update rows
        rows_updated = 0
        for idx, row in df.iterrows():
            seed = row['seed']
            algorithm = row['algorithm']
            key = (seed, algorithm)
            if key in updates:
                new_mce, new_modes = updates[key]
                df.at[idx, 'mce'] = new_mce
                df.at[idx, 'num_modes'] = new_modes
                rows_updated += 1

        # Save if any updates were made
        if rows_updated > 0:
            df.to_csv(csv_file, index=False)
            logger.info(f"  Updated {csv_file.name}: {rows_updated} rows")

        return rows_updated

    except Exception as e:
        logger.error(f"  Error updating {csv_file}: {e}")
        return 0


def collect_updates_by_seed_and_algo(
    experiments: List[Tuple[Path, float, int, float, int]]
) -> Dict[Tuple[int, str], Tuple[float, int]]:
    """
    Collect MCE updates grouped by (seed, algorithm).

    Args:
        experiments: List of (exp_dir, old_mce, old_modes, new_mce, new_modes)

    Returns:
        Dictionary mapping (seed, algorithm) -> (new_mce, new_modes)
    """
    updates = {}

    for exp_dir, old_mce, old_modes, new_mce, new_modes in experiments:
        # Try to extract seed and algorithm from metrics.json
        try:
            metrics_file = exp_dir / 'metrics.json'
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                seed = metrics.get('seed')
                algorithm = metrics.get('algorithm')
                if seed is not None and algorithm is not None:
                    updates[(seed, algorithm)] = (new_mce, new_modes)
        except Exception:
            continue

    return updates


def main():
    parser = argparse.ArgumentParser(
        description='Recompute MCE for all experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run (preview changes)
  python scripts/fix_mce_all_results.py --dry_run

  # Fix all results
  python scripts/fix_mce_all_results.py

  # Fix specific directory
  python scripts/fix_mce_all_results.py --results_dir results/baselines/ngrams

  # Use fixed eps instead of auto-tuning
  python scripts/fix_mce_all_results.py --eps 0.1
        """
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Base directory containing experiment results (default: results/)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without saving'
    )
    parser.add_argument(
        '--eps',
        type=str,
        default='auto',
        help='DBSCAN eps parameter (default: auto)'
    )
    parser.add_argument(
        '--min_samples',
        type=int,
        default=5,
        help='DBSCAN min_samples parameter (default: 5)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=5000,
        help='Maximum samples for MCE computation (default: 5000, prevents slowdown on large datasets)'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    logger.info(f"Searching for experiments in: {results_dir}")
    logger.info(f"MCE parameters: eps={args.eps}, min_samples={args.min_samples}")

    if args.dry_run:
        logger.info("DRY RUN MODE: No files will be modified")

    # Find all experiments
    experiments = find_all_experiments(results_dir)
    logger.info(f"Found {len(experiments)} experiments with objectives.npy")

    # Track changes
    total_changed = 0
    total_unchanged = 0
    significant_changes = []  # Changes where old_mce=0 or diff > 0.1
    all_changes = []  # Store all changes for CSV updates

    # Process each experiment
    for i, exp_dir in enumerate(experiments, 1):
        try:
            # Recompute MCE
            old_mce, old_modes, new_mce, new_modes = recompute_mce_for_experiment(
                exp_dir,
                eps=args.eps,
                min_samples=args.min_samples,
                max_samples_for_metrics=args.max_samples
            )

            # Compute change
            mce_diff = abs(new_mce - old_mce) if old_mce is not None else 0
            modes_diff = abs(new_modes - old_modes) if old_modes is not None else 0

            # Determine if changed
            changed = (mce_diff > 1e-6) or (modes_diff > 0)

            if changed:
                total_changed += 1
                # Store for CSV updates
                all_changes.append((exp_dir, old_mce, old_modes, new_mce, new_modes))
            else:
                total_unchanged += 1

            # Track significant changes
            if old_mce == 0.0 or mce_diff > 0.1:
                significant_changes.append({
                    'exp_dir': exp_dir,
                    'old_mce': old_mce,
                    'old_modes': old_modes,
                    'new_mce': new_mce,
                    'new_modes': new_modes,
                    'mce_diff': mce_diff,
                    'modes_diff': modes_diff
                })

            # Log progress
            if changed:
                logger.info(
                    f"[{i}/{len(experiments)}] {exp_dir.relative_to(results_dir)}: "
                    f"MCE {old_mce:.4f}→{new_mce:.4f}, modes {old_modes}→{new_modes}"
                )
            else:
                if i % 10 == 0:  # Log every 10 unchanged
                    logger.debug(
                        f"[{i}/{len(experiments)}] {exp_dir.relative_to(results_dir)}: "
                        f"MCE unchanged ({new_mce:.4f})"
                    )

            # Update file if not dry run
            if not args.dry_run and changed:
                update_metrics_file(exp_dir, new_mce, new_modes)

        except Exception as e:
            logger.error(f"Error processing {exp_dir}: {e}")
            continue

    # Update CSV files
    if all_changes and not args.dry_run:
        logger.info("\n" + "=" * 80)
        logger.info("UPDATING CSV FILES")
        logger.info("=" * 80)

        # Collect updates by seed
        updates_by_seed = collect_updates_by_seed_and_algo(all_changes)
        logger.info(f"Collected updates for {len(updates_by_seed)} seeds")

        # Find all CSV files
        csv_files = find_csv_files(results_dir)
        logger.info(f"Found {len(csv_files)} CSV files")

        total_csv_rows_updated = 0
        csv_files_updated = 0

        for csv_file in csv_files:
            rows_updated = update_csv_file(csv_file, updates_by_seed)
            if rows_updated > 0:
                total_csv_rows_updated += rows_updated
                csv_files_updated += 1

        logger.info(f"\n✓ Updated {csv_files_updated} CSV files ({total_csv_rows_updated} total rows)")

    elif all_changes and args.dry_run:
        logger.info("\n[DRY RUN] Would update CSV files with new MCE values")
        updates_by_seed = collect_updates_by_seed_and_algo(all_changes)
        logger.info(f"  Would update {len(updates_by_seed)} unique seeds in CSV files")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total experiments processed: {len(experiments)}")
    logger.info(f"Changed: {total_changed}")
    logger.info(f"Unchanged: {total_unchanged}")

    if significant_changes:
        logger.info(f"\n{len(significant_changes)} experiments with significant changes:")
        logger.info("=" * 80)

        for change in significant_changes[:20]:  # Show first 20
            exp_name = change['exp_dir'].relative_to(results_dir)
            logger.info(
                f"{exp_name}:\n"
                f"  Old: MCE={change['old_mce']:.4f}, modes={change['old_modes']}\n"
                f"  New: MCE={change['new_mce']:.4f}, modes={change['new_modes']}\n"
                f"  Diff: ΔMCE={change['mce_diff']:.4f}, Δmodes={change['modes_diff']}"
            )

        if len(significant_changes) > 20:
            logger.info(f"... and {len(significant_changes) - 20} more")

    if args.dry_run:
        logger.info("\nDRY RUN: No files were modified. Run without --dry_run to apply changes.")
    else:
        logger.info(f"\n✓ Updated {total_changed} experiments")

    logger.info("\nDone!")


if __name__ == '__main__':
    main()
