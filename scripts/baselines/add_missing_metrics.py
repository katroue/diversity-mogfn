#!/usr/bin/env python3
"""
Add missing metrics to baseline experiment results.

This script:
1. Scans all experiment directories in results/baselines/*
2. Identifies missing metrics (GD, IGD)
3. Computes empirical Pareto front from all experiments in each task
4. Adds GD/IGD metrics to individual metrics.json files
5. Regenerates all_results.csv with the new metrics

Usage:
    python scripts/baselines/add_missing_metrics.py --results_dir results/baselines/molecules
    python scripts/baselines/add_missing_metrics.py --results_dir results/baselines  # Process all subdirs
"""

import sys
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.metrics.traditional import generational_distance, inverted_generational_distance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_pareto_front(objectives: np.ndarray) -> np.ndarray:
    """
    Compute Pareto front from objectives.

    A solution is non-dominated if no other solution is better in all objectives
    and strictly better in at least one objective.

    Args:
        objectives: Objective values, shape (N, num_objectives)

    Returns:
        Pareto front subset of objectives
    """
    objectives = np.atleast_2d(objectives)
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i != j:
                # j dominates i if: j <= i in all objectives AND j < i in at least one
                if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                    is_dominated[i] = True
                    break

    pareto_front = objectives[~is_dominated]
    return pareto_front


def find_experiment_dirs(base_dir: Path) -> List[Path]:
    """Find all experiment directories (e.g., random_seed42, nsga2_seed153)."""
    experiment_dirs = []

    # Pattern: algorithm_seedXXX
    for item in base_dir.iterdir():
        if item.is_dir() and '_seed' in item.name:
            if (item / 'metrics.json').exists():
                experiment_dirs.append(item)

    return sorted(experiment_dirs)


def load_objectives_from_experiment(exp_dir: Path) -> Optional[np.ndarray]:
    """Load objectives from experiment directory."""
    objectives_path = exp_dir / 'objectives.npy'
    if objectives_path.exists():
        return np.load(objectives_path)
    return None


def compute_empirical_pareto_front(all_objectives: List[np.ndarray], max_samples: int = 10000) -> np.ndarray:
    """
    Compute empirical Pareto front from all objectives across experiments.

    Args:
        all_objectives: List of objective arrays from different experiments
        max_samples: Maximum number of samples to use for Pareto front computation
                    (prevents O(N²) explosion on large datasets)

    Returns:
        Pareto front array
    """
    # Concatenate all objectives
    combined = np.vstack(all_objectives)

    # Subsample if dataset is too large (for computational efficiency)
    original_size = len(combined)
    if original_size > max_samples:
        logger.info(f"  Subsampling {original_size:,} → {max_samples:,} for Pareto front computation")
        rng = np.random.RandomState(42)  # Reproducible subsampling
        indices = rng.choice(original_size, size=max_samples, replace=False)
        combined = combined[indices]

    # Remove duplicates
    combined_unique = np.unique(combined, axis=0)

    # Compute Pareto front
    pareto_front = compute_pareto_front(combined_unique)

    logger.info(f"  Empirical Pareto front: {len(pareto_front)} solutions from {original_size:,} total")

    return pareto_front


def add_convergence_metrics(
    exp_dir: Path,
    reference_pareto_front: np.ndarray,
    objectives: np.ndarray
) -> Dict[str, float]:
    """
    Compute and add GD/IGD metrics to experiment.

    Args:
        exp_dir: Experiment directory
        reference_pareto_front: Reference Pareto front for comparison
        objectives: Experiment objectives

    Returns:
        Dictionary with new metrics
    """
    new_metrics = {}

    # Compute GD
    gd = generational_distance(objectives, reference_pareto_front)
    new_metrics['gd'] = float(gd)

    # Compute IGD
    igd = inverted_generational_distance(objectives, reference_pareto_front)
    new_metrics['igd'] = float(igd)

    logger.info(f"    {exp_dir.name}: GD={gd:.6f}, IGD={igd:.6f}")

    return new_metrics


def update_metrics_json(exp_dir: Path, new_metrics: Dict[str, float]):
    """Update metrics.json file with new metrics."""
    metrics_path = exp_dir / 'metrics.json'

    # Load existing metrics
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Add new metrics
    metrics.update(new_metrics)

    # Save updated metrics
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def regenerate_all_results_csv(base_dir: Path, experiment_dirs: List[Path]):
    """Regenerate all_results.csv from individual metrics.json files."""
    all_results = []

    for exp_dir in experiment_dirs:
        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        all_results.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(all_results)

    # Sort by algorithm and seed
    if 'algorithm' in df.columns and 'seed' in df.columns:
        df = df.sort_values(['algorithm', 'seed'])

    # Save to CSV
    output_path = base_dir / 'all_results.csv'
    df.to_csv(output_path, index=False)
    logger.info(f"Regenerated {output_path}")

    return df


def regenerate_summary_by_algorithm(base_dir: Path, df: pd.DataFrame):
    """
    Regenerate summary_by_algorithm.csv from all_results DataFrame.

    Args:
        base_dir: Base directory for results
        df: DataFrame with all results
    """
    if 'algorithm' not in df.columns:
        logger.warning("No 'algorithm' column found, skipping summary generation")
        return

    # Define key metrics to summarize
    # Include all numeric metrics that are meaningful to aggregate
    metrics_to_summarize = [
        'hypervolume', 'r2_indicator', 'avg_pairwise_distance', 'spacing', 'spread',
        'gd', 'igd',  # Newly added metrics
        'mce', 'pmd', 'pfs', 'num_unique_solutions',
        'qds',
        'training_time'
    ]

    # Filter to only metrics that exist in the DataFrame
    available_metrics = [m for m in metrics_to_summarize if m in df.columns]

    if not available_metrics:
        logger.warning("No metrics to summarize")
        return

    # Create aggregation dict
    agg_dict = {metric: ['mean', 'std'] for metric in available_metrics}

    # Compute summary
    summary = df.groupby('algorithm').agg(agg_dict)

    # Save to CSV
    output_path = base_dir / 'summary_by_algorithm.csv'
    summary.to_csv(output_path)
    logger.info(f"Regenerated {output_path}")


def process_task_directory(task_dir: Path, use_empirical_pf: bool = True):
    """
    Process a single task directory (e.g., results/baselines/hypergrid).

    Args:
        task_dir: Task directory path
        use_empirical_pf: Use empirical Pareto front from all experiments
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {task_dir}")
    logger.info(f"{'='*70}")

    # Find all experiment directories
    experiment_dirs = find_experiment_dirs(task_dir)

    if not experiment_dirs:
        logger.warning(f"No experiment directories found in {task_dir}")
        return

    logger.info(f"Found {len(experiment_dirs)} experiment directories")

    # Load all objectives
    all_objectives = []
    objectives_by_exp = {}

    for exp_dir in experiment_dirs:
        objectives = load_objectives_from_experiment(exp_dir)
        if objectives is not None:
            all_objectives.append(objectives)
            objectives_by_exp[exp_dir] = objectives
        else:
            logger.warning(f"No objectives.npy found in {exp_dir}")

    if not all_objectives:
        logger.warning(f"No objectives found in {task_dir}")
        return

    # Compute reference Pareto front
    if use_empirical_pf:
        logger.info("Computing empirical Pareto front from all experiments...")
        reference_pf = compute_empirical_pareto_front(all_objectives)
    else:
        logger.warning("No true Pareto front available, skipping GD/IGD")
        return

    # Process each experiment
    logger.info("\nAdding GD/IGD metrics to experiments...")
    experiments_updated = 0

    for exp_dir in experiment_dirs:
        if exp_dir not in objectives_by_exp:
            continue

        objectives = objectives_by_exp[exp_dir]

        # Check if metrics already exist
        metrics_path = exp_dir / 'metrics.json'
        with open(metrics_path, 'r') as f:
            existing_metrics = json.load(f)

        if 'gd' in existing_metrics and 'igd' in existing_metrics:
            logger.debug(f"  {exp_dir.name}: GD/IGD already exist, skipping")
            continue

        # Compute new metrics
        new_metrics = add_convergence_metrics(exp_dir, reference_pf, objectives)

        # Update metrics.json
        update_metrics_json(exp_dir, new_metrics)
        experiments_updated += 1

    logger.info(f"\nUpdated {experiments_updated} experiments with GD/IGD metrics")

    # Regenerate all_results.csv
    logger.info("\nRegenerating all_results.csv...")
    df = regenerate_all_results_csv(task_dir, experiment_dirs)

    # Regenerate summary_by_algorithm.csv
    logger.info("Regenerating summary_by_algorithm.csv...")
    regenerate_summary_by_algorithm(task_dir, df)

    # Show summary statistics
    if 'gd' in df.columns and 'igd' in df.columns:
        logger.info("\nGD/IGD Summary by Algorithm:")
        if 'algorithm' in df.columns:
            summary = df.groupby('algorithm')[['gd', 'igd']].agg(['mean', 'std', 'count'])
            print(summary)

    logger.info(f"\n✓ Completed processing {task_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Add missing metrics (GD, IGD) to baseline experiment results'
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Results directory to process (e.g., results/baselines/hypergrid or results/baselines)'
    )
    parser.add_argument(
        '--use_empirical_pf',
        action='store_true',
        default=True,
        help='Use empirical Pareto front from all experiments (default: True)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process all subdirectories recursively'
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Directory not found: {results_dir}")
        return 1

    if args.recursive:
        # Process all subdirectories that look like task directories
        task_dirs = []
        for item in results_dir.iterdir():
            if item.is_dir():
                # Check if it has experiment directories
                exp_dirs = find_experiment_dirs(item)
                if exp_dirs:
                    task_dirs.append(item)

        if not task_dirs:
            logger.warning(f"No task directories found in {results_dir}")
            return 1

        logger.info(f"Found {len(task_dirs)} task directories to process")

        for task_dir in task_dirs:
            try:
                process_task_directory(task_dir, args.use_empirical_pf)
            except Exception as e:
                logger.error(f"Error processing {task_dir}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Process single directory
        try:
            process_task_directory(results_dir, args.use_empirical_pf)
        except Exception as e:
            logger.error(f"Error processing {results_dir}: {e}")
            import traceback
            traceback.print_exc()
            return 1

    logger.info("\n" + "="*70)
    logger.info("✅ All processing complete!")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
