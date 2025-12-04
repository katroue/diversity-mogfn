#!/usr/bin/env python3
"""
Script to add missing metrics results from experiment directories to results_temp.csv.

This script:
1. Scans all experiment directories in results/factorials/capacity_sampling/
2. Identifies which experiments are missing from results_temp.csv
3. Reads metrics.json and config.json from each missing experiment
4. Appends the missing results to results_temp.csv
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Set


def load_existing_experiments(csv_path: Path) -> Set[str]:
    """Load the list of experiments already in the CSV file."""
    existing_experiments = set()

    if csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_experiments.add(row['exp_name'])

    return existing_experiments


def get_all_experiment_dirs(results_dir: Path) -> List[Path]:
    """Get all experiment directories from the results directory."""
    experiment_dirs = []

    # Look for directories matching the pattern: {capacity}_{temperature}_seed{seed}
    for item in results_dir.iterdir():
        if item.is_dir() and any(
            item.name.startswith(prefix)
            for prefix in ['small_', 'medium_', 'large_',]
        ):
            experiment_dirs.append(item)

    return sorted(experiment_dirs)


def load_experiment_data(exp_dir: Path) -> Dict:
    """Load metrics and config data from an experiment directory."""
    metrics_file = exp_dir / 'metrics.json'
    #config_file = exp_dir / 'config.json'

    if not metrics_file.exists():
        raise FileNotFoundError(f"metrics.json not found in {exp_dir}")
    #if not config_file.exists():
    #    raise FileNotFoundError(f"config.json not found in {exp_dir}")

    with open(metrics_file, 'r') as f:
        metrics = json.load(f)

    #with open(config_file, 'r') as f:
    #    config = json.load(f)

    # Combine data from both files
    # The CSV expects these columns in this order:
    # hypervolume,r2_indicator,avg_pairwise_distance,spacing,spread,tds,mpd,mce,
    # num_modes,pmd,pfs,pas,rbd,fci,qds,der,num_parameters,training_time,
    # final_loss,seed,exp_name,condition_name,capacity_level,temperature_level

    row_data = {
        # Metrics from metrics.json
        'hypervolume': metrics['hypervolume'],
        'r2_indicator': metrics['r2_indicator'],
        'avg_pairwise_distance': metrics['avg_pairwise_distance'],
        'spacing': metrics['spacing'],
        'spread': metrics['spread'],
        'tds': metrics['tds'],
        'mpd': metrics['mpd'],
        'mce': metrics['mce'],
        'num_modes': metrics['num_modes'],
        'pmd': metrics['pmd'],
        'pfs': metrics['pfs'],
        'pas': metrics['pas'],
        'rbd': metrics['rbd'],
        'fci': metrics['fci'],
        'qds': metrics['qds'],
        'der': metrics['der'],
        'num_parameters': metrics['num_parameters'],
        'training_time': metrics['training_time'],
        'final_loss': metrics['final_loss'],
        'seed': metrics['seed'],
        'exp_name': metrics['exp_name'],
        # Config data
        'capacity': metrics['capacity'],
        'hidden_dim': metrics['hidden_dim'],
        'num_layers': metrics['num_layers'],
        'conditioning': metrics['conditioning'],
        'alpha': metrics['alpha'],
        'preference_sampling': metrics['preference_sampling'],
        'loss': metrics['loss'],
    }

    return row_data


def append_to_csv(csv_path: Path, rows: List[Dict]):
    """Append rows to the CSV file."""
    if not rows:
        print("No rows to append.")
        return

    # Define the column order
    fieldnames = [
        'hypervolume', 'r2_indicator', 'avg_pairwise_distance', 'spacing', 'spread',
        'tds', 'mpd', 'mce', 'num_modes', 'pmd', 'pfs', 'pas', 'rbd', 'fci', 'qds', 'der',
        'num_parameters', 'training_time', 'final_loss', 'seed', 'exp_name',
        'capacity', 'hidden_dim','num_layers', 'conditioning', 'alpha', 'preference_sampling', 'loss'
    ]

    # Append to CSV
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        for row in rows:
            writer.writerow(row)


def main():
    # Set up paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    results_dir = project_root / 'results' / 'ablations' / 'capacity'
    csv_path = results_dir / 'all_results.csv'

    print(f"Results directory: {results_dir}")
    print(f"CSV file: {csv_path}")

    # Load existing experiments from CSV
    existing_experiments = load_existing_experiments(csv_path)
    print(f"\nFound {len(existing_experiments)} experiments in CSV")

    # Get all experiment directories
    experiment_dirs = get_all_experiment_dirs(results_dir)
    print(f"Found {len(experiment_dirs)} experiment directories")

    # Find missing experiments
    missing_experiments = []
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        if exp_name not in existing_experiments:
            missing_experiments.append(exp_dir)

    print(f"\nFound {len(missing_experiments)} missing experiments:")
    for exp_dir in missing_experiments:
        print(f"  - {exp_dir.name}")

    if not missing_experiments:
        print("\nNo missing experiments found. CSV is up to date!")
        return

    # Load data from missing experiments
    print("\nLoading data from missing experiments...")
    rows_to_add = []

    for exp_dir in missing_experiments:
        try:
            row_data = load_experiment_data(exp_dir)
            rows_to_add.append(row_data)
            print(f"  ✓ Loaded {exp_dir.name}")
        except Exception as e:
            print(f"  ✗ Error loading {exp_dir.name}: {e}")

    # Append to CSV
    if rows_to_add:
        print(f"\nAppending {len(rows_to_add)} rows to CSV...")
        append_to_csv(csv_path, rows_to_add)
        print("✓ Done!")

        # Verify
        new_count = len(load_existing_experiments(csv_path))
        print(f"\nCSV now contains {new_count} experiments (was {len(existing_experiments)})")
    else:
        print("\nNo valid data to add.")


if __name__ == '__main__':
    main()
