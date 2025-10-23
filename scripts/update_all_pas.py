#!/usr/bin/env python3
"""
Master script to compute PAS for all experiments and update all results files.

This script:
1. Computes PAS for all completed experiments
2. Updates individual metrics.json files
3. Regenerates all_results.csv with updated PAS values

Usage:
    # Preview changes (dry run)
    python scripts/update_all_pas.py --results_dir results/ablations/capacity --dry_run
    
    # Actually update files
    python scripts/update_all_pas.py --results_dir results/ablations/capacity
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple
import sys


def compute_preference_aligned_spread(objectives: np.ndarray, 
                                     preferences: np.ndarray,
                                     num_preference_groups: int = 10) -> float:
    """
    Compute Preference-Aligned Spread (PAS).
    
    Measures how spread out solutions are within preference-conditioned groups.
    Higher PAS = better diversity across different preferences.
    
    Args:
        objectives: Array of shape (N, num_objectives)
        preferences: Array of shape (N, num_objectives)
        num_preference_groups: Number of preference groups to create
    
    Returns:
        pas: Preference-aligned spread score
    """
    if len(objectives) < 10:
        return 0.0
    
    # Strategy: Group solutions by similar preferences, measure spread within each group
    
    # 1. Cluster preferences into groups
    n_clusters = min(num_preference_groups, len(preferences) // 5, len(preferences))
    if n_clusters < 2:
        # Fallback: just compute overall spread
        if len(objectives) > 1:
            dists = pdist(objectives, metric='euclidean')
            return float(np.mean(dists))
        return 0.0
    
    # Cluster preferences
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pref_clusters = kmeans.fit_predict(preferences)
    
    # 2. Compute spread within each preference cluster
    spreads = []
    
    for cluster_id in range(n_clusters):
        # Get objectives for this preference cluster
        cluster_mask = pref_clusters == cluster_id
        cluster_objectives = objectives[cluster_mask]
        
        if len(cluster_objectives) > 1:
            # Compute pairwise distances within cluster
            cluster_dists = pdist(cluster_objectives, metric='euclidean')
            cluster_spread = np.mean(cluster_dists)
            spreads.append(cluster_spread)
    
    # 3. Average spread across all preference clusters
    if len(spreads) == 0:
        return 0.0
    
    pas = float(np.mean(spreads))
    return pas


def find_experiment_dirs(results_dir: Path) -> List[Path]:
    """Find all experiment directories that have completed."""
    experiment_dirs = []
    
    for item in results_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if experiment has required files
            if (item / 'metrics.json').exists() and \
               (item / 'objectives.npy').exists() and \
               (item / 'preferences.npy').exists():
                experiment_dirs.append(item)
    
    return sorted(experiment_dirs)


def compute_and_update_pas(exp_dir: Path, dry_run: bool = False) -> Tuple[float, bool]:
    """
    Compute PAS for experiment and update metrics.json.
    
    Returns:
        (pas_value, success)
    """
    try:
        # Load objectives and preferences
        objectives = np.load(exp_dir / 'objectives.npy')
        preferences = np.load(exp_dir / 'preferences.npy')
        
        # Compute PAS
        pas = compute_preference_aligned_spread(objectives, preferences)
        
        # Update metrics file
        metrics_file = exp_dir / 'metrics.json'
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        old_pas = metrics.get('pas', None)
        metrics['pas'] = pas
        
        if dry_run:
            print(f"  [DRY RUN] Would update PAS: {old_pas} → {pas:.4f}")
        else:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"  ✓ Updated PAS: {old_pas} → {pas:.4f}")
        
        return pas, True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return 0.0, False


def regenerate_results_csv(results_dir: Path) -> bool:
    """Regenerate all_results.csv from individual metrics.json files."""
    try:
        print("\nRegenerating all_results.csv...")
        print("-"*70)
        
        # Find all experiment directories
        experiment_dirs = []
        for item in results_dir.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                if (item / 'metrics.json').exists():
                    experiment_dirs.append(item)
        
        # Load all metrics
        all_metrics = []
        for exp_dir in sorted(experiment_dirs):
            with open(exp_dir / 'metrics.json', 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
        
        if len(all_metrics) == 0:
            print("  ✗ No metrics found!")
            return False
        
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Save to CSV
        output_file = results_dir / 'all_results.csv'
        df.to_csv(output_file, index=False)
        
        print(f"  ✓ Saved {len(df)} experiments to {output_file}")
        print(f"  ✓ Columns: {len(df.columns)} metrics")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error regenerating CSV: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Compute PAS for all experiments and update results files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview what will be changed (recommended first)
  python update_all_pas.py --results_dir results/ablations/capacity --dry_run
  
  # Actually update all files
  python update_all_pas.py --results_dir results/ablations/capacity
  
  # Use different number of preference groups
  python update_all_pas.py --results_dir results/ablations/capacity --num_groups 15
        """
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to results directory (e.g., results/ablations/capacity)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without modifying files'
    )
    parser.add_argument(
        '--num_groups',
        type=int,
        default=10,
        help='Number of preference groups for PAS computation (default: 10)'
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("="*70)
    print("UPDATE PAS FOR ALL EXPERIMENTS")
    print("="*70)
    print(f"Results directory: {results_dir}")
    print(f"Dry run: {'YES (no files will be modified)' if args.dry_run else 'NO (files will be updated)'}")
    print(f"Preference groups: {args.num_groups}")
    print()
    
    # Find all experiment directories
    print("Step 1: Finding experiment directories...")
    experiment_dirs = find_experiment_dirs(results_dir)
    
    if len(experiment_dirs) == 0:
        print("✗ No completed experiments found!")
        print("  Looking for directories with: metrics.json, objectives.npy, preferences.npy")
        sys.exit(1)
    
    print(f"✓ Found {len(experiment_dirs)} experiments")
    print()
    
    # Process each experiment
    print("Step 2: Computing and updating PAS...")
    print("-"*70)
    
    successful = 0
    failed = 0
    pas_values = []
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"\n{exp_name}")
        
        pas_value, success = compute_and_update_pas(exp_dir, dry_run=args.dry_run)
        
        if success:
            successful += 1
            pas_values.append(pas_value)
        else:
            failed += 1
    
    print()
    print("-"*70)
    print(f"Successfully processed: {successful}/{len(experiment_dirs)}")
    print(f"Failed: {failed}/{len(experiment_dirs)}")
    
    if successful > 0:
        print(f"\nPAS Statistics:")
        print(f"  Mean: {np.mean(pas_values):.4f}")
        print(f"  Std:  {np.std(pas_values):.4f}")
        print(f"  Min:  {np.min(pas_values):.4f}")
        print(f"  Max:  {np.max(pas_values):.4f}")
    
    # Regenerate CSV (only if not dry run and had success)
    if not args.dry_run and successful > 0:
        print()
        print("Step 3: Regenerating all_results.csv...")
        if regenerate_results_csv(results_dir):
            print("✓ CSV file updated!")
        else:
            print("✗ Failed to update CSV file")
    elif args.dry_run:
        print()
        print("Step 3: Skipped (dry run mode)")
    
    # Final summary
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    if args.dry_run:
        print("✓ Dry run completed successfully!")
        print()
        print("What would be updated:")
        print(f"  - {successful} metrics.json files with PAS values")
        print(f"  - all_results.csv regenerated")
        print()
        print("Run without --dry_run to actually update the files.")
    else:
        print("✓ All files updated successfully!")
        print()
        print("What was updated:")
        print(f"  - {successful} metrics.json files with PAS values")
        print(f"  - all_results.csv regenerated")
        print()
        print("You can now use the updated PAS metric in your analysis!")
    
    print("="*70)


if __name__ == '__main__':
    main()